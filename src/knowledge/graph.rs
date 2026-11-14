//! The knowledge graph: nodes, edges, provenance, persistence.
//!
//! Boundary: this file knows the data model and nothing else. It does not
//! parse text (`ingest`), score text (`index`) or decide what is relevant
//! (`retrieve`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub type NodeId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    /// One ingested source (a file).
    Document,
    /// A heading inside a document.
    Section,
    /// A passage of a document: what retrieval returns.
    Chunk,
    /// A named thing (an identifier, a concept) that passages mention.
    Term,
    /// Something the user or the agent wrote down.
    Note,
}

impl NodeKind {
    pub fn label(self) -> &'static str {
        match self {
            NodeKind::Document => "document",
            NodeKind::Section => "section",
            NodeKind::Chunk => "passage",
            NodeKind::Term => "term",
            NodeKind::Note => "note",
        }
    }

    /// Whether this kind carries text worth handing to a model.
    pub fn is_content(self) -> bool {
        matches!(self, NodeKind::Chunk | NodeKind::Note)
    }
}

/// How far a piece of knowledge can be relied on, lowest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trust {
    /// Extracted by a model, or read from an outside source such as a web
    /// page; may be wrong.
    Inferred,
    /// Written by the user or the agent from experience.
    Experience,
    /// Taken as-is from a source that can be re-read.
    Verified,
}

impl Trust {
    pub fn label(self) -> &'static str {
        match self {
            Trust::Inferred => "inferred",
            Trust::Experience => "experience",
            Trust::Verified => "verified",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Relation {
    /// Document → section → chunk.
    Contains,
    /// A chunk → the chunk after it in the same document.
    Next,
    /// A chunk or note → a term it names.
    Mentions,
    /// A chunk → a document it points at.
    References,
    /// Term → term, with a label a model gave it ("depends on", "replaces").
    Related(String),
}

impl Relation {
    pub fn label(&self) -> &str {
        match self {
            Relation::Contains => "contains",
            Relation::Next => "next",
            Relation::Mentions => "mentions",
            Relation::References => "references",
            Relation::Related(label) => label,
        }
    }
}

/// Where a node came from. Every node has one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// A file path, `user`, `agent`, `model` or `derived`.
    pub source: String,
    /// Unix seconds when it was recorded.
    pub at: u64,
    pub trust: Trust,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    /// The namespace this node lives in. Retrieval never leaves the spaces
    /// it was given.
    pub space: String,
    pub title: String,
    pub text: String,
    pub provenance: Provenance,
    /// The document this node belongs to, if any.
    #[serde(default)]
    pub root: Option<NodeId>,
    /// Paths this node points at, resolved to `References` edges by `relink`.
    #[serde(default)]
    pub refs: Vec<String>,
    /// For documents: a hash of the ingested text, to skip unchanged files.
    #[serde(default)]
    pub hash: Option<u64>,
    /// How it has fared: the tasks it was given to and how they were rated.
    /// Kept for the lessons the agent writes for itself.
    #[serde(default, skip_serializing_if = "Record::is_empty")]
    pub record: Record,
    /// Set when it was taken out of use, with the reason. A retired node is
    /// kept but never found by a lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retired: Option<String>,
    /// Notes that look like this one but were judged to say something
    /// else, so the question is not asked again.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub apart: Vec<NodeId>,
    /// The tasks that each, on their own, drew this same lesson. More than
    /// one is evidence that it is general and not one case's detail.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drawn_by: Vec<u32>,
}

/// What became of the tasks a piece of knowledge was given to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Tasks it was put in front of.
    #[serde(default)]
    pub given: u32,
    /// Of those the user rated, how many were good and how many bad.
    #[serde(default)]
    pub good: u32,
    #[serde(default)]
    pub bad: u32,
    /// Of those whose own checks told how they went (a test or build that
    /// passed or failed), how many passed and how many failed.
    #[serde(default)]
    pub passed: u32,
    #[serde(default)]
    pub failed: u32,
}

impl Record {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub relation: Relation,
    pub trust: Trust,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Graph {
    nodes: BTreeMap<NodeId, Node>,
    edges: Vec<Edge>,
    next_id: NodeId,
    /// Node → indices into `edges` that touch it.
    #[serde(skip)]
    adjacency: HashMap<NodeId, Vec<usize>>,
    #[serde(skip)]
    seen: HashSet<(NodeId, NodeId, Relation)>,
}

/// A node about to be added; the graph assigns the id.
pub struct NewNode {
    pub kind: NodeKind,
    pub space: String,
    pub title: String,
    pub text: String,
    pub provenance: Provenance,
    pub root: Option<NodeId>,
    pub refs: Vec<String>,
    pub hash: Option<u64>,
}

impl Graph {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let mut graph: Graph =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        graph.reindex();
        Ok(graph)
    }

    /// Writes the graph, via a temp file so a crash cannot truncate it.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn reindex(&mut self) {
        self.adjacency.clear();
        self.seen.clear();
        for (i, e) in self.edges.iter().enumerate() {
            self.adjacency.entry(e.from).or_default().push(i);
            self.adjacency.entry(e.to).or_default().push(i);
            self.seen.insert((e.from, e.to, e.relation.clone()));
        }
    }

    pub fn add_node(&mut self, new: NewNode) -> NodeId {
        self.next_id += 1;
        let id = self.next_id;
        self.nodes.insert(
            id,
            Node {
                id,
                kind: new.kind,
                space: new.space,
                title: new.title,
                text: new.text,
                provenance: new.provenance,
                root: new.root,
                refs: new.refs,
                hash: new.hash,
                record: Record::default(),
                retired: None,
                apart: Vec::new(),
                drawn_by: Vec::new(),
            },
        );
        id
    }

    /// Adds an edge unless the same one exists or an end is missing.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, relation: Relation, trust: Trust) {
        if from == to || !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return;
        }
        if !self.seen.insert((from, to, relation.clone())) {
            return;
        }
        let index = self.edges.len();
        self.edges.push(Edge {
            from,
            to,
            relation,
            trust,
        });
        self.adjacency.entry(from).or_default().push(index);
        self.adjacency.entry(to).or_default().push(index);
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Edges touching `id`, each with the node at the other end and whether
    /// the edge points away from `id`.
    pub fn neighbours(&self, id: NodeId) -> impl Iterator<Item = (NodeId, &Edge, bool)> {
        self.adjacency
            .get(&id)
            .into_iter()
            .flatten()
            .map(move |&i| {
                let e = &self.edges[i];
                if e.from == id {
                    (e.to, e, true)
                } else {
                    (e.from, e, false)
                }
            })
    }

    pub fn degree(&self, id: NodeId) -> usize {
        self.adjacency.get(&id).map_or(0, Vec::len)
    }

    /// The document ingested from `source` into `space`, if there is one.
    pub fn document(&self, space: &str, source: &str) -> Option<&Node> {
        self.nodes.values().find(|n| {
            n.kind == NodeKind::Document && n.space == space && n.provenance.source == source
        })
    }

    /// Removes a document and everything that belongs to it, then any term
    /// left with nothing pointing at it. Returns how many nodes went.
    pub fn remove_document(&mut self, doc: NodeId) -> usize {
        let before = self.nodes.len();
        self.nodes.retain(|id, n| *id != doc && n.root != Some(doc));
        self.drop_dangling_edges();
        let orphans: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::Term && self.degree(n.id) == 0)
            .map(|n| n.id)
            .collect();
        for id in orphans {
            self.nodes.remove(&id);
        }
        before - self.nodes.len()
    }

    fn drop_dangling_edges(&mut self) {
        let nodes = &self.nodes;
        self.edges
            .retain(|e| nodes.contains_key(&e.from) && nodes.contains_key(&e.to));
        self.reindex();
    }

    /// Recomputes every `References` edge from the paths nodes point at, so
    /// the result does not depend on the order files were ingested in.
    pub fn relink(&mut self) {
        self.edges.retain(|e| e.relation != Relation::References);
        self.reindex();
        let docs: Vec<(NodeId, String, String)> = self
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::Document)
            .map(|n| (n.id, n.space.clone(), n.provenance.source.clone()))
            .collect();
        let mut links = Vec::new();
        for n in self.nodes.values().filter(|n| !n.refs.is_empty()) {
            for r in &n.refs {
                let wanted = r.trim_start_matches("./");
                for (doc, space, source) in &docs {
                    let matches = source == wanted || source.ends_with(&format!("/{wanted}"));
                    if matches && *space == n.space && Some(*doc) != n.root {
                        links.push((n.id, *doc));
                    }
                }
            }
        }
        for (from, to) in links {
            self.add_edge(from, to, Relation::References, Trust::Verified);
        }
    }
}

/// FNV-1a, to tell whether a source changed. Stable across Rust versions,
/// unlike the standard hasher.
pub fn fingerprint(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}
