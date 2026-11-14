//! Graph lookup: find passages for a query.
//!
//! 1. **Seed**: the lexical index finds nodes whose words match the query,
//!    and, when the caller supplies the question as a vector, the meaning
//!    index finds passages close to it in meaning even if they share no
//!    word with it. A node found both ways counts for both.
//! 2. **Expand**: relevance spreads from the seeds along edges, weaker with
//!    every hop and through busy hubs, so passages connected to several
//!    seeds rise.
//! 3. **Pack**: the best passages are taken until a token budget is spent.
//!
//! Boundary: pure and deterministic over a graph and an index. It never
//! reads files or calls a model. The `Query` is the complete statement of
//! what a lookup may touch: nothing outside its spaces, below its trust
//! level, beyond its hop count or over its budget is returned or even
//! walked through.

use std::collections::BTreeMap;

use super::graph::{Graph, NodeId, Relation, Trust};
use super::index::Index;
use super::vectors::Vectors;

/// A passage must be at least this close in meaning to start a lookup.
/// Unrelated text scores about 0.3 to 0.4 with the model in use, related
/// text mostly 0.6 and up; a stray short snippet can reach about 0.52.
pub const MIN_CLOSENESS: f32 = 0.55;
/// How much a match on unconfirmed knowledge counts, against 1 for the rest.
const UNCONFIRMED_WEIGHT: f32 = 0.8;
/// Closeness at which a meaning match counts for nothing, for scaling.
const CLOSENESS_FLOOR: f32 = 0.4;

/// How many index matches start a lookup.
const SEEDS: usize = 12;
/// Relevance below this is not passed on.
const FLOOR: f32 = 0.02;

#[derive(Clone, Debug)]
pub struct Query {
    pub text: String,
    /// Spaces the lookup may use. `None` means every space.
    pub spaces: Option<Vec<String>>,
    /// Nodes and edges trusted less than this are ignored.
    pub min_trust: Trust,
    /// How many edges away from a seed a passage may be. 0 is plain search.
    pub max_hops: u8,
    pub max_results: usize,
    /// Upper bound on the text returned, in estimated tokens.
    pub token_budget: u32,
}

impl Query {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            spaces: None,
            min_trust: Trust::Inferred,
            max_hops: 2,
            max_results: 6,
            token_budget: 1_500,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub node: NodeId,
    pub score: f32,
    /// How the lookup got here: matched words, or the chain from a seed.
    pub why: String,
    /// 0 for a direct match.
    pub hops: u8,
    pub tokens: u32,
    /// The route from the question to this passage: first the node whose
    /// words matched, then each node reached over a link, ending here.
    pub path: Vec<Step>,
    /// The question's words found in the first node of `path`.
    pub matched: Vec<String>,
}

/// One node on the way to a passage.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub node: NodeId,
    /// The link followed to get here; `None` for the starting node.
    pub relation: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Retrieval {
    pub hits: Vec<Hit>,
    pub tokens: u32,
    /// Index matches the lookup started from.
    pub seeds: usize,
    /// Nodes relevance reached, including connecting ones.
    pub reached: usize,
    /// Passages that qualified but did not fit the limits.
    pub dropped: usize,
    /// The question reduced to the words that were searched for.
    pub words: Vec<String>,
    /// The limits the lookup ran under, in words.
    pub limits: String,
}

/// How much relevance crosses an edge, by relation and direction.
fn carry(relation: &Relation, outgoing: bool) -> f32 {
    match (relation, outgoing) {
        (Relation::Mentions, _) => 0.6,
        (Relation::Related(_), _) => 0.5,
        (Relation::References, true) => 0.5,
        (Relation::References, false) => 0.3,
        (Relation::Next, _) => 0.35,
        (Relation::Contains, true) => 0.3,
        (Relation::Contains, false) => 0.25,
    }
}

fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() as u32).div_ceil(4)
}

/// `meaning` is the meaning index and the question as a vector, when there
/// is a model to make one; without it the lookup is by words alone.
pub fn retrieve(
    graph: &Graph,
    index: &Index,
    meaning: Option<(&Vectors, &[f32])>,
    query: &Query,
) -> Retrieval {
    let allowed = |id: NodeId| {
        graph.node(id).is_some_and(|n| {
            n.retired.is_none()
                && n.provenance.trust >= query.min_trust
                && query
                    .spaces
                    .as_ref()
                    .is_none_or(|spaces| spaces.contains(&n.space))
        })
    };

    // 1. Seed.
    let seeds = index.search(&query.text, allowed, SEEDS);
    let top = seeds.first().map_or(1.0, |s| s.score).max(f32::EPSILON);
    // node → (relevance, hops from a seed, how it was reached)
    let mut reached: BTreeMap<NodeId, (f32, u8, String)> = BTreeMap::new();
    // node → the node it was first reached from, and over which link
    let mut came_from: BTreeMap<NodeId, (NodeId, String)> = BTreeMap::new();
    let mut matched: BTreeMap<NodeId, Vec<String>> = BTreeMap::new();
    let mut frontier: BTreeMap<NodeId, f32> = BTreeMap::new();
    // What is not confirmed (a model's inference, a web page, a lesson no
    // rating has backed yet) starts a lookup with less weight than what is.
    let weight = |id: NodeId| match graph.node(id).map(|n| n.provenance.trust) {
        Some(Trust::Inferred) => UNCONFIRMED_WEIGHT,
        _ => 1.0,
    };
    for s in &seeds {
        matched.insert(s.node, s.matched.clone());
        let score = s.score / top * weight(s.node);
        reached.insert(
            s.node,
            (score, 0, format!("matched: {}", s.matched.join(", "))),
        );
        frontier.insert(s.node, score);
    }
    // Passages close in meaning start the lookup too.
    let mut by_meaning = 0;
    if let Some((vectors, asked)) = meaning {
        for (node, closeness) in vectors.nearest(asked, allowed, SEEDS) {
            if closeness < MIN_CLOSENESS {
                break;
            }
            by_meaning += 1;
            let score = ((closeness - CLOSENESS_FLOOR) / (1.0 - CLOSENESS_FLOOR)).clamp(0.0, 1.0)
                * weight(node);
            let said = format!("close in meaning ({closeness:.2})");
            matched
                .entry(node)
                .or_default()
                .push(format!("≈ meaning {closeness:.2}"));
            match reached.get_mut(&node) {
                Some(entry) => {
                    entry.0 += score;
                    entry.2 = format!("{}; {said}", entry.2);
                }
                None => {
                    reached.insert(node, (score, 0, said));
                }
            }
            *frontier.entry(node).or_default() += score;
        }
    }

    // 2. Expand.
    for hop in 1..=query.max_hops {
        let mut next: BTreeMap<NodeId, f32> = BTreeMap::new();
        for (&from, &relevance) in &frontier {
            // A node with many edges spreads itself thinner.
            let spread = 1.0 + (graph.degree(from).max(1) as f32).ln();
            for (to, edge, outgoing) in graph.neighbours(from) {
                if edge.trust < query.min_trust || !allowed(to) {
                    continue;
                }
                let trust = if edge.trust == Trust::Inferred {
                    0.8
                } else {
                    1.0
                };
                let gain = relevance * carry(&edge.relation, outgoing) * trust / spread;
                if gain < FLOOR {
                    continue;
                }
                *next.entry(to).or_default() += gain;
                let from_title = graph.node(from).map_or("", |n| n.title.as_str());
                let via = format!("{} → \"{from_title}\"", edge.relation.label());
                if !reached.contains_key(&to) {
                    came_from.insert(to, (from, edge.relation.label().to_string()));
                }
                let entry = reached.entry(to).or_insert((0.0, hop, via));
                entry.0 += gain;
            }
        }
        frontier = next;
    }

    // 3. Pack.
    let mut passages: Vec<(NodeId, f32, u8, &String)> = reached
        .iter()
        .filter(|(id, _)| graph.node(**id).is_some_and(|n| n.kind.is_content()))
        .map(|(id, (score, hops, why))| (*id, *score, *hops, why))
        .collect();
    passages.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut words = super::index::tokens(&query.text);
    words.sort();
    words.dedup();
    let mut out = Retrieval {
        seeds: seeds.len() + by_meaning,
        reached: reached.len(),
        words,
        limits: format!(
            "{} · up to {} hop{} · {} passages · {} tokens · {} · {}",
            if meaning.is_some() {
                "by words and by meaning"
            } else {
                "by words only"
            },
            query.max_hops,
            if query.max_hops == 1 { "" } else { "s" },
            query.max_results,
            query.token_budget,
            match &query.spaces {
                Some(spaces) => format!("spaces: {}", spaces.join(", ")),
                None => "all spaces".into(),
            },
            match query.min_trust {
                Trust::Inferred => "any trust level",
                Trust::Experience => "experience or verified only",
                Trust::Verified => "verified only",
            }
        ),
        ..Default::default()
    };
    for (id, score, hops, why) in passages {
        let node = graph.node(id).expect("reached nodes exist");
        let tokens = estimate_tokens(&node.text) + estimate_tokens(&node.title);
        if out.hits.len() >= query.max_results || out.tokens + tokens > query.token_budget {
            out.dropped += 1;
            continue;
        }
        out.tokens += tokens;
        // Walk back to the node whose words matched.
        let mut path = vec![Step {
            node: id,
            relation: None,
        }];
        let mut at = id;
        while let Some((from, relation)) = came_from.get(&at) {
            if path.len() > query.max_hops as usize {
                break;
            }
            path[0].relation = Some(relation.clone());
            path.insert(
                0,
                Step {
                    node: *from,
                    relation: None,
                },
            );
            at = *from;
        }
        out.hits.push(Hit {
            matched: matched.get(&path[0].node).cloned().unwrap_or_default(),
            path,
            node: id,
            score,
            why: if hops == 0 {
                why.clone()
            } else {
                format!(
                    "{hops} hop{} away, via {why}",
                    if hops == 1 { "" } else { "s" }
                )
            },
            hops,
            tokens,
        });
    }
    out
}
