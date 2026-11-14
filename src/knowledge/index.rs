//! Lexical index: BM25 over node text.
//!
//! Boundary: this file turns a query string into scored node ids. It knows
//! nothing about edges, scopes or trust; the caller passes a filter.

use std::collections::HashMap;

use super::graph::{Graph, NodeId, NodeKind};

const K1: f32 = 1.4;
const B: f32 = 0.75;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with",
];

/// Splits text into lowercase search tokens. `PolicyStore` gives
/// `policystore`, `policy` and `store`; `max_hops` gives `max` and `hops`.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.chars().count() < 2 {
            continue;
        }
        let lower = word.to_lowercase();
        if STOPWORDS.contains(&lower.as_str()) {
            continue;
        }
        // camelCase parts, in addition to the whole word.
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut previous_lower = false;
        for c in word.chars() {
            if c.is_uppercase() && previous_lower && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            previous_lower = c.is_lowercase();
            current.push(c);
        }
        parts.push(current);
        if parts.len() > 1 {
            out.extend(
                parts
                    .iter()
                    .filter(|p| p.chars().count() >= 2)
                    .map(|p| p.to_lowercase()),
            );
        }
        out.push(lower);
    }
    out
}

pub struct Scored {
    pub node: NodeId,
    pub score: f32,
    /// Query tokens found in the node.
    pub matched: Vec<String>,
}

#[derive(Default)]
pub struct Index {
    /// token → (node, occurrences)
    postings: HashMap<String, Vec<(NodeId, u32)>>,
    lengths: HashMap<NodeId, u32>,
    average_length: f32,
}

impl Index {
    /// Indexes every node that has words to search: passages and notes by
    /// their text, terms and sections by their title. Titles count double.
    pub fn build(graph: &Graph) -> Self {
        let mut index = Index::default();
        for node in graph.nodes() {
            // What was taken out of use is kept, but not found.
            if node.kind == NodeKind::Document || node.retired.is_some() {
                continue;
            }
            let mut counts: HashMap<String, u32> = HashMap::new();
            for t in tokens(&node.title) {
                *counts.entry(t).or_default() += 2;
            }
            for t in tokens(&node.text) {
                *counts.entry(t).or_default() += 1;
            }
            let length: u32 = counts.values().sum();
            if length == 0 {
                continue;
            }
            index.lengths.insert(node.id, length);
            for (token, n) in counts {
                index.postings.entry(token).or_default().push((node.id, n));
            }
        }
        let total: u32 = index.lengths.values().sum();
        index.average_length = total as f32 / index.lengths.len().max(1) as f32;
        index
    }

    /// The best `limit` nodes for `query` among those `allow` accepts,
    /// best first; ties go to the lower node id so results are repeatable.
    pub fn search(&self, query: &str, allow: impl Fn(NodeId) -> bool, limit: usize) -> Vec<Scored> {
        let mut wanted = tokens(query);
        wanted.sort();
        wanted.dedup();
        let documents = self.lengths.len() as f32;
        let mut found: HashMap<NodeId, (f32, Vec<String>)> = HashMap::new();
        for token in &wanted {
            let Some(postings) = self.postings.get(token) else {
                continue;
            };
            let containing = postings.len() as f32;
            let idf = ((documents - containing + 0.5) / (containing + 0.5) + 1.0).ln();
            for &(node, count) in postings {
                if !allow(node) {
                    continue;
                }
                let tf = count as f32;
                let length = self.lengths[&node] as f32;
                let norm =
                    tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * length / self.average_length));
                let entry = found.entry(node).or_default();
                entry.0 += idf * norm;
                entry.1.push(token.clone());
            }
        }
        let mut scored: Vec<Scored> = found
            .into_iter()
            .map(|(node, (score, matched))| Scored {
                node,
                score,
                matched,
            })
            .collect();
        scored.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.node.cmp(&b.node)));
        scored.truncate(limit);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_identifiers_and_drop_noise() {
        assert_eq!(
            tokens("the PolicyStore is"),
            ["policy", "store", "policystore"]
        );
        assert_eq!(tokens("max_hops = 2"), ["max", "hops"]);
        assert_eq!(tokens("A b of"), Vec::<String>::new());
    }
}
