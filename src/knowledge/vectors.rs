//! Meaning index: one vector per passage, searched by closeness.
//!
//! Boundary: this file stores vectors and turns a query vector into scored
//! node ids. It never calls a model (`embed.rs` makes the vectors), and it
//! knows nothing about edges, spaces or trust; the caller passes a filter.
//!
//! Kept in `vectors.bin` beside the graph, apart from it: the graph is the
//! knowledge, this is derived from it and can always be rebuilt. A vector
//! is tied to the text it was made from (a fingerprint), so a passage that
//! changed is embedded again, and to the model that made it, so a change of
//! model starts the index afresh.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::graph::NodeId;

const MAGIC: &[u8; 4] = b"CSV1";

/// node → (fingerprint of the text embedded, unit-length vector)
type ByNode = HashMap<NodeId, (u64, Vec<f32>)>;

#[derive(Default)]
pub struct Vectors {
    path: Option<PathBuf>,
    /// The model the vectors were made with.
    model: String,
    by_node: ByNode,
    dirty: bool,
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let length = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if length > 0.0 {
        for x in &mut v {
            *x /= length;
        }
    }
    v
}

impl Vectors {
    /// Reads the saved vectors. A missing or unreadable file is an empty
    /// index: it is rebuilt from the graph.
    pub fn load(path: &Path) -> Self {
        let mut vectors = Self {
            path: Some(path.to_path_buf()),
            ..Self::default()
        };
        if let Ok(bytes) = std::fs::read(path)
            && let Some((model, by_node)) = Self::decode(&bytes)
        {
            vectors.model = model;
            vectors.by_node = by_node;
        }
        vectors
    }

    fn decode(bytes: &[u8]) -> Option<(String, ByNode)> {
        let mut at = 0;
        let mut take = |n: usize| {
            let part = bytes.get(at..at + n)?;
            at += n;
            Some(part)
        };
        if take(4)? != MAGIC {
            return None;
        }
        let u32_of = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap_or_default()) as usize;
        let u64_of = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap_or_default());
        let name_len = u32_of(take(4)?);
        let model = String::from_utf8(take(name_len)?.to_vec()).ok()?;
        let dim = u32_of(take(4)?);
        let count = u32_of(take(4)?);
        if dim == 0 || dim > 8_192 {
            return None;
        }
        let mut by_node = HashMap::with_capacity(count);
        for _ in 0..count {
            let node: NodeId = u64_of(take(8)?);
            let hash = u64_of(take(8)?);
            let vector = take(dim * 4)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            by_node.insert(node, (hash, vector));
        }
        Some((model, by_node))
    }

    /// Writes the vectors, if any changed since the last save.
    pub fn save(&mut self) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if !self.dirty {
            return Ok(());
        }
        let dim = self.by_node.values().next().map_or(0, |(_, v)| v.len());
        let mut bytes = Vec::with_capacity(16 + self.by_node.len() * (16 + dim * 4));
        bytes.extend(MAGIC);
        bytes.extend((self.model.len() as u32).to_le_bytes());
        bytes.extend(self.model.as_bytes());
        bytes.extend((dim as u32).to_le_bytes());
        bytes.extend((self.by_node.len() as u32).to_le_bytes());
        // In node order, so the same index is the same file.
        let mut nodes: Vec<&NodeId> = self.by_node.keys().collect();
        nodes.sort();
        for node in nodes {
            let (hash, vector) = &self.by_node[node];
            if vector.len() != dim {
                continue;
            }
            bytes.extend(node.to_le_bytes());
            bytes.extend(hash.to_le_bytes());
            for x in vector {
                bytes.extend(x.to_le_bytes());
            }
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        self.dirty = false;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.by_node.len()
    }

    /// Says which model makes the vectors from now on. Vectors made by
    /// another model cannot be compared with its, so they are dropped.
    pub fn made_by(&mut self, model: &str) {
        if self.model != model {
            self.dirty |= !self.by_node.is_empty() || !self.model.is_empty();
            self.by_node.clear();
            self.model = model.to_string();
        }
    }

    /// A node's vector (unit length), if it has one.
    pub fn get(&self, node: NodeId) -> Option<&[f32]> {
        self.by_node.get(&node).map(|(_, v)| v.as_slice())
    }

    /// Whether `node` has a vector made from text with this fingerprint.
    pub fn has(&self, node: NodeId, hash: u64) -> bool {
        self.by_node.get(&node).is_some_and(|(h, _)| *h == hash)
    }

    pub fn set(&mut self, node: NodeId, hash: u64, vector: Vec<f32>) {
        self.by_node.insert(node, (hash, unit(vector)));
        self.dirty = true;
    }

    /// Drops the vectors of nodes that are gone. Returns how many.
    pub fn keep_only(&mut self, exists: impl Fn(NodeId) -> bool) -> usize {
        let before = self.by_node.len();
        self.by_node.retain(|node, _| exists(*node));
        let gone = before - self.by_node.len();
        self.dirty |= gone > 0;
        gone
    }

    /// The `limit` nodes closest in meaning to `query` among those
    /// `allowed`, closest first, with their closeness (cosine, 1 is the
    /// same direction). Ties go to the lower node id.
    pub fn nearest(
        &self,
        query: &[f32],
        allowed: impl Fn(NodeId) -> bool,
        limit: usize,
    ) -> Vec<(NodeId, f32)> {
        let query = unit(query.to_vec());
        let mut scored: Vec<(NodeId, f32)> = self
            .by_node
            .iter()
            .filter(|(node, (_, v))| v.len() == query.len() && allowed(**node))
            .map(|(node, (_, v))| (*node, v.iter().zip(&query).map(|(a, b)| a * b).sum()))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(limit);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_closest_in_direction_come_first_and_only_those_allowed() {
        let mut v = Vectors::default();
        v.made_by("test");
        v.set(1, 10, vec![1.0, 0.0]);
        v.set(2, 20, vec![3.0, 3.0]);
        v.set(3, 30, vec![0.0, 2.0]);
        v.set(4, 40, vec![1.0, 0.0, 0.0]);
        let near = v.nearest(&[2.0, 0.1], |_| true, 10);
        // Length does not matter, direction does; a vector of another size
        // is not compared.
        assert_eq!(near.iter().map(|(n, _)| *n).collect::<Vec<_>>(), [1, 2, 3]);
        assert!(near[0].1 > 0.99 && near[2].1 < 0.1);
        let near = v.nearest(&[2.0, 0.1], |n| n != 1, 1);
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].0, 2);
        assert!(v.has(2, 20) && !v.has(2, 21) && !v.has(9, 20));
        assert_eq!(v.keep_only(|n| n != 3), 1);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn vectors_survive_a_restart_and_a_change_of_model_starts_afresh() {
        let dir = std::env::temp_dir().join(format!("cs-vectors-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("vectors.bin");
        let mut v = Vectors::load(&path);
        assert_eq!(v.len(), 0);
        v.made_by("model-a");
        v.set(7, 70, vec![0.6, 0.8]);
        v.set(8, 80, vec![1.0, 0.0]);
        v.save().unwrap();

        let mut again = Vectors::load(&path);
        assert_eq!(again.len(), 2);
        assert!(again.has(7, 70));
        let near = again.nearest(&[0.6, 0.8], |_| true, 1);
        assert_eq!(near[0].0, 7);
        assert!((near[0].1 - 1.0).abs() < 1e-5);
        // The same model keeps them; another one cannot use them.
        again.made_by("model-a");
        assert_eq!(again.len(), 2);
        again.made_by("model-b");
        assert_eq!(again.len(), 0);
        again.save().unwrap();
        assert_eq!(Vectors::load(&path).len(), 0);

        // A damaged file is an empty index, not an error.
        std::fs::write(&path, b"CSV1 and then nonsense").unwrap();
        assert_eq!(Vectors::load(&path).len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
