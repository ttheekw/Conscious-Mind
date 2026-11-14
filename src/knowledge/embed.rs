//! Turning text into vectors of meaning. The only part of the store that
//! runs the embedding model.
//!
//! The model is a small one that runs on this machine (`Local`): nothing
//! that is embedded leaves it. Its files are downloaded once, on first
//! use, into the cache folder; after that no network is needed. Everything
//! else sees only the `Embedder` trait, so the lookup is tested without
//! the model.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

/// Characters of a passage that are embedded. The model reads about 512
/// tokens; more would be cut off anyway.
pub const MAX_EMBEDDED_CHARS: usize = 2_000;

pub trait Embedder: Send + Sync {
    /// Names the model, so vectors made by another are not mixed with its.
    fn name(&self) -> &str;
    /// One vector per passage.
    fn passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
    /// The vector for a question, to compare passages with.
    fn query(&self, text: &str) -> Result<Vec<f32>, String>;
}

/// The model that runs on this machine.
pub struct Local {
    model: Mutex<TextEmbedding>,
}

/// What this model wants in front of a question (not in front of passages).
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

impl Local {
    /// Where the model's files are kept: a cache, shared by every data
    /// folder, since it is the same model for all of them.
    pub fn default_dir() -> Option<PathBuf> {
        let cache = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
        Some(cache.join("conscious").join("models"))
    }

    /// Loads the model, downloading its files first if they are not in
    /// `dir` yet (about 35 MB, once).
    pub fn load(dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let options = TextInitOptions::new(EmbeddingModel::BGESmallENV15Q)
            .with_cache_dir(dir.to_path_buf())
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).map_err(|e| e.to_string())?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

impl Embedder for Local {
    fn name(&self) -> &str {
        "bge-small-en-v1.5-q"
    }

    fn passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let cut: Vec<String> = texts
            .iter()
            .map(|t| t.chars().take(MAX_EMBEDDED_CHARS).collect())
            .collect();
        self.model
            .lock()
            .unwrap()
            .embed(cut, None)
            .map_err(|e| e.to_string())
    }

    fn query(&self, text: &str) -> Result<Vec<f32>, String> {
        let asked: String = text.chars().take(MAX_EMBEDDED_CHARS).collect();
        self.model
            .lock()
            .unwrap()
            .embed(vec![format!("{QUERY_PREFIX}{asked}")], None)
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .ok_or_else(|| "the model returned nothing".to_string())
    }
}

#[cfg(test)]
pub mod testing {
    use super::*;

    /// A stand-in model that knows a handful of ideas: each idea is one
    /// direction, and a text points along the ideas whose words it has.
    /// So "feline" and "cat" are close although they share no letters.
    pub struct Ideas;

    const IDEAS: [&[&str]; 4] = [
        &["cat", "cats", "feline", "kitten", "purr", "purrs"],
        &["boat", "ship", "vessel", "sail", "sails", "harbour"],
        &["money", "salary", "paid", "wage", "wages", "earn", "earns"],
        &["rain", "storm", "weather", "wet", "umbrella"],
    ];

    fn vector(text: &str) -> Vec<f32> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).collect();
        IDEAS
            .iter()
            .map(|idea| words.iter().filter(|w| idea.contains(w)).count() as f32)
            .collect()
    }

    impl Embedder for Ideas {
        fn name(&self) -> &str {
            "ideas"
        }
        fn passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|t| vector(t)).collect())
        }
        fn query(&self, text: &str) -> Result<Vec<f32>, String> {
            Ok(vector(text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Opt-in: loads the real model (downloading it if need be) and checks
    /// that meaning, not shared words, decides what is close.
    #[test]
    #[ignore]
    fn real_embed_model_puts_like_meanings_close() {
        let model = Local::load(&Local::default_dir().unwrap()).unwrap();
        let passages = [
            "The cat sat purring on the windowsill in the sun.".to_string(),
            "Quarterly revenue rose by twelve percent on strong exports.".to_string(),
            "To undo the last commit but keep the changes, run git reset --soft HEAD~1."
                .to_string(),
        ];
        let vectors = model.passages(&passages).unwrap();
        assert_eq!(vectors.len(), 3);
        let cos = |a: &[f32], b: &[f32]| {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let (la, lb) = (
                a.iter().map(|x| x * x).sum::<f32>().sqrt(),
                b.iter().map(|x| x * x).sum::<f32>().sqrt(),
            );
            dot / (la * lb)
        };
        // How close passages that say the same thing are, against passages
        // that are about the same subject but say something else.
        for (a, b) in [
            (
                "Lesson: the real cause of the failing test was the override in conf.d/90-local.ini, not the main config.",
                "Lesson: the test failed because of an override in conf.d/90-local.ini rather than the main configuration file.",
            ),
            (
                "Lesson: the real cause of the failing test was the override in conf.d/90-local.ini, not the main config.",
                "Lesson: run the full test suite with run_tests.py, since pytest alone skips the fixtures.",
            ),
            (
                "User's name is Sam.\n\nKind: identity\nUser said: I am Sam\n\nAnswer given: Nice to meet you, Sam.",
                "The user is called Sam.\n\nKind: identity\nUser said: my name is Sam\n\nAnswer given: Hello Sam!",
            ),
            (
                "The meeting is on Monday at 10.\n\nKind: event\nUser said: the meeting is Monday 10am",
                "The meeting is on Tuesday at 15.\n\nKind: event\nUser said: the meeting is Tuesday 3pm",
            ),
            (
                "User prefers tabs for indentation.\n\nKind: instruction",
                "User prefers spaces for indentation.\n\nKind: instruction",
            ),
        ] {
            let v = model.passages(&[a.to_string(), b.to_string()]).unwrap();
            println!(
                "pair {:.3}: {} || {}",
                cos(&v[0], &v[1]),
                &a[..40],
                &b[..40]
            );
        }
        for (a, b) in [
            (
                "User is planning a trip to Kyoto in April.\n\nKind: event\nUser said: I'm going to Kyoto in April\n\nAnswer given: Sounds lovely.",
                "User booked a ryokan in Kyoto for the April trip.\n\nKind: event\nUser said: booked the ryokan for Kyoto\n\nAnswer given: Great.",
            ),
            (
                "User is planning a trip to Kyoto in April.\n\nKind: event\nUser said: I'm going to Kyoto in April\n\nAnswer given: Sounds lovely.",
                "The Kyoto trip budget is 300,000 yen.\n\nKind: fact\nUser said: budget for kyoto is 300k yen\n\nAnswer given: Noted.",
            ),
            (
                "User is planning a trip to Kyoto in April.\n\nKind: event\nUser said: I'm going to Kyoto in April\n\nAnswer given: Sounds lovely.",
                "The build server runs Ubuntu 24.04.\n\nKind: fact\nUser said: our build server is ubuntu 24.04\n\nAnswer given: Noted.",
            ),
            (
                "The deploy script needs the STAGING flag.\n\nKind: fact\nUser said: deploy needs STAGING flag\n\nAnswer given: Ok.",
                "User asked how many python files exist; there are 4.\n\nKind: event\nUser said: how many py files\n\nAnswer given: 4",
            ),
            (
                "The deploy script needs the STAGING flag.\n\nKind: fact\nUser said: deploy needs STAGING flag\n\nAnswer given: Ok.",
                "Deploys to staging run through deploy.sh with STAGING=1.\n\nKind: fact\nUser said: staging deploy uses deploy.sh STAGING=1\n\nAnswer given: Ok.",
            ),
        ] {
            let v = model.passages(&[a.to_string(), b.to_string()]).unwrap();
            println!("mem {:.3}: {} || {}", cos(&v[0], &v[1]), &a[..30], &b[..30]);
        }
        for (question, best) in [
            ("a feline resting by the glass", 0),
            ("how did the company's sales do", 1),
            (
                "how do I take back my most recent check-in without losing work",
                2,
            ),
        ] {
            let q = model.query(question).unwrap();
            let sims: Vec<f32> = vectors.iter().map(|v| cos(&q, v)).collect();
            println!("{question}: {sims:?}");
            let top = sims
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert_eq!(top, best, "{question}: {sims:?}");
        }
    }
}
