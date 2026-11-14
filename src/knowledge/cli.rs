//! `conscious knowledge …`: ingest, search and inspect the knowledge store.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use super::{DEFAULT_SPACE, Ingested, KnowledgeBase, Query, Trust};
use crate::provider;

pub const USAGE: &str = "\
conscious knowledge ingest PATH...     add files or folders (re-ingest replaces; unchanged are skipped)
conscious knowledge search QUERY       graph lookup; shows each passage, its source and why it was found
conscious knowledge add TEXT           remember a note (words in `backticks` link it to related passages)
conscious knowledge stats              what the store holds

Options:
  --space NAME        namespace to write to or search in (default: `default`; search: all spaces)
  --hops N            search: edges a passage may be from a direct match (default 2; 0 = plain search)
  --results N         search: most passages returned (default 6)
  --budget N          search: most tokens returned (default 1500)
  --trust LEVEL       search: ignore knowledge below verified | experience | inferred (default inferred)
  --extract --model SPEC   ingest: also ask a model for relations in each passage (recorded as inferred)
";

/// Files larger than this are skipped.
const MAX_FILE_BYTES: u64 = 300_000;
/// Passages sent to the model per ingest when extracting.
const MAX_EXTRACTED: usize = 40;
const EXTENSIONS: [&str; 14] = [
    "md", "markdown", "txt", "rst", "rs", "toml", "py", "js", "ts", "go", "java", "c", "h", "rhai",
];
const SKIPPED_DIRS: [&str; 5] = ["target", "node_modules", "dist", "build", "vendor"];

struct Options {
    space: Option<String>,
    hops: Option<u8>,
    results: Option<usize>,
    budget: Option<u32>,
    trust: Option<Trust>,
    extract: bool,
    model: Option<String>,
    rest: Vec<String>,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        space: None,
        hops: None,
        results: None,
        budget: None,
        trust: None,
        extract: false,
        model: std::env::var("CONSCIOUS_MODEL")
            .ok()
            .filter(|m| !m.is_empty()),
        rest: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        let number = |name: &str, v: String| {
            v.parse::<u32>()
                .map_err(|_| format!("{name} needs a number"))
        };
        match a.as_str() {
            "--space" => o.space = Some(value("--space")?),
            "--hops" => o.hops = Some(number("--hops", value("--hops")?)?.min(6) as u8),
            "--results" => o.results = Some(number("--results", value("--results")?)? as usize),
            "--budget" => o.budget = Some(number("--budget", value("--budget")?)?),
            "--trust" => {
                o.trust = Some(match value("--trust")?.as_str() {
                    "verified" => Trust::Verified,
                    "experience" => Trust::Experience,
                    "inferred" => Trust::Inferred,
                    other => return Err(format!("unknown trust level `{other}`")),
                })
            }
            "--extract" => o.extract = true,
            "--model" | "-m" => o.model = Some(value("--model")?),
            _ if a.starts_with("--") => return Err(format!("unknown option `{a}`")),
            _ => o.rest.push(a.clone()),
        }
    }
    Ok(o)
}

/// Most files taken from a project when syncing it automatically.
const MAX_SYNCED_FILES: usize = 1_500;
/// Files whose presence marks a folder as a project worth syncing.
const PROJECT_MARKERS: [&str; 9] = [
    ".git",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
    "go.mod",
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "Makefile",
];

#[derive(Debug, Default, PartialEq)]
pub struct Synced {
    /// The space the project lives in: its folder name.
    pub space: String,
    pub files: usize,
    /// Files added or re-read because they changed.
    pub changed: usize,
    /// Documents dropped because their file is gone.
    pub removed: usize,
    /// Whether the project had more files than are taken.
    pub capped: bool,
}

/// Brings the store up to date with the project in `root`: new and changed
/// files are (re)ingested, unchanged ones skipped, deleted ones dropped.
/// Returns `None` if `root` does not look like a project, so that running
/// from a home folder does not swallow it.
pub fn sync_project(kb: &mut KnowledgeBase, root: &Path) -> Result<Option<Synced>, String> {
    if !PROJECT_MARKERS.iter().any(|m| root.join(m).exists()) {
        return Ok(None);
    }
    let space = root
        .canonicalize()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let mut files = Vec::new();
    collect(root, &mut files);
    let capped = files.len() > MAX_SYNCED_FILES;
    files.truncate(MAX_SYNCED_FILES);

    let mut synced = Synced {
        space: space.clone(),
        capped,
        ..Default::default()
    };
    let mut keep = std::collections::HashSet::new();
    for file in &files {
        let source = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .into_owned();
        let too_big = std::fs::metadata(file).is_ok_and(|m| m.len() > MAX_FILE_BYTES);
        let Some(text) = (!too_big)
            .then(|| std::fs::read_to_string(file).ok())
            .flatten()
        else {
            continue;
        };
        synced.files += 1;
        if let Ingested::Added { .. } = kb.ingest(&space, &source, &text) {
            synced.changed += 1;
        }
        keep.insert(source);
    }
    synced.removed = kb.prune(&space, &keep);
    if synced.changed > 0 || synced.removed > 0 {
        kb.commit()?;
    }
    Ok(Some(synced))
}

/// Files under `path` worth ingesting, in a stable order.
fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        let mut entries: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for entry in entries {
            let name = entry
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if name.starts_with('.') || (entry.is_dir() && SKIPPED_DIRS.contains(&name)) {
                continue;
            }
            collect(&entry, out);
        }
    } else if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
    {
        out.push(path.to_path_buf());
    }
}

pub fn run(args: &[String]) -> ExitCode {
    let Some((command, rest)) = args.split_first() else {
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };
    let options = match parse(rest) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let Some(dir) = KnowledgeBase::default_dir() else {
        eprintln!("error: no home directory to keep the knowledge store in");
        return ExitCode::FAILURE;
    };
    let mut kb = match KnowledgeBase::open(&dir) {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = match command.as_str() {
        "ingest" => ingest(&mut kb, &options),
        "search" => {
            by_meaning(&mut kb);
            search(&kb, &options)
        }
        "add" => add(&mut kb, &options),
        "stats" => {
            stats(&kb);
            Ok(())
        }
        other => Err(format!("unknown knowledge command `{other}`\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn ingest(kb: &mut KnowledgeBase, o: &Options) -> Result<(), String> {
    if o.rest.is_empty() {
        return Err("ingest needs at least one file or folder".into());
    }
    let extractor = match (o.extract, &o.model) {
        (false, _) => None,
        (true, None) => return Err("--extract needs --model SPEC (or CONSCIOUS_MODEL)".into()),
        (true, Some(spec)) => Some(provider::from_spec(spec).map_err(|e| e.to_string())?),
    };
    let space = o.space.as_deref().unwrap_or(DEFAULT_SPACE);
    let mut files = Vec::new();
    for path in &o.rest {
        let path = Path::new(path);
        if !path.exists() {
            return Err(format!("{}: no such file or folder", path.display()));
        }
        collect(path, &mut files);
    }
    let (mut added, mut unchanged, mut skipped, mut passages) = (0, 0, 0, 0);
    let mut fresh = Vec::new();
    for file in &files {
        let source = file.to_string_lossy().trim_start_matches("./").to_string();
        let too_big = std::fs::metadata(file).is_ok_and(|m| m.len() > MAX_FILE_BYTES);
        let Some(text) = (!too_big)
            .then(|| std::fs::read_to_string(file).ok())
            .flatten()
        else {
            skipped += 1;
            continue;
        };
        match kb.ingest(space, &source, &text) {
            Ingested::Added { passages: n, .. } => {
                added += 1;
                passages += n;
                fresh.push(source);
            }
            Ingested::Unchanged => unchanged += 1,
        }
    }
    kb.commit()?;
    println!(
        "ingested {added} file{} ({passages} passages) into `{space}`; {unchanged} unchanged, {skipped} skipped",
        if added == 1 { "" } else { "s" }
    );

    if let Some(model) = extractor {
        let targets: Vec<_> = fresh
            .iter()
            .flat_map(|source| kb.passages_of(space, source))
            .take(MAX_EXTRACTED)
            .collect();
        let mut relations = 0;
        for (i, id) in targets.iter().enumerate() {
            let text = kb.node(*id).map(|n| n.text.clone()).unwrap_or_default();
            match super::relations(model.as_ref(), &text) {
                Ok(triples) => relations += kb.add_relations(*id, &triples),
                Err(e) => return Err(format!("extraction stopped at passage {}: {e}", i + 1)),
            }
            eprint!("\rextracting relations: {}/{}", i + 1, targets.len());
        }
        eprintln!();
        kb.commit()?;
        println!(
            "extracted {relations} inferred relation{} from {} passage{}",
            if relations == 1 { "" } else { "s" },
            targets.len(),
            if targets.len() == 1 { "" } else { "s" }
        );
    }
    Ok(())
}

fn search(kb: &KnowledgeBase, o: &Options) -> Result<(), String> {
    if o.rest.is_empty() {
        return Err("search needs a query".into());
    }
    let mut query = Query::new(o.rest.join(" "));
    query.spaces = o.space.clone().map(|s| vec![s]);
    if let Some(hops) = o.hops {
        query.max_hops = hops;
    }
    if let Some(results) = o.results {
        query.max_results = results;
    }
    if let Some(budget) = o.budget {
        query.token_budget = budget;
    }
    if let Some(trust) = o.trust {
        query.min_trust = trust;
    }
    let found = kb.retrieve(&query);
    for (i, hit) in found.hits.iter().enumerate() {
        let Some(n) = kb.node(hit.node) else { continue };
        println!(
            "[{}] {}  (score {:.2}, {} tokens)\n    source: {} · {} · space `{}`\n    why: {}",
            i + 1,
            n.title,
            hit.score,
            hit.tokens,
            n.provenance.source,
            n.provenance.trust.label(),
            n.space,
            hit.why
        );
        for line in n.text.lines().take(4) {
            let shown: String = line.chars().take(110).collect();
            println!("    | {shown}");
        }
        println!();
    }
    println!(
        "{} passage{}, {} tokens · {} direct matches (by words or meaning), {} nodes reached within {} hop{} · {} did not fit the limits",
        found.hits.len(),
        if found.hits.len() == 1 { "" } else { "s" },
        found.tokens,
        found.seeds,
        found.reached,
        query.max_hops,
        if query.max_hops == 1 { "" } else { "s" },
        found.dropped
    );
    Ok(())
}

fn add(kb: &mut KnowledgeBase, o: &Options) -> Result<(), String> {
    let text = o.rest.join(" ");
    if text.trim().is_empty() {
        return Err("add needs the text to remember".into());
    }
    let space = o.space.as_deref().unwrap_or(DEFAULT_SPACE);
    kb.add_note(space, "user", &text);
    kb.commit()?;
    println!("noted in `{space}`");
    Ok(())
}

/// Switches on lookup by meaning for this command: loads the local model
/// and embeds whatever has no vector yet. If the model cannot be had, the
/// search is by words and links, and that is said.
fn by_meaning(kb: &mut KnowledgeBase) {
    use super::embed::Local;
    let loaded = Local::default_dir()
        .ok_or_else(|| "no home directory for its files".to_string())
        .and_then(|dir| Local::load(&dir));
    match loaded {
        Ok(model) => {
            kb.set_embedder(std::sync::Arc::new(model));
            let mut made = 0;
            loop {
                match kb.embed_some(64) {
                    Ok(0) => break,
                    Ok(n) => made += n,
                    Err(e) => {
                        eprintln!("note: embedding stopped: {e}");
                        break;
                    }
                }
            }
            if made > 0 {
                eprintln!("[embedded {made} passages that had no vector yet]");
            }
        }
        Err(e) => {
            eprintln!("note: searching by words only; the meaning model is not available: {e}")
        }
    }
}

fn stats(kb: &KnowledgeBase) {
    let s = kb.stats();
    println!(
        "{} documents · {} passages · {} terms · {} notes · {} edges ({} inferred) · {} embedded for lookup by meaning\nspaces: {}",
        s.documents,
        s.passages,
        s.terms,
        s.notes,
        s.edges,
        s.inferred_edges,
        s.embedded,
        if s.spaces.is_empty() {
            "none".to_string()
        } else {
            s.spaces.join(", ")
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_project_is_synced_incrementally_and_other_folders_are_left_alone() {
        let root = std::env::temp_dir().join(format!("conscious-sync-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        let mut kb = KnowledgeBase::in_memory();

        // Not a project yet: nothing is taken.
        std::fs::write(root.join("src/lib.rs"), "pub fn alpha() {}\n").unwrap();
        assert_eq!(sync_project(&mut kb, &root).unwrap(), None);

        std::fs::write(root.join("README.md"), "# Demo\n\nUses `alpha`.\n").unwrap();
        std::fs::write(root.join("target/junk.rs"), "fn ignored() {}\n").unwrap();
        let first = sync_project(&mut kb, &root).unwrap().unwrap();
        assert_eq!((first.files, first.changed, first.removed), (2, 2, 0));
        assert_eq!(first.space, root.file_name().unwrap().to_string_lossy());
        assert_eq!(kb.stats().documents, 2);

        // Nothing changed: nothing is re-read.
        let second = sync_project(&mut kb, &root).unwrap().unwrap();
        assert_eq!((second.files, second.changed, second.removed), (2, 0, 0));

        // One file edited, one deleted.
        std::fs::write(root.join("README.md"), "# Demo\n\nNow uses `beta`.\n").unwrap();
        std::fs::remove_file(root.join("src/lib.rs")).unwrap();
        let third = sync_project(&mut kb, &root).unwrap().unwrap();
        assert_eq!((third.files, third.changed, third.removed), (1, 1, 1));
        assert_eq!(kb.stats().documents, 1);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
