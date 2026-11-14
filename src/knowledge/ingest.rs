//! Turns a source text into the pieces of a graph: sections, passages, the
//! terms they name and the paths they point at.
//!
//! Boundary: pure functions from text to a description. No files, no store,
//! no model. Everything here is deterministic, so what it produces is
//! recorded as verified.

/// Passages are merged up to about this many characters...
const CHUNK_TARGET: usize = 900;
/// ...and anything longer than this is split.
const CHUNK_MAX: usize = 1_600;
const TERM_MAX: usize = 60;

#[derive(Debug, Default, PartialEq)]
pub struct Parsed {
    /// Whether the source is prose (with headings) rather than code.
    pub prose: bool,
    pub title: String,
    pub sections: Vec<Section>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Section {
    /// Heading path, e.g. `Stack > Commands`. Empty before the first heading
    /// and for sources without headings.
    pub heading: String,
    pub chunks: Vec<Chunk>,
}

#[derive(Debug, Default, PartialEq)]
pub struct Chunk {
    pub text: String,
    /// Names this passage mentions, as written.
    pub terms: Vec<String>,
    /// Paths this passage points at.
    pub refs: Vec<String>,
}

pub fn is_prose(source: &str) -> bool {
    let lower = source.to_ascii_lowercase();
    // A web page is kept as the text read from it.
    lower.starts_with("http://")
        || lower.starts_with("https://")
        || [".md", ".markdown", ".txt", ".rst"]
            .iter()
            .any(|ext| lower.ends_with(ext))
}

fn file_name(source: &str) -> &str {
    source.rsplit('/').next().unwrap_or(source)
}

/// Splits a source into sections and passages.
pub fn parse(source: &str, text: &str) -> Parsed {
    let mut parsed = if is_prose(source) {
        parse_prose(text)
    } else {
        Parsed {
            prose: false,
            title: String::new(),
            sections: vec![Section {
                heading: String::new(),
                chunks: blocks(text).into_iter().map(code_chunk).collect(),
            }],
        }
    };
    parsed.prose = is_prose(source);
    if parsed.title.is_empty() {
        // The file's name without its extension: a name, not a path.
        let name = file_name(source);
        parsed.title = name
            .rsplit_once('.')
            .map_or(name, |(stem, _)| stem)
            .to_string();
    }
    parsed.sections.retain(|s| !s.chunks.is_empty());
    parsed
}

fn parse_prose(text: &str) -> Parsed {
    let mut parsed = Parsed::default();
    // Heading text at each depth, to build the path.
    let mut path: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut heading = String::new();
    let mut fenced = false;
    let flush = |heading: &str, body: &mut String, parsed: &mut Parsed| {
        let chunks: Vec<Chunk> = blocks(body).into_iter().map(prose_chunk).collect();
        parsed.sections.push(Section {
            heading: heading.to_string(),
            chunks,
        });
        body.clear();
    };
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        let depth = line.chars().take_while(|c| *c == '#').count();
        let is_heading = !fenced && (1..=6).contains(&depth) && line[depth..].starts_with(' ');
        if !is_heading {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        flush(&heading, &mut body, &mut parsed);
        let name = line[depth..].trim().to_string();
        if depth == 1 && parsed.title.is_empty() {
            parsed.title = name.clone();
        }
        path.truncate(depth - 1);
        // A document that skips levels still gets a sensible path.
        while path.len() < depth - 1 {
            path.push(String::new());
        }
        path.push(name);
        heading = path
            .iter()
            .filter(|p| !p.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" > ");
    }
    flush(&heading, &mut body, &mut parsed);
    parsed
}

/// Groups blank-line-separated paragraphs into passages of a useful size.
/// A fenced code block stays in one piece with its fence lines.
fn blocks(text: &str) -> Vec<String> {
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut fenced = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        if line.trim().is_empty() && !fenced {
            if !current.trim().is_empty() {
                paragraphs.push(std::mem::take(&mut current));
            }
            current.clear();
        } else {
            current.push_str(line);
            current.push('\n');
        }
    }
    if !current.trim().is_empty() {
        paragraphs.push(current);
    }

    let mut out: Vec<String> = Vec::new();
    for paragraph in paragraphs {
        for piece in split_long(&paragraph) {
            match out.last_mut() {
                Some(last) if last.len() + piece.len() <= CHUNK_TARGET => {
                    last.push('\n');
                    last.push_str(&piece);
                }
                _ => out.push(piece),
            }
        }
    }
    out.into_iter().map(|b| b.trim_end().to_string()).collect()
}

/// Splits a paragraph longer than `CHUNK_MAX` at line ends.
fn split_long(paragraph: &str) -> Vec<String> {
    if paragraph.len() <= CHUNK_MAX {
        return vec![paragraph.to_string()];
    }
    let mut out = vec![String::new()];
    for line in paragraph.lines() {
        if out.last().unwrap().len() + line.len() > CHUNK_MAX && !out.last().unwrap().is_empty() {
            out.push(String::new());
        }
        let last = out.last_mut().unwrap();
        last.push_str(line);
        last.push('\n');
    }
    out
}

fn looks_like_path(s: &str) -> bool {
    let known = [
        ".rs", ".md", ".toml", ".json", ".rhai", ".py", ".js", ".ts", ".txt", ".sh",
    ];
    !s.contains(' ') && !s.contains("://") && known.iter().any(|ext| s.ends_with(ext))
}

/// A passage of prose: terms are its inline `code` spans, references its
/// links and any span that is a path.
fn prose_chunk(text: String) -> Chunk {
    let mut terms = Vec::new();
    let mut refs = Vec::new();
    let mut fenced = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        // Inline code: the odd-numbered pieces between backticks.
        for span in line.split('`').skip(1).step_by(2) {
            let span = span.trim();
            if span.len() < 2 || span.len() > TERM_MAX {
                continue;
            }
            if looks_like_path(span) {
                refs.push(span.to_string());
            }
            terms.push(span.to_string());
        }
        // Markdown links: [text](target)
        for target in line.split("](").skip(1) {
            let target = target.split(')').next().unwrap_or_default();
            let target = target.split('#').next().unwrap_or_default().trim();
            if !target.is_empty() && !target.contains("://") {
                refs.push(target.to_string());
            }
        }
    }
    dedup(&mut terms);
    dedup(&mut refs);
    Chunk { text, terms, refs }
}

/// A passage of code: terms are the things it defines.
fn code_chunk(text: String) -> Chunk {
    const KEYWORDS: [&str; 11] = [
        "fn ",
        "struct ",
        "enum ",
        "trait ",
        "mod ",
        "type ",
        "const ",
        "static ",
        "def ",
        "class ",
        "function ",
    ];
    const PREFIXES: [&str; 6] = [
        "pub(crate) ",
        "pub(super) ",
        "pub ",
        "async ",
        "export ",
        "unsafe ",
    ];
    let mut terms = Vec::new();
    for line in text.lines() {
        let mut rest = line.trim_start();
        while let Some(stripped) = PREFIXES.iter().find_map(|p| rest.strip_prefix(p)) {
            rest = stripped;
        }
        let Some(after) = KEYWORDS.iter().find_map(|k| rest.strip_prefix(k)) else {
            continue;
        };
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.len() >= 2 && name.len() <= TERM_MAX {
            terms.push(name);
        }
    }
    dedup(&mut terms);
    Chunk {
        text,
        terms,
        refs: Vec::new(),
    }
}

fn dedup(items: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    items.retain(|i| seen.insert(i.clone()));
}

/// The names a term is filed under: itself, lowercased and without a
/// trailing `()`, plus its last path segment (`learn::Knowledge` is also
/// `knowledge`), so different spellings of one thing meet at one node.
pub fn term_keys(term: &str) -> Vec<String> {
    let full = term.trim().trim_end_matches("()").to_lowercase();
    let mut keys = vec![full.clone()];
    // A path's short name is its file name; an identifier's is its last
    // segment.
    let last = match full.rsplit_once('/') {
        Some((_, file)) => file,
        None => full.rsplit([':', '.']).next().unwrap_or_default(),
    }
    .to_string();
    if last.len() >= 3 && last != full && !full.contains(' ') {
        keys.push(last);
    }
    keys.retain(|k| k.len() >= 2);
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "# Guide\n\nIntro with `PolicyStore` and a link to [the demo](src/demo.rs).\n\n## Setup\n\nRun `cargo test`.\n\n```sh\n# not a heading\ncargo run\n```\n\n### Details\n\nSee `src/router.rs` for `Router::choose()`.\n";

    #[test]
    fn prose_is_split_by_heading_with_terms_and_references() {
        let p = parse("docs/guide.md", DOC);
        assert_eq!(p.title, "Guide");
        let headings: Vec<&str> = p.sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(
            headings,
            ["Guide", "Guide > Setup", "Guide > Setup > Details"]
        );
        let intro = &p.sections[0].chunks[0];
        assert_eq!(intro.terms, ["PolicyStore"]);
        assert_eq!(intro.refs, ["src/demo.rs"]);
        // The fenced block stays with its section and is not a heading.
        let setup = &p.sections[1].chunks[0];
        assert!(setup.text.contains("# not a heading"));
        assert_eq!(setup.terms, ["cargo test"]);
        let details = &p.sections[2].chunks[0];
        assert_eq!(details.terms, ["src/router.rs", "Router::choose()"]);
        assert_eq!(details.refs, ["src/router.rs"]);
    }

    #[test]
    fn code_is_chunked_by_blank_lines_and_names_its_definitions() {
        let code = format!(
            "pub struct Router {{\n    x: u32,\n}}\n\n{}\n\npub(crate) fn choose() {{}}\nasync fn run() {{}}\n",
            "// filler line that makes this block long enough to stand alone\n".repeat(16)
        );
        let p = parse("src/router.rs", &code);
        assert_eq!(p.title, "router");
        assert!(!p.prose);
        assert_eq!(p.sections.len(), 1);
        let terms: Vec<&String> = p.sections[0].chunks.iter().flat_map(|c| &c.terms).collect();
        assert_eq!(terms, ["Router", "choose", "run"]);
        assert!(p.sections[0].chunks.len() >= 2);
    }

    #[test]
    fn long_text_is_split_and_short_paragraphs_merged() {
        let short = "one.\n\ntwo.\n\nthree.\n";
        assert_eq!(blocks(short), ["one.\n\ntwo.\n\nthree."]);
        let long = "a line of text that is repeated many times over\n".repeat(80);
        let pieces = blocks(&long);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|p| p.len() <= CHUNK_MAX));
    }

    #[test]
    fn terms_are_filed_under_their_full_and_short_names() {
        assert_eq!(
            term_keys("learn::Knowledge::select()"),
            ["learn::knowledge::select", "select"]
        );
        assert_eq!(term_keys("PolicyStore"), ["policystore"]);
        assert_eq!(term_keys("cargo test"), ["cargo test"]);
        assert_eq!(term_keys("src/demo.rs"), ["src/demo.rs", "demo.rs"]);
    }
}
