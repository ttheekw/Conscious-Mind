//! Markdown → styled terminal lines, for model replies.
//!
//! Covers what models actually write: headings, bold, italic, inline code,
//! fenced code blocks, bullet and numbered lists, block quotes, rules, links
//! and tables. Anything else is shown as plain text. The renderer wraps to a
//! given width itself, so callers can prefix every line.

use ratatui::style::{Modifier, Style};
use ratatui::text::Span;

use crate::theme::*;

pub type Row = Vec<Span<'static>>;

fn cells(s: &str) -> usize {
    s.chars().count()
}

/// A run of text in one style.
type Piece = (String, Style);

/// Splits a line of prose into styled runs: `**bold**`, `*italic*`,
/// `` `code` `` and `[text](url)`.
fn inline(text: &str, base: Style) -> Vec<Piece> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Piece> = Vec::new();
    let mut buf = String::new();
    let (mut strong, mut emph) = (false, false);
    let style = |strong: bool, emph: bool| {
        let mut s = base;
        if strong {
            s = s.fg(TEXT).add_modifier(Modifier::BOLD);
        }
        if emph {
            s = s.add_modifier(Modifier::ITALIC);
        }
        s
    };
    let flush = |buf: &mut String, out: &mut Vec<Piece>, s: Style| {
        if !buf.is_empty() {
            out.push((std::mem::take(buf), s));
        }
    };
    let find = |from: usize, needle: &[char]| -> Option<usize> {
        (from..chars.len().saturating_sub(needle.len() - 1))
            .find(|&i| chars[i..i + needle.len()] == *needle)
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let prev = i.checked_sub(1).map(|p| chars[p]);
        match c {
            '`' => {
                if let Some(end) = find(i + 1, &['`']) {
                    flush(&mut buf, &mut out, style(strong, emph));
                    out.push((
                        chars[i + 1..end].iter().collect(),
                        Style::new().fg(CYAN).bg(BAND),
                    ));
                    i = end + 1;
                    continue;
                }
                buf.push(c);
            }
            '*' if next == Some('*') => {
                flush(&mut buf, &mut out, style(strong, emph));
                strong = !strong;
                i += 2;
                continue;
            }
            // A lone marker opens emphasis before a word and closes it after
            // one. `_` inside a word (snake_case) is left alone.
            '*' | '_' => {
                let opens = !emph
                    && next.is_some_and(|n| !n.is_whitespace())
                    && (c == '*' || prev.is_none_or(|p| !p.is_alphanumeric()));
                let closes = emph
                    && prev.is_some_and(|p| !p.is_whitespace())
                    && (c == '*' || next.is_none_or(|n| !n.is_alphanumeric()));
                if opens || closes {
                    flush(&mut buf, &mut out, style(strong, emph));
                    emph = !emph;
                } else {
                    buf.push(c);
                }
            }
            '[' => {
                let link = find(i + 1, &[']', '(']).and_then(|mid| {
                    let end = find(mid + 2, &[')'])?;
                    Some((mid, end))
                });
                if let Some((mid, end)) = link {
                    flush(&mut buf, &mut out, style(strong, emph));
                    out.push((
                        chars[i + 1..mid].iter().collect(),
                        Style::new().fg(CYAN).add_modifier(Modifier::UNDERLINED),
                    ));
                    i = end + 1;
                    continue;
                }
                buf.push(c);
            }
            _ => buf.push(c),
        }
        i += 1;
    }
    flush(&mut buf, &mut out, style(strong, emph));
    out
}

/// The text of a line with its inline markers removed.
fn plain(text: &str) -> String {
    inline(text, Style::new())
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// Word-wraps styled runs to `width`. The first row starts with `first`,
/// the others with `rest`.
fn wrap(pieces: Vec<Piece>, width: usize, first: Row, rest: Row) -> Vec<Row> {
    let indent = |row: &Row| row.iter().map(|s| cells(&s.content)).sum::<usize>();
    let mut rows: Vec<Row> = Vec::new();
    let mut row = first;
    let mut used = indent(&row);
    let mut has_text = false;
    for (text, style) in pieces {
        // Words keep their trailing space, so breaks fall between words.
        for word in text.split_inclusive(' ') {
            let mut word = word.to_string();
            loop {
                let w = cells(word.trim_end());
                if used + w <= width {
                    break;
                }
                if has_text {
                    rows.push(std::mem::replace(&mut row, rest.clone()));
                    used = indent(&row);
                    has_text = false;
                    continue;
                }
                // A word longer than a whole row is cut.
                let room = width.saturating_sub(used).max(1);
                let head: String = word.chars().take(room).collect();
                word = word.chars().skip(room).collect();
                row.push(Span::styled(head, style));
                rows.push(std::mem::replace(&mut row, rest.clone()));
                used = indent(&row);
            }
            if word.is_empty() || (!has_text && word.trim().is_empty()) {
                continue;
            }
            used += cells(&word);
            has_text = true;
            row.push(Span::styled(word, style));
        }
    }
    rows.push(row);
    rows
}

fn is_rule(line: &str) -> bool {
    let t: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    t.len() >= 3
        && ["-", "*", "_"]
            .iter()
            .any(|m| t.chars().all(|c| m.contains(c)))
}

/// A list item's marker and text: `- x`, `* x`, `+ x`, `1. x`, `1) x`.
fn list_item(line: &str) -> Option<(String, &str)> {
    for bullet in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(bullet) {
            return Some(("• ".into(), rest));
        }
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if (1..=3).contains(&digits) {
        let after = &line[digits..];
        if let Some(rest) = after
            .strip_prefix(". ")
            .or_else(|| after.strip_prefix(") "))
        {
            return Some((format!("{}. ", &line[..digits]), rest));
        }
    }
    None
}

fn table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|c| plain(c.trim()))
        .collect()
}

fn is_table_divider(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.contains('-') && t.chars().all(|c| "|-: ".contains(c))
}

/// A table with columns aligned, shrunk to fit `width`.
fn table(lines: &[&str], width: usize) -> Vec<Row> {
    let rows: Vec<Vec<String>> = lines
        .iter()
        .filter(|l| !is_table_divider(l))
        .map(|l| table_cells(l))
        .collect();
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    if columns == 0 {
        return Vec::new();
    }
    let gaps = 3 * (columns - 1);
    let cap = (width.saturating_sub(gaps) / columns).max(4);
    let widths: Vec<usize> = (0..columns)
        .map(|c| {
            rows.iter()
                .filter_map(|r| r.get(c))
                .map(|cell| cells(cell))
                .max()
                .unwrap_or(0)
                .min(cap)
        })
        .collect();
    let fit = |cell: &str, w: usize| -> String {
        let mut out: String = if cells(cell) > w {
            let mut cut: String = cell.chars().take(w.saturating_sub(1)).collect();
            cut.push('…');
            cut
        } else {
            cell.to_string()
        };
        while cells(&out) < w {
            out.push(' ');
        }
        out
    };
    let mut out = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let mut spans = Vec::new();
        for (c, w) in widths.iter().enumerate() {
            if c > 0 {
                spans.push(Span::styled(" │ ", fg(GRID)));
            }
            let cell = row.get(c).map(String::as_str).unwrap_or_default();
            spans.push(Span::styled(
                fit(cell, *w),
                if r == 0 { bold(AMBER) } else { fg(TEXT) },
            ));
        }
        out.push(spans);
        if r == 0 {
            let rule: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
            out.push(vec![Span::styled(rule.join("─┼─"), fg(GRID))]);
        }
    }
    out
}

/// Renders markdown to rows no wider than `width`.
pub fn render(text: &str, width: usize) -> Vec<Row> {
    let width = width.max(8);
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<Row> = Vec::new();
    let blank = |out: &mut Vec<Row>| {
        if out.last().is_some_and(|r| !r.is_empty()) {
            out.push(Vec::new());
        }
    };
    let body = fg(TEXT);
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let indent = (cells(line) - cells(trimmed)).min(8);

        // Fenced code: kept exactly, never re-wrapped by words.
        if trimmed.starts_with("```") {
            let language = trimmed.trim_start_matches('`').trim();
            let end = (i + 1..lines.len())
                .find(|&j| lines[j].trim_start().starts_with("```"))
                .unwrap_or(lines.len());
            if !language.is_empty() {
                out.push(vec![Span::styled(format!(" {language} "), fg(DIM))]);
            }
            let code = Style::new().fg(NEON).bg(BAND);
            for raw in &lines[(i + 1).min(end)..end] {
                let chars: Vec<char> = raw.replace('\t', "    ").chars().collect();
                let room = width.saturating_sub(2).max(1);
                let mut parts: Vec<String> =
                    chars.chunks(room).map(|c| c.iter().collect()).collect();
                if parts.is_empty() {
                    parts.push(String::new());
                }
                for part in parts {
                    let pad = " ".repeat(room.saturating_sub(cells(&part)));
                    out.push(vec![Span::styled(format!(" {part}{pad} "), code)]);
                }
            }
            i = end + 1;
            continue;
        }
        if trimmed.is_empty() {
            blank(&mut out);
            i += 1;
            continue;
        }
        if trimmed.starts_with('|') && lines.get(i + 1).is_some_and(|l| is_table_divider(l)) {
            let end = (i..lines.len())
                .find(|&j| !lines[j].trim_start().starts_with('|'))
                .unwrap_or(lines.len());
            out.extend(table(&lines[i..end], width));
            i = end;
            continue;
        }
        let depth = trimmed.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&depth) && trimmed[depth..].starts_with(' ') {
            blank(&mut out);
            let style = if depth <= 2 { bold(AMBER) } else { bold(CYAN) };
            out.extend(wrap(
                inline(trimmed[depth..].trim(), style),
                width,
                Vec::new(),
                Vec::new(),
            ));
            i += 1;
            continue;
        }
        if is_rule(trimmed) {
            out.push(vec![Span::styled("─".repeat(width), fg(GRID))]);
            i += 1;
            continue;
        }
        if let Some(quote) = trimmed.strip_prefix('>') {
            let bar = vec![Span::styled("▎ ", fg(VIOLET))];
            let style = Style::new().fg(DIM).add_modifier(Modifier::ITALIC);
            out.extend(wrap(inline(quote.trim(), style), width, bar.clone(), bar));
            i += 1;
            continue;
        }
        if let Some((marker, rest)) = list_item(trimmed) {
            let pad = " ".repeat(indent);
            let first = vec![
                Span::raw(pad.clone()),
                Span::styled(marker.clone(), bold(CYAN)),
            ];
            let hang = vec![Span::raw(format!("{pad}{}", " ".repeat(cells(&marker))))];
            out.extend(wrap(inline(rest, body), width, first, hang));
            i += 1;
            continue;
        }
        // A paragraph: this line and the plain lines that follow it.
        let mut paragraph = trimmed.to_string();
        i += 1;
        while let Some(next) = lines.get(i).map(|l| l.trim()) {
            let starts_block = next.is_empty()
                || next.starts_with("```")
                || next.starts_with('#')
                || next.starts_with('>')
                || next.starts_with('|')
                || is_rule(next)
                || list_item(next).is_some();
            if starts_block {
                break;
            }
            paragraph.push(' ');
            paragraph.push_str(next);
            i += 1;
        }
        out.extend(wrap(
            inline(&paragraph, body),
            width,
            Vec::new(),
            Vec::new(),
        ));
    }
    while out.last().is_some_and(Vec::is_empty) {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(row: &Row) -> String {
        row.iter().map(|s| s.content.as_ref()).collect()
    }

    fn all(rows: &[Row]) -> Vec<String> {
        rows.iter().map(text).collect()
    }

    fn styled<'a>(rows: &'a [Row], needle: &str) -> &'a Span<'static> {
        rows.iter()
            .flatten()
            .find(|s| s.content.contains(needle))
            .unwrap_or_else(|| panic!("no span containing {needle:?}"))
    }

    #[test]
    fn inline_markers_become_styles_and_disappear() {
        let rows = render("Use **bold**, *italic*, `code` and a [link](http://x).", 80);
        assert_eq!(all(&rows), ["Use bold, italic, code and a link."]);
        assert!(
            styled(&rows, "bold")
                .style
                .add_modifier
                .contains(Modifier::BOLD)
        );
        assert!(
            styled(&rows, "italic")
                .style
                .add_modifier
                .contains(Modifier::ITALIC)
        );
        assert_eq!(styled(&rows, "code").style.bg, Some(BAND));
        assert!(
            styled(&rows, "link")
                .style
                .add_modifier
                .contains(Modifier::UNDERLINED)
        );
    }

    #[test]
    fn identifiers_and_stray_markers_are_left_alone() {
        let rows = render("Set max_hops and token_budget; 2 * 3 = 6.", 80);
        assert_eq!(all(&rows), ["Set max_hops and token_budget; 2 * 3 = 6."]);
        assert_eq!(all(&render("an unclosed `tick", 80)), ["an unclosed `tick"]);
    }

    #[test]
    fn headings_lists_quotes_and_rules() {
        let md = "# Title\nIntro line\ncontinues here.\n\n- first\n- second item\n  1. nested\n\n> quoted\n\n---\n### Small";
        assert_eq!(
            all(&render(md, 40)),
            [
                "Title",
                "Intro line continues here.",
                "",
                "• first",
                "• second item",
                "  1. nested",
                "",
                "▎ quoted",
                "",
                "────────────────────────────────────────",
                "",
                "Small",
            ]
        );
    }

    #[test]
    fn long_text_wraps_at_words_with_a_hanging_indent() {
        let rows = render("- alpha beta gamma delta epsilon zeta", 16);
        assert_eq!(
            all(&rows),
            ["• alpha beta ", "  gamma delta ", "  epsilon zeta"]
        );
        assert!(
            all(&rows)
                .iter()
                .all(|r| r.trim_end().chars().count() <= 16)
        );
        // A word longer than the row is cut rather than overflowing.
        let rows = render("abcdefghijklmnopqrstuvwxyz", 10);
        assert_eq!(all(&rows), ["abcdefghij", "klmnopqrst", "uvwxyz"]);
    }

    #[test]
    fn code_blocks_keep_their_lines_and_spacing() {
        let md = "Before\n```rust\nfn main() {\n    let x = 1;\n}\n```\nAfter";
        let rows = render(md, 24);
        assert_eq!(
            all(&rows),
            [
                "Before",
                " rust ",
                " fn main() {            ",
                "     let x = 1;         ",
                " }                      ",
                "After",
            ]
        );
        assert_eq!(styled(&rows, "let x").style.bg, Some(BAND));
        // Markdown inside code is not interpreted.
        assert_eq!(
            all(&render("```\n**not bold**\n```", 24))[0].trim(),
            "**not bold**"
        );
    }

    #[test]
    fn tables_are_aligned_and_fit_the_width() {
        let md = "| Model | Tier |\n|---|---|\n| haiku | **small** |\n| opus | large |";
        assert_eq!(
            all(&render(md, 40)),
            [
                "Model │ Tier ",
                "──────┼──────",
                "haiku │ small",
                "opus  │ large"
            ]
        );
        let wide = "| A | B |\n|---|---|\n| a very long cell indeed | another long cell here |";
        assert!(
            all(&render(wide, 24))
                .iter()
                .all(|r| r.chars().count() <= 24)
        );
    }
}
