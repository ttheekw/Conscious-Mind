//! Rendering. Pure function of `App` state; records clickable regions in
//! `app.hits` as it draws so mouse input can be resolved afterwards.
//!
//! Visual language: HUD frames (dim rule + bright corner brackets), slanted
//! gradient title chips, segmented gauges, smooth braille area charts, per-agent
//! activity pulses and a dot-grid texture in empty space.

use std::f64::consts::PI;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Circle, Line as CanvasLine, Points};
use ratatui::widgets::{
    Block, BorderType, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use crate::app::{
    App, BOOT_LINES, BOOT_STEP, Button, Hits, MemState, Modal, Panel, Tab, TaskView, ToolCall,
    ToolStatus,
};
use crate::event::{
    AgentState, EventKind, LookupPick, MemKind, Risk, Severity, StepStatus, TaskState,
};
use crate::live::MAIN;
use crate::markdown;
use crate::theme::*;

// ───────────────────────────── helpers ─────────────────────────────

fn t(d: Duration) -> String {
    let s = d.as_secs();
    format!("T+{:02}:{:02}", s / 60, s % 60)
}

fn tok(n: u64) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

fn width(s: &str) -> usize {
    s.chars().count()
}

fn trunc(s: &str, w: usize) -> String {
    if width(s) <= w {
        s.to_string()
    } else if w == 0 {
        String::new()
    } else {
        let mut out: String = s.chars().take(w - 1).collect();
        out.push('…');
        out
    }
}

fn pad(s: &str, w: usize) -> String {
    let s = trunc(s, w);
    let n = width(&s);
    format!("{s}{}", " ".repeat(w.saturating_sub(n)))
}

fn wrap(s: &str, w: usize) -> Vec<String> {
    let w = w.max(8);
    let mut lines = Vec::new();
    for para in s.split('\n') {
        let mut cur = String::new();
        for word in para.split(' ') {
            let mut word = word.to_string();
            while width(&word) > w {
                if !cur.is_empty() {
                    lines.push(std::mem::take(&mut cur));
                }
                let head: String = word.chars().take(w).collect();
                word = word.chars().skip(w).collect();
                lines.push(head);
            }
            let extra = if cur.is_empty() { 0 } else { 1 };
            if width(&cur) + extra + width(&word) > w {
                lines.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&word);
        }
        lines.push(cur);
    }
    lines
}

fn row(area: Rect, i: u16) -> Rect {
    Rect {
        x: area.x,
        y: area.y + i,
        width: area.width,
        height: 1,
    }
}

fn centered(area: Rect, w_pct: u16, h_pct: u16) -> Rect {
    let w = (area.width * w_pct / 100).max(40).min(area.width);
    let h = (area.height * h_pct / 100).max(12).min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn blink(app: &App) -> bool {
    (app.tick / 6).is_multiple_of(2)
}

fn spinner(app: &App) -> &'static str {
    SPINNER[(app.tick as usize) % SPINNER.len()]
}

/// Accent colour pushed most of the way toward the panel background.
fn dark(c: Color) -> Color {
    lerp(c, PANEL, 0.78)
}

fn state_color(s: AgentState) -> Color {
    match s {
        AgentState::Idle => DIM,
        AgentState::Thinking => MAGENTA,
        AgentState::Acting => CYAN,
        AgentState::Waiting => AMBER,
        AgentState::Done => NEON,
        AgentState::Failed => RED,
        AgentState::Reflecting => VIOLET,
    }
}

fn severity_glyph(s: Severity) -> (&'static str, Color) {
    match s {
        Severity::Info => ("●", CYAN),
        Severity::Success => ("✓", NEON),
        Severity::Warn => ("▲", AMBER),
        Severity::Error => ("✗", RED),
    }
}

fn risk_color(r: Risk) -> Color {
    match r {
        Risk::Low => NEON,
        Risk::Medium => AMBER,
        Risk::High => RED,
    }
}

/// Status badge for a tool call, padded to a fixed width so columns line up.
fn tool_badge(app: &App, t: &ToolCall) -> (String, Style) {
    let (label, style) = match t.status {
        ToolStatus::AwaitingApproval => (
            " ◷ AWAIT ".to_string(),
            if blink(app) {
                badge(BG, AMBER)
            } else {
                bold(AMBER)
            },
        ),
        ToolStatus::Running => (format!(" {} RUN ", spinner(app)), bold(CYAN)),
        ToolStatus::Ok => (format!(" ✓ {}ms ", t.duration_ms.unwrap_or(0)), bold(NEON)),
        ToolStatus::Failed => (" ✗ FAIL ".into(), bold(RED)),
        ToolStatus::Denied => (" ⊘ DENIED ".into(), bold(RED)),
    };
    (pad(&label, 10), style)
}

// ───────────────────────────── HUD primitives ─────────────────────────────

/// A slanted chip: `◢ LABEL ◤`. Solid chips use a left-to-right gradient fill;
/// muted chips use a dark fill with accent text. Width is `label + 2`.
fn chip(label: &str, accent: Color, solid: bool) -> Vec<Span<'static>> {
    let n = width(label).max(1) as f32;
    let fill_at = |i: usize| {
        if solid {
            lerp(accent, lerp(accent, MAGENTA, 0.55), i as f32 / n)
        } else {
            dark(accent)
        }
    };
    let mut spans = vec![Span::styled("◢", fg(fill_at(0)))];
    for (i, c) in label.chars().enumerate() {
        let style = if solid {
            Style::new().fg(BG).bg(fill_at(i))
        } else {
            Style::new().fg(accent).bg(fill_at(i))
        };
        spans.push(Span::styled(
            c.to_string(),
            style.add_modifier(Modifier::BOLD),
        ));
    }
    spans.push(Span::styled("◤", fg(fill_at(width(label)))));
    spans
}

fn chip_width(label: &str) -> u16 {
    width(label) as u16 + 2
}

/// Bright corner brackets over a dim frame.
fn corners(buf: &mut Buffer, r: Rect, c: Color) {
    if r.width < 6 || r.height < 2 {
        return;
    }
    let (x0, y0) = (r.x, r.y);
    let (x1, y1) = (r.x + r.width - 1, r.y + r.height - 1);
    let mut marks = vec![
        (x0, y0, "┏"),
        (x0 + 1, y0, "━"),
        (x1, y0, "┓"),
        (x1 - 1, y0, "━"),
        (x0, y1, "┗"),
        (x0 + 1, y1, "━"),
        (x1, y1, "┛"),
        (x1 - 1, y1, "━"),
    ];
    if r.height >= 4 {
        marks.extend([
            (x0, y0 + 1, "┃"),
            (x1, y0 + 1, "┃"),
            (x0, y1 - 1, "┃"),
            (x1, y1 - 1, "┃"),
        ]);
    }
    for (x, y, s) in marks {
        buf[(x, y)].set_symbol(s).set_fg(c);
    }
}

/// Draws a HUD panel and returns its interior.
fn hud(
    f: &mut Frame,
    area: Rect,
    title: &str,
    accent: Color,
    focused: bool,
    bottom: Option<Line<'static>>,
) -> Rect {
    let rule = if focused {
        lerp(accent, GRID, 0.45)
    } else {
        GRID
    };
    let mut block = Block::bordered()
        .border_type(BorderType::Plain)
        .border_style(fg(rule))
        .style(Style::new().bg(PANEL));
    if let Some(b) = bottom {
        block = block.title_bottom(b.right_aligned());
    }
    let inner = block.inner(area);
    f.render_widget(block, area);
    corners(
        f.buffer_mut(),
        area,
        if focused {
            accent
        } else {
            lerp(accent, GRID, 0.4)
        },
    );
    let label = format!(" {title} ");
    let w = chip_width(&label).min(area.width.saturating_sub(4));
    f.render_widget(
        Paragraph::new(Line::from(chip(&label, accent, focused))),
        Rect {
            x: area.x + 2,
            y: area.y,
            width: w,
            height: 1,
        },
    );
    inner
}

/// Segmented gauge: `LABEL ▰▰▰▱▱▱ detail`.
fn segmented(label: &str, ratio: f64, detail: &str, color: Color, w: u16) -> Line<'static> {
    let ratio = ratio.clamp(0.0, 1.0);
    let head = format!(" {label} ");
    let tail = format!(" {detail} ");
    let n = (w as usize)
        .saturating_sub(width(&head) + width(&tail) + 1)
        .max(4);
    let filled = (ratio * n as f64).round() as usize;
    let mut spans = vec![Span::styled(head, bold(color))];
    for i in 0..n {
        if i < filled {
            let c = lerp(dark(color), color, (i + 1) as f32 / filled.max(1) as f32);
            spans.push(Span::styled("▰", fg(c)));
        } else {
            spans.push(Span::styled("▱", fg(GRID)));
        }
    }
    spans.push(Span::styled(tail, fg(DIM)));
    Line::from(spans)
}

/// Eight-cell activity pulse for an agent.
fn pulse(app: &App, state: AgentState, seed: u32) -> Vec<Span<'static>> {
    const BARS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let c = state_color(state);
    (0..8)
        .map(|i| {
            let phase = app.tick as f32 * 0.55 + i as f32 * 0.9 + seed as f32 * 2.1;
            match state {
                AgentState::Thinking | AgentState::Acting => {
                    let v = (phase.sin() * 0.5 + 0.5) * (phase * 0.37).cos().abs();
                    let lvl = (v * 7.0).round() as usize;
                    Span::styled(BARS[lvl.min(7)], fg(lerp(dark(c), c, v)))
                }
                AgentState::Waiting => {
                    let beat = (app.tick / 3) as usize % 12 == i;
                    Span::styled(
                        if beat { "▃" } else { "▁" },
                        fg(if beat { c } else { dark(c) }),
                    )
                }
                AgentState::Reflecting => {
                    // Slow, broad swell: low-frequency activity.
                    let v = (phase * 0.35).sin() * 0.5 + 0.5;
                    let lvl = (v * 4.0).round() as usize;
                    Span::styled(BARS[lvl.min(7)], fg(lerp(dark(c), c, v)))
                }
                _ => Span::styled("─", fg(dark(c))),
            }
        })
        .collect()
}

/// A smooth curve through `values`, read at a fractional index. Catmull-Rom,
/// held between its two nearest samples so it never overshoots them.
fn smooth(values: &[f64], x: f64) -> f64 {
    let Some(last) = values.len().checked_sub(1) else {
        return 0.0;
    };
    let x = x.clamp(0.0, last as f64);
    let i = x.floor() as usize;
    let t = x - i as f64;
    let at = |k: isize| values[(i as isize + k).clamp(0, last as isize) as usize];
    let (p0, p1, p2, p3) = (at(-1), at(0), at(1), at(2));
    let v = 0.5
        * (2.0 * p1
            + (p2 - p0) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t * t);
    v.clamp(p1.min(p2), p1.max(p2))
}

/// Draws a filled area chart at braille resolution: two dots across and four
/// up per cell, so the line is eight times finer than block characters.
/// `height_at` maps 0..1 across the chart to 0..1 up it. The line itself is
/// drawn in `color`, the area under it dimmed.
fn area_chart(buf: &mut Buffer, area: Rect, color: Color, height_at: impl Fn(f64) -> f64) {
    // Braille dot bits by [column][row from the top].
    const DOT: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    if area.width == 0 || area.height == 0 {
        return;
    }
    let (w, h) = (area.width as usize, area.height as usize);
    let (cols, rows) = (w * 2, h * 4);
    // Dots lit in each dot-column, counted up from the bottom. Always at
    // least one, so a quiet graph is a continuous baseline, not a gap.
    let lit: Vec<usize> = (0..cols)
        .map(|c| {
            let height = height_at(c as f64 / (cols - 1).max(1) as f64).clamp(0.0, 1.0);
            (height * (rows - 1) as f64).round() as usize + 1
        })
        .collect();
    let fill = lerp(color, PANEL, 0.7);
    for cx in 0..w {
        for cy in 0..h {
            let mut bits = 0;
            let mut on_line = false;
            for (sub, column) in DOT.iter().enumerate() {
                let n = lit[cx * 2 + sub];
                for (r, bit) in column.iter().enumerate() {
                    let from_bottom = (h - 1 - cy) * 4 + (3 - r);
                    if from_bottom < n {
                        bits |= bit;
                        on_line |= from_bottom + 1 == n;
                    }
                }
            }
            if bits == 0 {
                continue;
            }
            let symbol = char::from_u32(0x2800 + bits).unwrap_or(' ');
            buf[(area.x + cx as u16, area.y + cy as u16)]
                .set_char(symbol)
                .set_fg(if on_line { color } else { fill });
        }
    }
}

/// Fills fully empty rows of the given regions with a faint dot grid.
fn texture(buf: &mut Buffer, regions: &[Rect]) {
    for r in regions {
        for y in r.y..r.y + r.height {
            let empty = (r.x..r.x + r.width)
                .all(|x| buf[(x, y)].symbol() == " " && buf[(x, y)].bg != SELECT);
            if !empty {
                continue;
            }
            for x in r.x..r.x + r.width {
                if (x + 2 * (y % 2)) % 4 == 0 {
                    buf[(x, y)].set_symbol("·").set_fg(TEXTURE);
                }
            }
        }
    }
}

// ───────────────────────────── entry point ─────────────────────────────

pub fn render(f: &mut Frame, app: &mut App) {
    app.hits = Hits::default();
    let area = f.area();
    f.render_widget(Block::new().style(Style::new().bg(BG).fg(TEXT)), area);

    if app.booting() {
        render_boot(f, app, area);
        return;
    }

    let [header, body, signals, input] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(10),
        Constraint::Length(4),
        Constraint::Length(1),
    ])
    .areas(area);

    render_header(f, app, header);

    let side = if area.width >= 150 { 40 } else { 34 };
    let left_w = if area.width >= 150 { 32 } else { 28 };
    let [left, center, right] = Layout::horizontal([
        Constraint::Length(left_w),
        Constraint::Min(30),
        Constraint::Length(side),
    ])
    .areas(body);

    render_left(f, app, left);
    render_center(f, app, center);
    render_right(f, app, right);
    render_signals(f, app, signals);
    render_input(f, app, input);

    let regions = std::mem::take(&mut app.hits.texture);
    texture(f.buffer_mut(), &regions);

    if let Some(m) = app.modal {
        render_modal(f, app, m, area);
    }
    // Over everything else: nothing behind it takes a key while it is open.
    if app.secret_form.is_some() {
        render_secret_form(f, app, area);
    }
}

/// The window a secret is typed into. The value is never drawn: only how
/// many characters have been typed.
fn render_secret_form(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(form) = app.secret_form.clone() else {
        return;
    };
    let w = area.width.clamp(40, 84);
    let h = 13.min(area.height);
    let r = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w.min(area.width),
        height: h,
    };
    app.hits.modal = Some(r);
    f.render_widget(Clear, r);
    let inner = hud(f, r, "SECRET", AMBER, true, None);
    let iw = inner.width.saturating_sub(2) as usize;
    let field = |label: &str, shown: String, active: bool| {
        Line::from(vec![
            Span::styled(
                format!(" {label:<6}"),
                bold(if active { AMBER } else { DIM }),
            ),
            Span::styled(if active { "❯ " } else { "  " }, bold(AMBER)),
            Span::styled(shown, if active { bold(TEXT) } else { fg(TEXT) }),
            Span::styled(if active { "▌" } else { "" }, fg(AMBER)),
        ])
    };
    let dots = form.value.chars().count();
    let masked = if dots == 0 {
        String::new()
    } else {
        format!("{} ({dots} characters)", "•".repeat(dots.min(24)))
    };
    let mut lines = vec![
        Line::styled(
            " Typed here, it goes straight to the secret store on this machine.",
            fg(TEXT),
        ),
        Line::styled(
            " It is never sent to a model, shown on screen or written to the timeline.",
            fg(TEXT),
        ),
        Line::from(""),
    ];
    if form.asked {
        for l in wrap(
            &format!(
                "The agent asks for it{}",
                if form.why.is_empty() {
                    ".".to_string()
                } else {
                    format!(": {}", form.why)
                }
            ),
            iw,
        )
        .into_iter()
        .take(2)
        {
            lines.push(Line::styled(format!(" {l}"), fg(DIM)));
        }
    } else {
        lines.push(Line::styled(
            " Name it in capitals, like GITHUB_TOKEN or TELEGRAM_BOT_TOKEN.",
            fg(DIM),
        ));
    }
    lines.push(Line::from(""));
    lines.push(field("NAME", form.name.clone(), !form.on_value));
    lines.push(field("VALUE", masked, form.on_value));
    lines.push(Line::from(""));
    lines.push(if form.error.is_empty() {
        Line::styled(
            if form.asked {
                " ⏎ store it · esc do not give it"
            } else {
                " tab switch field · ⏎ store it · esc cancel"
            },
            fg(DIM),
        )
    } else {
        Line::styled(format!(" {}", trunc(&form.error, iw)), bold(RED))
    });
    f.render_widget(Paragraph::new(lines), inner);
}

// ───────────────────────────── boot ─────────────────────────────

const LOGO: [&str; 2] = [
    "▄▀▀ ▄▀▄ █▄ █ ▄▀▀ ▄▀▀ █ ▄▀▄ █ █ ▄▀▀",
    "▀▄▄ ▀▄▀ █ ▀█ ▄▄▀ ▀▄▄ █ ▀▄▀ ▀▄▀ ▄▄▀",
];

const NOISE: [&str; 12] = ["▓", "▒", "░", "/", "\\", "#", "%", "0", "1", "<", ">", "|"];

fn logo_lines(tick: u64) -> Vec<Line<'static>> {
    let glitch = tick % 37 < 2;
    LOGO.iter()
        .enumerate()
        .map(|(row_i, l)| {
            let n = width(l).max(1) as f32;
            let mut spans: Vec<Span> = Vec::new();
            if glitch && row_i == 1 {
                spans.push(Span::raw(" "));
            }
            spans.extend(l.chars().enumerate().map(|(i, c)| {
                let tt = ((i as f32 / n) + tick as f32 * 0.02).fract();
                let tt = if tt > 0.5 { 1.0 - tt } else { tt } * 2.0;
                let color = if glitch && row_i == 1 {
                    RED
                } else {
                    lerp(CYAN, MAGENTA, tt)
                };
                Span::styled(c.to_string(), bold(color))
            }));
            Line::from(spans)
        })
        .collect()
}

/// Text that resolves from noise left to right as `reveal` goes 0 → 1.
fn decode(text: &str, reveal: f32, tick: u64, color: Color) -> Vec<Span<'static>> {
    let n = width(text);
    let shown = (reveal.clamp(0.0, 1.0) * n as f32) as usize;
    text.chars()
        .enumerate()
        .map(|(j, c)| {
            if j < shown || c == ' ' {
                Span::styled(c.to_string(), fg(color))
            } else {
                let k = (j * 31 + tick as usize * 17) % NOISE.len();
                Span::styled(NOISE[k], fg(dark(MAGENTA)))
            }
        })
        .collect()
}

fn render_boot(f: &mut Frame, app: &App, area: Rect) {
    let elapsed = app.boot.map(|b| b.elapsed()).unwrap_or_default();
    let step = BOOT_STEP.as_millis() as f32;
    let shown = ((elapsed.as_millis() as f32 / step) as usize).min(BOOT_LINES.len());

    let mut lines = logo_lines(app.tick);
    lines.push(Line::from(""));
    let sub = "goal-driven agent harness  //  prototype · simulated data";
    lines.push(Line::from(decode(
        sub,
        elapsed.as_millis() as f32 / (step * 3.0),
        app.tick,
        DIM,
    )));
    lines.push(Line::from(""));
    for (i, (name, detail)) in BOOT_LINES.iter().take(shown).enumerate() {
        let since = elapsed.as_millis() as f32 - step * (i as f32 + 1.0);
        let reveal = (since / (step * 1.2)).max(0.0) + 0.15;
        let ok = reveal >= 1.0;
        let mut spans = vec![
            Span::styled("[ ", fg(DIM)),
            Span::styled(
                if ok { "OK" } else { spinner(app) },
                bold(if ok { NEON } else { MAGENTA }),
            ),
            Span::styled(if ok { " ] " } else { "  ] " }, fg(DIM)),
            Span::styled(format!("{name:.<14} "), fg(TEXT)),
        ];
        spans.extend(decode(detail, reveal, app.tick, CYAN));
        lines.push(Line::from(spans));
    }
    if shown < BOOT_LINES.len() {
        lines.push(Line::styled(if blink(app) { "█" } else { " " }, fg(NEON)));
    }
    let h = lines.len() as u16;
    let w = 66.min(area.width);
    let r = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + area.height.saturating_sub(h) / 2,
        width: w,
        height: h.min(area.height),
    };
    let frame = Rect {
        x: r.x.saturating_sub(4),
        y: r.y.saturating_sub(2),
        width: (r.width + 8).min(area.width),
        height: (r.height + 4).min(area.height),
    };
    corners(f.buffer_mut(), frame, lerp(CYAN, GRID, 0.3));
    f.render_widget(Paragraph::new(lines), r);
    f.render_widget(
        Paragraph::new(Line::styled("press any key", fg(DIM))).alignment(Alignment::Center),
        row(area, area.height.saturating_sub(2)),
    );
}

// ───────────────────────────── header ─────────────────────────────

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let (state, alarm) = app.goal_state();
    let state_color = match state {
        "RUNNING" => CYAN,
        "VERIFIED" => NEON,
        "BLOCKED" => RED,
        "IDLE" => DIM,
        _ => AMBER,
    };
    let solid = !alarm || blink(app);
    let title = app
        .goal
        .as_ref()
        .map(|g| g.title.clone())
        .unwrap_or_else(|| "no goal set".into());
    let busy = app.agents.iter().any(|a| a.state.is_active());

    let top = row(area, 0);
    f.render_widget(Block::new().style(Style::new().bg(BAND)), top);

    let mut left = chip(" ◈ CONSCIOUS ", NEON, true);
    left.push(Span::raw(" "));
    // What is behind the screen: the scripted demo, or how much is going on.
    if app.live {
        let open = app.open_tasks();
        let label = match open {
            0 => " IDLE ".to_string(),
            n => format!(" {n} BRANCH{} OPEN ", if n == 1 { "" } else { "ES" }),
        };
        left.extend(chip(&label, if open == 0 { DIM } else { CYAN }, false));
    } else {
        left.extend(chip(" DEMO ", AMBER, false));
    }
    left.push(Span::styled(
        format!("  {} ", if busy { spinner(app) } else { "·" }),
        fg(NEON),
    ));
    let tok_s = app.throughput.last().copied().unwrap_or(0);
    let sep = || Span::styled("  ╱  ", fg(GRID));
    let right = Line::from(vec![
        Span::styled("⌁ ", fg(MAGENTA)),
        Span::styled(
            // The demo pretends to run a local model on the processor.
            format!("{}{}", app.last_model, if app.live { "" } else { "·cpu" }),
            fg(TEXT),
        ),
        sep(),
        Span::styled(format!("▲ {tok_s} tok/s"), fg(NEON)),
        sep(),
        Span::styled(
            format!("Σ {}", tok(app.tokens_in + app.tokens_out)),
            fg(CYAN),
        ),
        sep(),
        Span::styled("$0.000", fg(TEXT)),
        sep(),
        Span::styled(format!("{} ", t(app.started.elapsed())), bold(DIM)),
    ]);

    let label = match app.shown_task() {
        Some(task) if task.id == MAIN => "MAIN :: ".to_string(),
        Some(task) => format!("BRANCH #{} :: ", task.id),
        None => "GOAL :: ".to_string(),
    };
    let state_chip = chip(&format!(" {state} "), state_color, solid);
    // The title gets what is left between the fixed parts and the figures
    // on the right, so it never runs under them.
    let taken: usize = left.iter().map(|s| s.width()).sum::<usize>()
        + width(&label)
        + state_chip.iter().map(|s| s.width()).sum::<usize>()
        + right.width()
        + 6;
    left.push(Span::styled(label, fg(DIM)));
    left.push(Span::styled(
        trunc(&title, (area.width as usize).saturating_sub(taken).max(8)),
        bold(TEXT),
    ));
    left.push(Span::raw("  "));
    left.extend(state_chip);
    f.render_widget(Paragraph::new(Line::from(left)), top);
    f.render_widget(Paragraph::new(right).alignment(Alignment::Right), top);

    // Segmented gauges.
    let [a, b, c] = Layout::horizontal([Constraint::Ratio(1, 3); 3]).areas(row(area, 1));
    let budget = app
        .goal
        .as_ref()
        .map(|g| g.token_budget)
        .unwrap_or(1)
        .max(1);
    let used = app.tokens_in + app.tokens_out;
    let ctx_used = app.context_used() as u64;
    let done = app
        .plan
        .iter()
        .filter(|s| s.status == StepStatus::Done)
        .count();
    let plan_ratio = if app.plan.is_empty() {
        0.0
    } else {
        done as f64 / app.plan.len() as f64
    };
    f.render_widget(
        Paragraph::new(segmented(
            "BUDGET",
            used as f64 / budget as f64,
            &format!("{}/{}", tok(used), tok(budget)),
            NEON,
            a.width,
        )),
        a,
    );
    f.render_widget(
        Paragraph::new(segmented(
            "CONTEXT",
            ctx_used as f64 / app.window.max(1) as f64,
            &format!("{}/{}", tok(ctx_used), tok(app.window as u64)),
            CYAN,
            b.width,
        )),
        b,
    );
    f.render_widget(
        Paragraph::new(segmented(
            "PLAN",
            plan_ratio,
            &format!("{done}/{} steps", app.plan.len()),
            MAGENTA,
            c.width,
        )),
        c,
    );
}

// ───────────────────────────── left: agents + plan ─────────────────────────────

fn render_left(f: &mut Frame, app: &mut App, area: Rect) {
    let agents_h = (app.agents.len() as u16 * 2 + 2).clamp(6, 16);
    let [top, bottom] =
        Layout::vertical([Constraint::Length(agents_h), Constraint::Min(4)]).areas(area);

    let focused = app.focus == Panel::Agents;
    let bottom_title = app.agent_filter.map(|id| {
        Line::from(vec![
            Span::styled(" filter ", fg(DIM)),
            Span::styled(app.agent_name(id), bold(AMBER)),
            Span::styled(" (f) ", fg(DIM)),
        ])
    });
    let inner = hud(f, top, "AGENTS", VIOLET, focused, bottom_title);
    app.hits.panels.push((top, Panel::Agents));

    let w = inner.width as usize;
    let count = app.agents.len();
    for (i, a) in app.agents.iter().enumerate() {
        let y = i as u16 * 2;
        if y + 1 >= inner.height {
            break;
        }
        let depth = {
            let mut d = 0;
            let mut p = a.parent;
            while let Some(pid) = p {
                d += 1;
                p = app
                    .agents
                    .iter()
                    .find(|x| x.id == pid)
                    .and_then(|x| x.parent);
            }
            d
        };
        let last_sibling = !app.agents[i + 1..count]
            .iter()
            .any(|x| x.parent == a.parent);
        let (branch, cont) = if depth == 0 {
            (String::new(), String::new())
        } else {
            let indent = "  ".repeat(depth - 1);
            (
                format!("{indent}{} ", if last_sibling { "└─" } else { "├─" }),
                format!("{indent}{}  ", if last_sibling { " " } else { "│" }),
            )
        };
        let st = a.state;
        let sc = state_color(st);
        let glyph = if depth == 0 { "◉ " } else { "◎ " };
        let state = if st.is_active() {
            format!("{} {}", spinner(app), st.label())
        } else {
            st.label().to_string()
        };
        let name_w = w.saturating_sub(width(&branch) + 2 + 8);
        let selected = i == app.agent_sel && focused;
        let filtered = app.agent_filter == Some(a.id);
        let l1 = Line::from(vec![
            Span::styled(branch, fg(lerp(VIOLET, GRID, 0.3))),
            Span::styled(glyph, bold(sc)),
            Span::styled(
                pad(&a.name.to_uppercase(), name_w),
                if filtered { bold(AMBER) } else { bold(TEXT) },
            ),
            Span::styled(format!("{state:>8}"), bold(sc)),
        ]);
        let mut s2 = vec![Span::styled(
            format!("{cont}  "),
            fg(lerp(VIOLET, GRID, 0.3)),
        )];
        s2.extend(pulse(app, st, a.id));
        s2.push(Span::styled(format!(" {} tok", tok(a.tokens)), fg(DIM)));
        let l2 = Line::from(s2);
        let (l1, l2) = if selected {
            let s = Style::new().bg(SELECT);
            (l1.style(s), l2.style(s))
        } else {
            (l1, l2)
        };
        let r = Rect {
            height: 2,
            ..row(inner, y)
        };
        f.render_widget(Paragraph::new(vec![l1, l2]), r);
        app.hits.agents.push((r, i));
    }
    if app.agents.is_empty() {
        f.render_widget(
            Paragraph::new(Line::styled("no agents online", fg(DIM))),
            inner,
        );
    }
    app.hits.texture.push(inner);

    // Plan + criteria.
    let title = if app.plan_version > 0 {
        format!("PLAN v{}", app.plan_version)
    } else {
        "PLAN".into()
    };
    let inner = hud(f, bottom, &title, MAGENTA, false, None);
    let mut lines: Vec<Line> = Vec::new();
    let verified = app.goal_state().0 == "VERIFIED";
    let iw = inner.width as usize;
    if let Some(g) = &app.goal {
        lines.push(Line::styled("// CRITERIA", bold(DIM)));
        for c in &g.criteria {
            let (gl, col) = if verified {
                ("✓ ", NEON)
            } else {
                ("◇ ", AMBER)
            };
            for (j, l) in wrap(c, iw.saturating_sub(2)).into_iter().enumerate() {
                lines.push(Line::from(vec![
                    Span::styled(if j == 0 { gl } else { "  " }, fg(col)),
                    Span::styled(l, fg(TEXT)),
                ]));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::styled("// STEPS", bold(DIM)));
    }
    for (i, s) in app.plan.iter().enumerate() {
        let (gl, col) = match s.status {
            StepStatus::Pending => ("○".to_string(), DIM),
            StepStatus::Active => (spinner(app).to_string(), CYAN),
            StepStatus::Done => ("✓".to_string(), NEON),
            StepStatus::Failed => ("✗".to_string(), RED),
        };
        let text_style = match s.status {
            StepStatus::Done => fg(DIM),
            StepStatus::Active => bold(TEXT),
            StepStatus::Failed => fg(RED),
            StepStatus::Pending => fg(TEXT),
        };
        for (j, l) in wrap(&s.text, iw.saturating_sub(5)).into_iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    if j == 0 {
                        format!("{gl} {:02} ", i + 1)
                    } else {
                        "     ".into()
                    },
                    fg(col),
                ),
                Span::styled(l, text_style),
            ]));
        }
    }
    if lines.is_empty() {
        lines.push(Line::styled("waiting for a plan…", fg(DIM)));
    }
    f.render_widget(Paragraph::new(lines), inner);
    app.hits.texture.push(inner);
}

// ───────────────────────────── center: timeline / events / memory ─────────────────────────────

const PREFIX_W: usize = 21;

/// When an event happened, as shown in the timeline: time into this
/// session, or the clock time for events restored from earlier ones.
#[derive(Clone, Copy)]
struct Stamp {
    at: Duration,
    /// Local seconds since midnight, for a restored event.
    clock: Option<u64>,
}

fn stamp(app: &App, ev: &crate::event::Event) -> Stamp {
    Stamp {
        at: ev.at,
        clock: ev
            .restored
            .then(|| (ev.wall as i64 + app.utc_offset).rem_euclid(86_400) as u64),
    }
}

fn prefix(at: Stamp, glyph: &str, label: &str, color: Color) -> Vec<Span<'static>> {
    let when = match at.clock {
        // Same width as `T+mm:ss`, dimmer: it is the past.
        Some(s) => format!("{:02}:{:02}  ", s / 3600, s % 3600 / 60),
        None => t(at.at),
    };
    vec![
        Span::styled(
            "▍",
            fg(if at.clock.is_some() {
                dark(color)
            } else {
                color
            }),
        ),
        Span::styled(format!("{when} "), fg(DIM)),
        Span::styled(format!("{glyph} "), fg(color)),
        Span::styled(format!("{} ", pad(label, 8)), bold(color)),
    ]
}

fn continuation(color: Color) -> Vec<Span<'static>> {
    vec![
        Span::styled("▏", fg(dark(color))),
        Span::styled(format!("{:>w$}", "┊ ", w = PREFIX_W - 1), fg(GRID)),
    ]
}

fn body_lines(
    head: Vec<Span<'static>>,
    text: &str,
    style: Style,
    w: usize,
    color: Color,
) -> Vec<Line<'static>> {
    let avail = w.saturating_sub(PREFIX_W);
    wrap(text, avail)
        .into_iter()
        .enumerate()
        .map(|(i, l)| {
            let mut spans = if i == 0 {
                head.clone()
            } else {
                continuation(color)
            };
            spans.push(Span::styled(l, style));
            Line::from(spans)
        })
        .collect()
}

/// Whether an event belongs in the activity view: what was said, what was
/// answered, what a branch brought back, and what went wrong. The user
/// asked for this (2026-11-14): the steps in between were too much.
///
/// A branch's tab also keeps what the agent said on the way (`Thought`):
/// that text was on screen while it was being written, and it must not
/// vanish when the model call that wrote it ends and the tab is left and
/// come back to. Main stays the conversation only.
fn in_conversation(kind: &EventKind, in_branch: bool) -> bool {
    match kind {
        EventKind::Message { .. } | EventKind::Merged { .. } => true,
        EventKind::Thought { .. } => in_branch,
        EventKind::Alert { severity, .. } => {
            matches!(severity, Severity::Warn | Severity::Error)
        }
        _ => false,
    }
}

fn timeline_entry(app: &App, idx: usize, w: usize) -> Vec<Line<'static>> {
    let ev = &app.events[idx];
    let at = stamp(app, ev);
    let name = |id: u32| app.agent_name(id).to_uppercase();
    match &ev.kind {
        EventKind::TaskOpened { .. } if ev.task == Some(MAIN) => body_lines(
            prefix(at, "▣", "MAIN", AMBER),
            "the main conversation: it keeps going, and branches off when something is work of its own",
            fg(DIM),
            w,
            AMBER,
        ),
        EventKind::TaskOpened { title, origin } => body_lines(
            prefix(at, "▣", "BRANCH", AMBER),
            &format!(
                "#{} opened by {origin}: {title}",
                ev.task.unwrap_or_default()
            ),
            bold(TEXT),
            w,
            AMBER,
        ),
        EventKind::Merged {
            branch,
            title,
            result,
            ok,
        } => {
            let color = if *ok { NEON } else { AMBER };
            let mut v = body_lines(
                prefix(at, "◂", "MERGED", color),
                &if *ok {
                    format!("branch #{branch} ended and is merged into main: {title}")
                } else {
                    format!("branch #{branch} ended without an answer, nothing to merge: {title}")
                },
                bold(TEXT),
                w,
                color,
            );
            // What it brought back, as the branch wrote it.
            for row in markdown::render(result, w.saturating_sub(PREFIX_W)) {
                let mut spans = continuation(color);
                spans.extend(row);
                v.push(Line::from(spans));
            }
            v
        }
        EventKind::TaskUpdate { state, note } => {
            let (glyph, color) = task_mark(*state);
            let label = match state {
                TaskState::Researching => "paused for knowledge research",
                TaskState::Queued => "queued",
                TaskState::Working => "working",
                TaskState::Done => "done",
                TaskState::Failed => "failed",
                TaskState::Stopped => "stopped",
            };
            body_lines(
                prefix(at, glyph, "STATE", color),
                &format!("{label}: {note}"),
                fg(if state.is_open() { TEXT } else { color }),
                w,
                color,
            )
        }
        EventKind::CriteriaChecked { checked } => {
            use crate::goal::Verdict;
            let all = checked.iter().all(|c| c.verdict == Verdict::Met);
            let color = if all { NEON } else { AMBER };
            let mut v = body_lines(
                prefix(at, "◇", "JUDGED", color),
                &crate::goal::tally(checked),
                bold(TEXT),
                w,
                color,
            );
            for c in checked {
                // A word as well as a mark, so it does not rest on colour.
                let (mark, c_color) = match c.verdict {
                    Verdict::Met => ("✓ met", NEON),
                    Verdict::NotMet => ("✗ not met", RED),
                    Verdict::Unknown => ("? unknown", DIM),
                };
                let mut head = continuation(color);
                head[1] = Span::styled(format!("{:>w$}", "└ ", w = PREFIX_W - 1), fg(c_color));
                let evidence = if c.evidence.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", c.evidence)
                };
                v.extend(body_lines(
                    head,
                    &format!("{mark}: {}{evidence}", c.criterion),
                    fg(if c.verdict == Verdict::Met { DIM } else { TEXT }),
                    w,
                    color,
                ));
            }
            v
        }
        EventKind::Research {
            phase,
            text,
            source,
        } => {
            let label = match phase.as_str() {
                "search" => "SEARCH",
                "read" => "READ",
                "kept" => "KEPT",
                "known" => "KNOWN",
                _ => "RESEARCH",
            };
            let color = if phase == "failed" { AMBER } else { VIOLET };
            let mut v = body_lines(prefix(at, "◈", label, color), text, fg(TEXT), w, color);
            if !source.is_empty() {
                let mut head = continuation(color);
                head[1] = Span::styled(format!("{:>w$}", "└ ", w = PREFIX_W - 1), fg(color));
                v.extend(body_lines(head, source, fg(DIM), w, color));
            }
            v
        }
        EventKind::GoalSet {
            title, criteria, ..
        } => {
            let mut v = body_lines(prefix(at, "★", "GOAL", AMBER), title, bold(TEXT), w, AMBER);
            for c in criteria {
                let mut head = continuation(AMBER);
                head[1] = Span::styled(format!("{:>w$}", "◇ ", w = PREFIX_W - 1), fg(AMBER));
                v.extend(body_lines(head, c, fg(DIM), w, AMBER));
            }
            v
        }
        EventKind::PlanUpdated { version, steps } => body_lines(
            prefix(at, "■", &format!("PLAN v{version}"), MAGENTA),
            &steps
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{:02} {s}", i + 1))
                .collect::<Vec<_>>()
                .join("  ·  "),
            fg(TEXT),
            w,
            MAGENTA,
        ),
        EventKind::StepUpdate { index, status } => {
            let text = app
                .plan
                .get(*index)
                .map(|s| s.text.clone())
                .unwrap_or_default();
            let (g, l, c) = match status {
                StepStatus::Done => ("✓", "DONE", NEON),
                StepStatus::Failed => ("✗", "FAILED", RED),
                StepStatus::Active => ("▶", "STEP", CYAN),
                StepStatus::Pending => return Vec::new(),
            };
            body_lines(
                prefix(at, g, l, c),
                &format!("{:02} {text}", index + 1),
                fg(DIM),
                w,
                c,
            )
        }
        EventKind::AgentSpawned {
            parent: Some(_),
            name: n,
            purpose,
            ..
        } => body_lines(
            prefix(at, "+", "SPAWN", VIOLET),
            &format!("{} — {purpose}", n.to_uppercase()),
            fg(TEXT),
            w,
            VIOLET,
        ),
        EventKind::Message { agent: None, text } => {
            body_lines(prefix(at, "▸", "YOU", NEON), text, bold(TEXT), w, NEON)
        }
        EventKind::Message {
            agent: Some(a),
            text,
        } => {
            // Replies are markdown: render it rather than showing the markup.
            let head = prefix(at, "◆", &name(*a), CYAN);
            markdown::render(text, w.saturating_sub(PREFIX_W))
                .into_iter()
                .enumerate()
                .map(|(i, row)| {
                    let mut spans = if i == 0 {
                        head.clone()
                    } else {
                        continuation(CYAN)
                    };
                    spans.extend(row);
                    Line::from(spans)
                })
                .collect()
        }
        EventKind::KnowledgeLookup { picks, .. } => {
            let direct = picks.iter().filter(|p| p.hops == 0).count();
            let mut v = body_lines(
                prefix(at, "◈", "LOOKUP", VIOLET),
                &format!(
                    "{} passage{} from the knowledge graph: {direct} matched your words, {} reached over links · press g to see how",
                    picks.len(),
                    if picks.len() == 1 { "" } else { "s" },
                    picks.len() - direct
                ),
                fg(TEXT),
                w,
                VIOLET,
            );
            for p in picks {
                let mut head = continuation(VIOLET);
                head[1] = Span::styled(format!("{:>w$}", "└ ", w = PREFIX_W - 1), fg(VIOLET));
                let how = match p.hops {
                    0 => "direct".to_string(),
                    1 => "1 link away".to_string(),
                    n => format!("{n} links away"),
                };
                v.extend(body_lines(
                    head,
                    &format!("{} · {how}", p.title),
                    fg(DIM),
                    w,
                    VIOLET,
                ));
            }
            v
        }
        EventKind::Thought { agent, text } => body_lines(
            prefix(at, "∴", &app.agent_name(*agent), DIM),
            text,
            Style::new().fg(DIM).add_modifier(Modifier::ITALIC),
            w,
            DIM,
        ),
        EventKind::ModelCall {
            agent,
            model,
            tokens_in,
            cached_in,
            tokens_out,
            latency_ms,
        } => {
            let rate = *tokens_out as f64 / (*latency_ms as f64 / 1000.0).max(0.001);
            body_lines(
                prefix(at, "⌁", &app.agent_name(*agent), GRID_HOT),
                &format!(
                    "{model}  in {} ({} cached)  out {}  · {rate:.1} tok/s · {:.1}s",
                    tok(*tokens_in as u64),
                    tok(*cached_in as u64),
                    tokens_out,
                    *latency_ms as f64 / 1000.0
                ),
                fg(DIM),
                w,
                GRID_HOT,
            )
        }
        EventKind::ToolRequested { id, .. } => {
            let Some(tc) = app.tool(*id) else {
                return Vec::new();
            };
            let (b, bs) = tool_badge(app, tc);
            let avail = w.saturating_sub(PREFIX_W + width(&tc.tool) + width(&b) + 5);
            let args = tc.args.lines().next().unwrap_or("").to_string();
            let mut spans = prefix(at, "λ", &name(tc.agent), CYAN);
            spans.push(Span::styled(format!("#{} ", tc.id), fg(DIM)));
            spans.push(Span::styled(tc.tool.clone(), bold(CYAN)));
            spans.push(Span::styled(format!(" {} ", pad(&args, avail)), fg(TEXT)));
            spans.push(Span::styled(b, bs));
            vec![Line::from(spans)]
        }
        EventKind::MemoryRead { query, hits, .. } => body_lines(
            prefix(at, "◈", "RECALL", VIOLET),
            &format!(
                "\"{query}\" → {}",
                hits.iter()
                    .map(|(id, _)| format!("#{id}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            fg(TEXT),
            w,
            VIOLET,
        ),
        EventKind::MemoryWrite { node, .. } => body_lines(
            prefix(at, "✦", "LEARNED", NEON),
            &format!("#{} [{}] {}", node.id, node.kind.label(), node.label),
            bold(TEXT),
            w,
            NEON,
        ),
        EventKind::MemoryIndexed { nodes } => body_lines(
            prefix(at, "◈", "MEMORY", VIOLET),
            &format!("{} long-term memories online", nodes.len()),
            fg(TEXT),
            w,
            VIOLET,
        ),
        EventKind::MemoryConsolidated { from, into, .. } => body_lines(
            prefix(at, "⊕", "MERGE", CYAN),
            &format!(
                "{} → #{} {}",
                from.iter()
                    .map(|id| format!("#{id}"))
                    .collect::<Vec<_>>()
                    .join(" + "),
                into.id,
                into.label
            ),
            fg(TEXT),
            w,
            CYAN,
        ),
        EventKind::Alert { severity, text } => {
            let (g, c) = severity_glyph(*severity);
            body_lines(prefix(at, g, "SIGNAL", c), text, bold(c), w, c)
        }
        _ => Vec::new(),
    }
}

fn raw_entry(app: &App, idx: usize, w: usize) -> Vec<Line<'static>> {
    let ev = &app.events[idx];
    let name = ev.kind.name();
    let color = match &ev.kind {
        EventKind::ToolRequested { .. }
        | EventKind::ToolDecision { .. }
        | EventKind::ToolFinished { .. } => CYAN,
        EventKind::ModelCall { .. } => GRID_HOT,
        EventKind::MemoryRead { .. }
        | EventKind::KnowledgeLookup { .. }
        | EventKind::MemoryWrite { .. }
        | EventKind::MemoryIndexed { .. }
        | EventKind::MemoryConsolidated { .. }
        | EventKind::Vitals { .. } => VIOLET,
        EventKind::Alert { .. } => AMBER,
        EventKind::Message { .. } | EventKind::Thought { .. } => TEXT,
        _ => MAGENTA,
    };
    let dbg = format!("{:?}", ev.kind).replace('\n', "⏎");
    let fields = dbg.strip_prefix(name).unwrap_or(&dbg).trim().to_string();
    let head = format!("0x{:04X} {} ", ev.seq, t(ev.at));
    let avail = w.saturating_sub(width(&head) + 15);
    vec![Line::from(vec![
        Span::styled("▍", fg(color)),
        Span::styled(head, fg(DIM)),
        Span::styled(pad(name, 14), bold(color)),
        Span::styled(trunc(&fields, avail), fg(TEXT)),
    ])]
}

fn memory_entry(app: &App, idx: usize, w: usize) -> Vec<Line<'static>> {
    let ev = &app.events[idx];
    match &ev.kind {
        EventKind::MemoryRead { agent, query, hits } => {
            let mut v = body_lines(
                prefix(stamp(app, ev), "◈", "RECALL", VIOLET),
                &format!("{} asked: \"{query}\"", app.agent_name(*agent)),
                bold(TEXT),
                w,
                VIOLET,
            );
            for (id, h) in hits {
                let mut head = continuation(VIOLET);
                head[1] = Span::styled(format!("{:>w$}", "└ ", w = PREFIX_W - 1), fg(VIOLET));
                v.extend(body_lines(head, &format!("#{id} {h}"), fg(TEXT), w, VIOLET));
            }
            v
        }
        EventKind::MemoryIndexed { .. }
        | EventKind::KnowledgeLookup { .. }
        | EventKind::MemoryWrite { .. }
        | EventKind::Research { .. }
        | EventKind::MemoryConsolidated { .. } => timeline_entry(app, idx, w),
        EventKind::TaskUpdate {
            state: TaskState::Researching,
            ..
        } => timeline_entry(app, idx, w),
        // What the model was actually given, and why each part was there.
        EventKind::Context { sections, window } => {
            let total: u32 = sections.iter().map(|s| s.tokens).sum();
            let mut v = body_lines(
                prefix(stamp(app, ev), "▤", "CONTEXT", CYAN),
                &format!(
                    "{} tokens given to the model, {:.1}% of its window",
                    tok(total as u64),
                    total as f64 * 100.0 / (*window).max(1) as f64
                ),
                bold(TEXT),
                w,
                CYAN,
            );
            for s in sections {
                let mut head = continuation(CYAN);
                head[1] = Span::styled(format!("{:>w$}", "└ ", w = PREFIX_W - 1), fg(CYAN));
                v.extend(body_lines(
                    head,
                    &format!("{} · {} tokens · {}", s.name, tok(s.tokens as u64), s.why),
                    fg(DIM),
                    w,
                    CYAN,
                ));
            }
            v
        }
        _ => Vec::new(),
    }
}

/// The glyph and colour that stand for a task's state. The glyph differs
/// per state, so the state does not rest on colour alone.
fn task_mark(state: TaskState) -> (&'static str, Color) {
    match state {
        TaskState::Queued => ("○", DIM),
        TaskState::Working => ("●", CYAN),
        TaskState::Researching => ("◈", VIOLET),
        TaskState::Done => ("✓", NEON),
        TaskState::Failed => ("✗", RED),
        TaskState::Stopped => ("■", AMBER),
    }
}

/// How a task stands right now, for its tab and its row on the board: a
/// call waiting for the user outranks the state it is otherwise in.
fn task_status(app: &App, task: &TaskView) -> (&'static str, &'static str, Color) {
    if app.task_needs_you(task.id) {
        return ("!", "NEEDS YOU", AMBER);
    }
    let (glyph, color) = task_mark(task.state);
    // Main does not finish: having answered, it is ready for what is next.
    if task.id == MAIN && task.state == TaskState::Done {
        return ("○", "READY", NEON);
    }
    (glyph, task.state.label(), color)
}

/// How long a task has taken (so far), in words.
fn task_time(task: &TaskView) -> String {
    let end = task.ended.unwrap_or_else(crate::learn::now);
    let secs = end.saturating_sub(task.opened);
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m {:02}s", s / 60, s % 60),
        s => format!("{}h {:02}m", s / 3600, s % 3600 / 60),
    }
}

/// What a task has used, in a line.
fn task_counts(task: &TaskView) -> String {
    let mut parts = vec![format!(
        "{} model call{}",
        task.calls,
        if task.calls == 1 { "" } else { "s" }
    )];
    if task.tokens > 0 {
        parts.push(format!("{} tok", tok(task.tokens)));
    }
    if task.tools > 0 {
        parts.push(format!(
            "{} tool call{}",
            task.tools,
            if task.tools == 1 { "" } else { "s" }
        ));
    }
    if task.lookups > 0 {
        parts.push(format!(
            "{} lookup{}",
            task.lookups,
            if task.lookups == 1 { "" } else { "s" }
        ));
    }
    if task.researched > 0 {
        parts.push(format!(
            "{} page{} researched",
            task.researched,
            if task.researched == 1 { "" } else { "s" }
        ));
    }
    parts.join(" · ")
}

/// The overview: every task, newest first, with where it stands and what
/// it is doing. Returns nothing drawn if there are no tasks.
fn render_board(f: &mut Frame, app: &mut App, inner: Rect) {
    let w = inner.width.saturating_sub(1) as usize;
    let all: Vec<TaskView> = app.branches().cloned().collect();
    let hidden = app.hidden_branches();
    let branches: Vec<TaskView> = app.listed_branches().into_iter().cloned().collect();
    let running = branches
        .iter()
        .filter(|t| matches!(t.state, TaskState::Working | TaskState::Researching))
        .count();
    let queued = branches
        .iter()
        .filter(|t| t.state == TaskState::Queued)
        .count();
    let waiting = branches.iter().filter(|t| app.task_needs_you(t.id)).count();
    let done = all.len() - app.open_tasks();
    let mut summary = vec![
        Span::styled(format!(" {running} running"), bold(CYAN)),
        Span::styled(
            format!(
                " · {queued} queued · {done} ended{}",
                match (done, hidden) {
                    (0, _) => "",
                    (_, 0) => " (e hides them)",
                    _ => ", hidden (e shows them)",
                }
            ),
            fg(DIM),
        ),
    ];
    if waiting > 0 {
        summary.push(Span::styled(
            format!(" · {waiting} waiting for you"),
            bold(AMBER),
        ));
    }
    let hint = "⏎ open · 0 main · ]/[ next tab ";
    let used: usize = summary.iter().map(|s| s.width()).sum();
    if w > used + width(hint) {
        summary.push(Span::styled(
            format!("{:>pad$}", hint, pad = w - used),
            fg(DIM),
        ));
    }
    f.render_widget(Paragraph::new(Line::from(summary)), row(inner, 0));

    app.board_sel = app.board_sel.min(branches.len().saturating_sub(1));
    // Each task takes two rows; keep the selected one in view.
    let fit = (inner.height.saturating_sub(2) / 2).max(1) as usize;
    let first = app.board_sel.saturating_sub(fit - 1);
    let order: Vec<usize> = (0..branches.len()).rev().collect();
    for (shown, &at) in order.iter().enumerate().skip(first).take(fit) {
        let task = &branches[at];
        let (glyph, label, color) = task_status(app, task);
        let glyph = if task.state.is_open() && glyph == "●" {
            spinner(app)
        } else {
            glyph
        };
        let selected = shown == app.board_sel;
        let y = 2 + ((shown - first) * 2) as u16;
        let time = task_time(task);
        let head_w = 6 + 2 + 10 + 9;
        let title_w = w.saturating_sub(head_w + 1);
        let top = Line::from(vec![
            Span::styled("▍", fg(color)),
            Span::styled(format!("#{:<4}", task.id), bold(TEXT)),
            Span::styled(format!("{glyph} "), fg(color)),
            Span::styled(pad(label, 10), bold(color)),
            Span::styled(format!("{time:>8} "), fg(DIM)),
            Span::styled(
                trunc(&task.title, title_w),
                if task.unread { bold(NEON) } else { bold(TEXT) },
            ),
        ]);
        let counts = task_counts(task);
        let note_w = w.saturating_sub(8 + width(&counts) + 2);
        let bottom = Line::from(vec![
            Span::styled("▏", fg(dark(color))),
            Span::styled("     └ ", fg(GRID_HOT)),
            Span::styled(pad(&trunc(&task.note, note_w), note_w), fg(DIM)),
            Span::styled(format!(" {counts}"), fg(GRID_HOT)),
        ]);
        let style = if selected {
            Style::new().bg(SELECT)
        } else {
            Style::new()
        };
        let (r0, r1) = (row(inner, y), row(inner, y + 1));
        if y + 1 < inner.height {
            f.render_widget(Paragraph::new(top.style(style)), r0);
            f.render_widget(Paragraph::new(bottom.style(style)), r1);
            let both = Rect { height: 2, ..r0 };
            app.hits.board.push((both, task.id));
        }
    }
    app.hits.texture.push(inner);
}

/// The top of a task's tab: where it stands, what it has used, and what it
/// is doing now. Says plainly when it is paused for research or for you.
fn render_task_head(f: &mut Frame, app: &App, task: &TaskView, area: Rect) {
    let w = area.width.saturating_sub(1) as usize;
    let (glyph, label, color) = task_status(app, task);
    let glyph = if glyph == "●" { spinner(app) } else { glyph };
    let rest = if task.id == MAIN {
        let open = app.open_tasks();
        format!(
            "  the conversation that keeps going · {} · {}",
            match open {
                0 => "no branch open".to_string(),
                1 => "1 branch open".to_string(),
                n => format!("{n} branches open"),
            },
            task_counts(task)
        )
    } else {
        format!(
            "  {} · opened by {} · merges into main when it ends · {}",
            task_time(task),
            if task.origin.is_empty() {
                "main"
            } else {
                &task.origin
            },
            task_counts(task)
        )
    };
    let mut top = vec![Span::styled(
        if task.id == MAIN {
            " MAIN ".to_string()
        } else {
            format!(" #{} ", task.id)
        },
        bold(TEXT),
    )];
    top.extend(chip(
        &format!(" {glyph} {label} "),
        color,
        task.state.is_open(),
    ));
    top.push(Span::styled(trunc(&rest, w.saturating_sub(24)), fg(DIM)));
    f.render_widget(Paragraph::new(Line::from(top)), row(area, 0));

    let (lead, lead_color, text) = if app.task_needs_you(task.id) {
        (
            " ! WAITING FOR YOU ▸ ",
            AMBER,
            "a call needs your approval: a accepts, d denies, ⏎ shows it".to_string(),
        )
    } else if task.state == TaskState::Researching {
        (
            " ◈ PAUSED FOR KNOWLEDGE RESEARCH ▸ ",
            VIOLET,
            task.note.clone(),
        )
    } else if task.state.is_open() {
        (" now ▸ ", CYAN, task.note.clone())
    } else {
        (" result ▸ ", DIM, task.note.clone())
    };
    let avail = w.saturating_sub(width(lead));
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(lead, bold(lead_color)),
            Span::styled(trunc(&text, avail), fg(TEXT)),
        ])),
        row(area, 1),
    );
    f.render_widget(
        Paragraph::new(Line::styled("─".repeat(area.width as usize), fg(GRID))),
        row(area, 2),
    );
}

fn render_stream(f: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.focus == Panel::Center;
    let accent = CYAN;
    let rule = if focused {
        lerp(accent, GRID, 0.45)
    } else {
        GRID
    };
    let block = Block::bordered()
        .border_type(BorderType::Plain)
        .border_style(fg(rule))
        .style(Style::new().bg(PANEL));
    let inner = block.inner(area);
    f.render_widget(block, area);
    corners(
        f.buffer_mut(),
        area,
        if focused {
            accent
        } else {
            lerp(accent, GRID, 0.4)
        },
    );
    app.hits.panels.push((area, Panel::Center));

    // One tab per task over the top border, newest first, after the
    // overview. Drawn as chips so their positions are known exactly.
    let mut x = area.x + 2;
    // Room is kept on the right for the live marker (not drawn over the
    // board) and for the count of tabs that did not fit.
    let limit = area.x
        + area
            .width
            .saturating_sub(if app.on_board() { 6 } else { 18 });
    let mut tabs: Vec<(Option<u32>, String, Color)> = Vec::new();
    if let Some(main) = app.task(MAIN) {
        let (glyph, _, color) = task_status(app, main);
        tabs.push((Some(MAIN), format!(" 0 MAIN {glyph} "), color));
    }
    tabs.push((
        None,
        match (app.branches().count(), app.open_tasks()) {
            (0, _) if app.task(MAIN).is_none() => " ALL ".to_string(),
            (_, open) => format!(" b BRANCHES {open} "),
        },
        accent,
    ));
    for task in app.listed_branches().into_iter().rev() {
        let (glyph, _, color) = task_status(app, task);
        tabs.push((
            Some(task.id),
            format!(" #{} {glyph} {} ", task.id, trunc(&task.title, 12)),
            color,
        ));
    }
    // The open tab is always drawn, even when older than what fits.
    let mut hidden = 0;
    for (id, label, color) in &tabs {
        let w = chip_width(label);
        let active = *id == app.task_sel;
        if x + w >= limit && !active {
            hidden += 1;
            continue;
        }
        if x + w >= limit {
            break;
        }
        let unread = id.and_then(|id| app.task(id)).is_some_and(|t| t.unread);
        let spans = if active {
            chip(label, *color, true)
        } else if unread {
            chip(label, *color, false)
        } else {
            vec![Span::styled(
                format!(" {label} "),
                fg(lerp(*color, DIM, 0.55)),
            )]
        };
        let r = Rect {
            x,
            y: area.y,
            width: w,
            height: 1,
        };
        f.render_widget(Paragraph::new(Line::from(spans)), r);
        app.hits.task_tabs.push((r, *id));
        x += w + 1;
    }
    if hidden > 0 {
        f.render_widget(
            Paragraph::new(Line::styled(format!("+{hidden}"), fg(DIM))),
            Rect {
                x,
                y: area.y,
                width: 5,
                height: 1,
            },
        );
    }

    // The three ways of looking at a tab, on the bottom border.
    let mut vx = area.x + 2;
    let vy = area.y + area.height.saturating_sub(1);
    for (i, tab) in Tab::ALL.iter().enumerate() {
        let label = format!(" {} {} ", i + 1, tab.label());
        let w = chip_width(&label);
        if vx + w >= area.x + area.width.saturating_sub(2) {
            break;
        }
        let spans = if *tab == app.tab {
            chip(&label, accent, false)
        } else {
            vec![Span::styled(format!(" {label} "), fg(DIM))]
        };
        let r = Rect {
            x: vx,
            y: vy,
            width: w,
            height: 1,
        };
        f.render_widget(Paragraph::new(Line::from(spans)), r);
        app.hits.tabs.push((r, *tab));
        vx += w + 1;
    }

    // The overview's first view is the board of tasks.
    if app.on_board() {
        render_board(f, app, inner);
        return;
    }
    // A task's tab opens with where the task stands.
    let inner = match app.task_sel.and_then(|id| app.task(id)).cloned() {
        Some(task) if inner.height > 8 => {
            let [head, rest] =
                Layout::vertical([Constraint::Length(3), Constraint::Min(1)]).areas(inner);
            render_task_head(f, app, &task, head);
            rest
        }
        _ => inner,
    };

    // Build rows for the active tab.
    let w = inner.width.saturating_sub(1) as usize;
    let mut rows: Vec<(Line<'static>, usize)> = Vec::new();
    let mut visible = Vec::new();
    let in_branch = app.task_sel.is_some_and(|task| task != MAIN);
    for idx in 0..app.events.len() {
        if let Some(filter) = app.agent_filter
            && app.events[idx].kind.agent() != Some(filter)
        {
            continue;
        }
        // A tab shows its own task's events; main's also shows what
        // belongs to no task (the harness itself speaking).
        let own = app.events[idx].task;
        if app.task_sel.is_some()
            && own != app.task_sel
            && !(app.task_sel == Some(MAIN) && own.is_none())
        {
            continue;
        }
        let lines = match app.tab {
            // Activity is the conversation and nothing else; every step
            // behind it is in the raw events view.
            Tab::Timeline if !in_conversation(&app.events[idx].kind, in_branch) => Vec::new(),
            Tab::Timeline => timeline_entry(app, idx, w),
            Tab::Events => raw_entry(app, idx, w),
            Tab::Memory => memory_entry(app, idx, w),
        };
        if !lines.is_empty() {
            visible.push(idx);
            rows.extend(lines.into_iter().map(|l| (l, idx)));
        }
    }
    app.visible_center = visible;
    // The reply being written right now, growing as it comes. It belongs
    // to no event yet, so it cannot be selected.
    if app.tab == Tab::Timeline
        && let Some(draft) = app.draft()
    {
        let head = vec![
            Span::styled("▍", fg(CYAN)),
            Span::styled(format!("{:<8}", "writing"), fg(DIM)),
            Span::styled(format!("{} ", spinner(app)), fg(CYAN)),
            Span::styled(format!("{} ", pad("CORE", 8)), bold(CYAN)),
        ];
        let mut body = markdown::render(draft, w.saturating_sub(PREFIX_W));
        match body.last_mut() {
            Some(last) => last.push(Span::styled("▌", fg(CYAN))),
            None => body.push(vec![Span::styled("▌", fg(CYAN))]),
        }
        for (i, row) in body.into_iter().enumerate() {
            let mut spans = if i == 0 {
                head.clone()
            } else {
                continuation(CYAN)
            };
            spans.extend(row);
            rows.push((Line::from(spans), usize::MAX));
        }
    }

    let h = inner.height as usize;
    let total = rows.len();
    let max_scroll = total.saturating_sub(h);
    if app.ensure_sel {
        app.ensure_sel = false;
        if let Some(sel) = app.center_sel
            && let (Some(r0), Some(r1)) = (
                rows.iter().position(|(_, i)| *i == sel),
                rows.iter().rposition(|(_, i)| *i == sel),
            )
        {
            let start = total.saturating_sub(h + app.center_scroll);
            if r0 < start {
                app.center_scroll = total.saturating_sub(h + r0);
            } else if r1 >= start + h {
                app.center_scroll = total.saturating_sub(r1 + 1);
            }
        }
    }
    app.center_scroll = app.center_scroll.min(max_scroll);
    let start = total.saturating_sub(h + app.center_scroll);

    // Live / scrolled indicator on the top-right border.
    let live: Vec<Span> = if app.center_scroll == 0 {
        chip(
            if blink(app) {
                " ● LIVE "
            } else {
                " ○ LIVE "
            },
            NEON,
            false,
        )
    } else {
        chip(&format!(" ‖ +{} (G) ", app.center_scroll), AMBER, false)
    };
    let lw: u16 = live.iter().map(|s| s.width() as u16).sum();
    if area.width > lw + 6 {
        f.render_widget(
            Paragraph::new(Line::from(live)),
            Rect {
                x: area.x + area.width - lw - 3,
                y: area.y,
                width: lw,
                height: 1,
            },
        );
    }

    if total == 0 {
        let msg = match app.tab {
            Tab::Memory => "// nothing looked up, researched or learned here yet",
            _ if app.task_sel.is_some() => "// nothing has happened in this task yet",
            _ => "// nothing yet: press i and say what you want done",
        };
        f.render_widget(
            Paragraph::new(Line::styled(msg, fg(DIM))).alignment(Alignment::Center),
            row(inner, inner.height / 2),
        );
        app.hits.texture.push(inner);
        return;
    }

    for (i, (line, idx)) in rows.iter().skip(start).take(h).enumerate() {
        let r = row(inner, i as u16);
        let line = if app.center_sel == Some(*idx) {
            line.clone().style(Style::new().bg(SELECT))
        } else {
            line.clone()
        };
        f.render_widget(Paragraph::new(line), r);
        if *idx != usize::MAX {
            app.hits.center.push((r, *idx));
        }
    }
    app.hits.texture.push(inner);

    if total > h {
        let mut sb = ScrollbarState::new(max_scroll).position(start);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("┊"))
                .track_style(fg(GRID))
                .thumb_symbol("┃")
                .thumb_style(fg(accent)),
            area.inner(ratatui::layout::Margin {
                vertical: 2,
                horizontal: 0,
            }),
            &mut sb,
        );
    }
}

// ───────────────────────────── mind: memory field, heartbeat, live thought ─────────────────────────────

const MIND_H: u16 = 11;

fn render_center(f: &mut Frame, app: &mut App, area: Rect) {
    if area.height >= MIND_H + 10 {
        let [mind, stream] =
            Layout::vertical([Constraint::Length(MIND_H), Constraint::Min(8)]).areas(area);
        render_mind(f, app, mind);
        render_stream(f, app, stream);
    } else {
        render_stream(f, app, area);
    }
}

/// The mark a memory has on the map: one shape per kind.
fn mem_glyph(k: MemKind) -> &'static str {
    match k {
        MemKind::Episodic => "●",
        MemKind::Semantic => "◆",
        MemKind::Procedural => "■",
    }
}

fn mem_color(k: MemKind) -> Color {
    match k {
        MemKind::Episodic => VIOLET,
        MemKind::Semantic => CYAN,
        MemKind::Procedural => NEON,
    }
}

fn age(i: Instant) -> f64 {
    i.elapsed().as_secs_f64()
}

/// Where a memory sits in the field. Long-term memories orbit at a distance;
/// recalled ones are pulled toward the core (working memory) and drift back.
fn node_pos(m: &MemState, tick: u64) -> (f64, f64) {
    let id = m.node.id as f64;
    // Each memory has a place of its own that does not drift, so it can be
    // found again and clicked.
    let _ = tick;
    let angle = id * 2.399_963;
    let base = 0.45 + 0.5 * (id * 0.618_034).fract();
    let pull = m
        .touched
        .map(|t| (1.0 - age(t) / 8.0).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let r = base * (1.0 - 0.6 * pull);
    (r * angle.cos() * 2.0, r * angle.sin())
}

/// Heart period in ticks: faster when more events are flowing.
fn heart_period(app: &App) -> f64 {
    (36.0 - app.events_per_min().min(60) as f64 * 0.42).max(11.0)
}

/// One ECG cycle sampled at phase 0..1.
fn ecg(ph: f64) -> f64 {
    match ph {
        p if p < 0.06 => 0.15 * (p / 0.06 * PI).sin(),
        p if p < 0.09 => 0.0,
        p if p < 0.11 => -0.25,
        p if p < 0.14 => 1.5,
        p if p < 0.17 => -0.45,
        p if p < 0.30 => 0.0,
        p if p < 0.42 => 0.3 * ((p - 0.30) / 0.12 * PI).sin(),
        _ => 0.0,
    }
}

fn render_mind(f: &mut Frame, app: &mut App, area: Rect) {
    let wm = app
        .memory
        .iter()
        .filter(|m| m.absorbed.is_none() && m.touched.is_some_and(|t| age(t) < 8.0))
        .count();
    let ltm = app.memory.iter().filter(|m| m.absorbed.is_none()).count();
    let learned = app.memory.iter().filter(|m| m.born >= app.started).count();
    let bottom = Line::from(vec![
        Span::styled(" long-term ", fg(DIM)),
        Span::styled(ltm.to_string(), bold(CYAN)),
        Span::styled(" ╱ working ", fg(DIM)),
        Span::styled(wm.to_string(), bold(VIOLET)),
        Span::styled(" ╱ learned ", fg(DIM)),
        Span::styled(format!("{learned} "), bold(NEON)),
    ]);
    let inner = hud(f, area, "MIND", MAGENTA, false, Some(bottom));
    app.hits.panels.push((area, Panel::Center));

    let field_w = (inner.width / 2).clamp(24, 44);
    let [field, _, side] = Layout::horizontal([
        Constraint::Length(field_w),
        Constraint::Length(1),
        Constraint::Min(20),
    ])
    .areas(inner);

    let core_state = app
        .agents
        .iter()
        .find(|a| a.parent.is_none())
        .map(|a| a.state)
        .unwrap_or(AgentState::Idle);
    let (alarm_state, alarm) = app.goal_state();
    let pulse_color = if alarm && alarm_state == "BLOCKED" {
        RED
    } else if alarm {
        AMBER
    } else if core_state == AgentState::Reflecting {
        VIOLET
    } else {
        NEON
    };
    let period = heart_period(app);
    let offset = app.tick as f64 * 0.6;
    let edge_phase = ((side.width as f64 + offset) % period) / period;
    let beat = if (0.09..0.25).contains(&edge_phase) {
        1.0
    } else {
        0.0
    };

    // ── memory field ──
    let tick = app.tick;
    let pos_of = |id: u32| {
        app.memory
            .iter()
            .find(|m| m.node.id == id)
            .map(|m| node_pos(m, tick))
    };
    /// (from, to, age in seconds)
    type Link = ((f64, f64), (f64, f64), f64);
    /// (x, y, colour, touched age, born age, label, absorbed)
    type Node = (f64, f64, Color, Option<f64>, Option<f64>, String, bool);
    let links: Vec<Link> = app
        .links
        .iter()
        .filter(|(_, _, t)| age(*t) < 10.0)
        .filter_map(|(a, b, t)| Some((pos_of(*a)?, pos_of(*b)?, age(*t))))
        .collect();
    let nodes: Vec<Node> = app
        .memory
        .iter()
        .filter(|m| m.absorbed.is_none_or(|t| age(t) < 3.0))
        .map(|m| {
            let (x, y) = node_pos(m, tick);
            let touched = m.touched.map(age).filter(|a| *a < 6.0);
            let born = Some(age(m.born)).filter(|a| *a < 3.0);
            (
                x,
                y,
                mem_color(m.node.kind),
                touched,
                born,
                format!("#{} {}", m.node.id, trunc(&m.node.label, 20)),
                m.absorbed.is_some(),
            )
        })
        .collect();
    let core_r = 0.06 + 0.07 * beat;
    f.render_widget(
        Canvas::default()
            .background_color(PANEL)
            .marker(Marker::Braille)
            .x_bounds([-2.2, 2.2])
            .y_bounds([-1.12, 1.12])
            .paint(|ctx| {
                for r in [0.42, 0.97] {
                    ctx.draw(&Circle {
                        x: 0.0,
                        y: 0.0,
                        radius: r,
                        color: TEXTURE,
                    });
                }
                for ((ax, ay), (bx, by), a) in &links {
                    ctx.draw(&CanvasLine::new(
                        *ax,
                        *ay,
                        *bx,
                        *by,
                        lerp(CYAN, PANEL, (*a / 10.0) as f32),
                    ));
                }
                for (x, y, c, touched, _, _, _) in &nodes {
                    if let Some(a) = touched {
                        ctx.draw(&CanvasLine::new(
                            *x,
                            *y,
                            0.0,
                            0.0,
                            lerp(*c, PANEL, (*a / 6.0) as f32),
                        ));
                    }
                }
                ctx.layer();
                for (x, y, c, touched, born, _, absorbed) in &nodes {
                    let color = if *absorbed {
                        lerp(*c, PANEL, 0.6)
                    } else if touched.is_some() {
                        *c
                    } else {
                        lerp(*c, PANEL, 0.35)
                    };
                    let _ = color;
                    if let Some(a) = born {
                        ctx.draw(&Circle {
                            x: *x,
                            y: *y,
                            radius: 0.05 + a * 0.16,
                            color: lerp(NEON, PANEL, (*a / 3.0) as f32),
                        });
                    }
                }
                ctx.draw(&Circle {
                    x: 0.0,
                    y: 0.0,
                    radius: core_r,
                    color: pulse_color,
                });
                ctx.draw(&Points {
                    coords: &[(0.0, 0.0)],
                    color: pulse_color,
                });
                ctx.layer();
                for (x, y, c, touched, born, label, absorbed) in &nodes {
                    if !*absorbed && (touched.is_some() || born.is_some()) {
                        ctx.print(*x + 0.1, *y, Line::styled(label.clone(), bold(*c)));
                    }
                }
            }),
        field,
    );
    // Each memory is one mark, shaped by its kind (not colour alone), that
    // can be clicked to open what is remembered.
    let open = match app.modal {
        Some(Modal::Memory(id)) => Some(id),
        _ => None,
    };
    let marks: Vec<(u32, MemKind, (f64, f64), bool)> = app
        .memory
        .iter()
        .filter(|m| m.absorbed.is_none())
        .map(|m| {
            let lit = m.touched.is_some_and(|t| age(t) < 6.0) || age(m.born) < 3.0;
            (m.node.id, m.node.kind, node_pos(m, tick), lit)
        })
        .collect();
    for (id, kind, (x, y), lit) in marks {
        if field.width < 3 || field.height == 0 {
            break;
        }
        let cx = field.x + (((x + 2.2) / 4.4) * (field.width - 1) as f64).round() as u16;
        let cy = field.y + (((1.12 - y) / 2.24) * (field.height - 1) as f64).round() as u16;
        let (cx, cy) = (
            cx.min(field.x + field.width - 1),
            cy.min(field.y + field.height - 1),
        );
        let color = mem_color(kind);
        let style = if open == Some(id) {
            Style::new().fg(BG).bg(color).add_modifier(Modifier::BOLD)
        } else if lit {
            bold(color)
        } else {
            fg(lerp(color, PANEL, 0.25))
        };
        if let Some(cell) = f.buffer_mut().cell_mut((cx, cy)) {
            cell.set_symbol(mem_glyph(kind)).set_style(style);
        }
        // A little wider than the mark, so it is easy to hit.
        app.hits.memories.push((
            Rect {
                x: cx.saturating_sub(1).max(field.x),
                y: cy,
                width: 3,
                height: 1,
            },
            id,
        ));
    }

    // ── vitals ──
    let sw = side.width as usize;
    let epm = app.events_per_min();
    let mut v = vec![
        Span::styled(
            "♥ ",
            if beat > 0.0 {
                bold(pulse_color)
            } else {
                fg(dark(pulse_color))
            },
        ),
        Span::styled(format!("{epm:>2} ev/min"), fg(TEXT)),
    ];
    // What the marks on the map are, and how to open one.
    for legend in [
        "   ● event  ◆ fact  ■ always kept in mind  · click one, or m",
        "  ● event ◆ fact ■ kept",
    ] {
        if sw >= 11 + width(legend) {
            v.push(Span::styled(legend, fg(DIM)));
            break;
        }
    }
    f.render_widget(Paragraph::new(Line::from(v)), row(side, 0));

    // ── heartbeat trace ──
    let ecg_area = Rect {
        y: side.y + 1,
        height: 2.min(side.height.saturating_sub(1)),
        ..side
    };
    let wcells = ecg_area.width as f64;
    let trace: Vec<(f64, f64)> = (0..=(ecg_area.width as usize * 4))
        .map(|i| {
            let x = i as f64 / 4.0;
            (x, ecg(((x + offset) % period) / period))
        })
        .collect();
    f.render_widget(
        Canvas::default()
            .background_color(PANEL)
            .marker(Marker::Braille)
            .x_bounds([0.0, wcells])
            .y_bounds([-0.6, 1.6])
            .paint(|ctx| {
                for pair in trace.windows(2) {
                    let fade = (pair[0].0 / wcells) as f32;
                    ctx.draw(&CanvasLine::new(
                        pair[0].0,
                        pair[0].1,
                        pair[1].0,
                        pair[1].1,
                        lerp(dark(pulse_color), pulse_color, fade),
                    ));
                }
            }),
        ecg_area,
    );

    // ── stream of thought ──
    if let Some((agent, text, arrived)) = &app.thought {
        let elapsed = age(*arrived);
        let shown = ((elapsed * 60.0) as usize).min(width(text));
        let typing = shown < width(text);
        let stale = elapsed > 20.0;
        let who = app.agent_name(*agent).to_uppercase();
        let head_style = if stale { fg(DIM) } else { bold(MAGENTA) };
        let status = if typing {
            format!("{} thinking", spinner(app))
        } else if stale {
            format!("{:.0}s ago", elapsed)
        } else {
            "now".into()
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("∴ ", head_style),
                Span::styled(format!("{who} "), head_style),
                Span::styled(status, fg(DIM)),
            ])),
            row(side, 3),
        );
        let visible: String = text.chars().take(shown).collect();
        let body_rows = side.height.saturating_sub(4) as usize;
        let mut lines = wrap(&visible, sw.saturating_sub(2));
        if lines.len() > body_rows {
            lines = lines.split_off(lines.len() - body_rows);
        }
        let n = lines.len();
        for (i, l) in lines.into_iter().enumerate() {
            let mut spans = vec![
                Span::styled("┊ ", fg(dark(MAGENTA))),
                Span::styled(
                    l,
                    if stale {
                        Style::new().fg(DIM).add_modifier(Modifier::ITALIC)
                    } else {
                        Style::new().fg(TEXT).add_modifier(Modifier::ITALIC)
                    },
                ),
            ];
            if i + 1 == n && (typing || blink(app)) && !stale {
                spans.push(Span::styled("▌", fg(MAGENTA)));
            }
            f.render_widget(Paragraph::new(Line::from(spans)), row(side, 4 + i as u16));
        }
    } else {
        f.render_widget(
            Paragraph::new(Line::styled("∴ // no thoughts yet", fg(DIM))),
            row(side, 3),
        );
    }
}

// ───────────────────────────── right: context, oscilloscope, tools ─────────────────────────────

fn render_right(f: &mut Frame, app: &mut App, area: Rect) {
    let ctx_h = (app.context.len() as u16 + 5).max(7);
    let [ctx, thr, tools] = Layout::vertical([
        Constraint::Length(ctx_h),
        Constraint::Length(7),
        Constraint::Min(5),
    ])
    .areas(area);

    // Context window.
    let inner = hud(f, ctx, "CONTEXT WINDOW", CYAN, false, None);
    let used = app.context_used();
    let pct = used as f64 / app.window.max(1) as f64 * 100.0;
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(tok(used as u64), bold(CYAN)),
            Span::styled(format!(" / {} tok", tok(app.window as u64)), fg(DIM)),
            Span::styled(format!("   {pct:.0}% loaded"), bold(TEXT)),
        ])),
        row(inner, 0),
    );
    // Stacked bar: each section's share of the window.
    let bw = inner.width as usize;
    let mut spans = Vec::new();
    let mut drawn = 0usize;
    for (i, s) in app.context.iter().enumerate() {
        let n = ((s.tokens as f64 / app.window.max(1) as f64) * bw as f64).round() as usize;
        let n = n.min(bw - drawn);
        spans.push(Span::styled("▰".repeat(n), fg(ACCENTS[i % ACCENTS.len()])));
        drawn += n;
    }
    spans.push(Span::styled("▱".repeat(bw - drawn), fg(GRID)));
    f.render_widget(Paragraph::new(Line::from(spans)), row(inner, 1));
    let max = app
        .context
        .iter()
        .map(|s| s.tokens)
        .max()
        .unwrap_or(1)
        .max(1);
    for (i, s) in app.context.iter().enumerate() {
        let y = 3 + i as u16;
        if y >= inner.height {
            break;
        }
        let bar_w = bw.saturating_sub(18);
        let n = (s.tokens as f64 / max as f64 * bar_w as f64).round() as usize;
        let color = ACCENTS[i % ACCENTS.len()];
        let r = row(inner, y);
        let mut spans = vec![
            Span::styled("◆ ", fg(color)),
            Span::styled(pad(&s.name.to_uppercase(), 8), fg(TEXT)),
            Span::styled(format!("{:>6} ", tok(s.tokens as u64)), fg(DIM)),
        ];
        for k in 0..bar_w {
            if k < n {
                let c = lerp(dark(color), color, (k + 1) as f32 / n.max(1) as f32);
                spans.push(Span::styled("■", fg(c)));
            } else {
                spans.push(Span::styled("·", fg(GRID)));
            }
        }
        f.render_widget(Paragraph::new(Line::from(spans)), r);
        app.hits.context.push((r, i));
    }
    if app.context.is_empty() {
        f.render_widget(
            Paragraph::new(Line::styled("// not assembled yet", fg(DIM))),
            row(inner, 3),
        );
    }

    // Throughput: a smooth area chart that scrolls between samples.
    let tok_s = app.throughput.last().copied().unwrap_or(0);
    let inner = hud(
        f,
        thr,
        &format!("THROUGHPUT {tok_s} tok/s"),
        NEON,
        false,
        None,
    );
    // Ease each sample toward the one before it, so a burst rises and falls
    // rather than spiking.
    let mut eased = 0.0;
    let history: Vec<f64> = app
        .throughput
        .iter()
        .map(|v| {
            eased += (*v as f64 - eased) * 0.5;
            eased
        })
        .collect();
    let cells = inner.width.max(1) as f64;
    let newest = history.len().saturating_sub(1) as f64;
    // How far the next sample has slid in, 0..1.
    let slide = (age(app.sampled_at) / App::SAMPLE_SECS).min(1.0);
    let first_shown = (newest - cells).max(0.0) as usize;
    let peak = history[first_shown..].iter().copied().fold(6.0, f64::max) * 1.2;
    area_chart(f.buffer_mut(), inner, NEON, |across| {
        let x = newest - 1.0 + slide - (1.0 - across) * (cells - 1.0);
        smooth(&history, x) / peak
    });

    // Tool calls.
    let focused = app.focus == Panel::Tools;
    let pending = app.pending().len();
    let title = if pending > 0 {
        format!("TOOLS {} ◷{pending}", app.tools.len())
    } else {
        format!("TOOLS {}", app.tools.len())
    };
    let inner = hud(f, tools, &title, AMBER, focused, None);
    app.hits.panels.push((tools, Panel::Tools));
    let h = inner.height as usize;
    let len = app.tools.len();
    let start = if focused {
        app.tool_sel
            .saturating_sub(h.saturating_sub(1))
            .min(len.saturating_sub(h))
    } else {
        len.saturating_sub(h)
    };
    for (i, tc) in app.tools.iter().enumerate().skip(start).take(h) {
        let (b, bs) = tool_badge(app, tc);
        let agent = app.agent_name(tc.agent);
        let name_w = (inner.width as usize).saturating_sub(5 + 9 + width(&b));
        let line = Line::from(vec![
            Span::styled("▍", fg(risk_color(tc.risk))),
            Span::styled(format!("{:<3} ", tc.id), fg(DIM)),
            Span::styled(pad(&tc.tool, name_w), bold(TEXT)),
            Span::styled(pad(&agent, 9), fg(DIM)),
            Span::styled(b, bs),
        ]);
        let line = if focused && i == app.tool_sel {
            line.style(Style::new().bg(SELECT))
        } else {
            line
        };
        let r = row(inner, (i - start) as u16);
        f.render_widget(Paragraph::new(line), r);
        app.hits.tools.push((r, i));
    }
    if app.tools.is_empty() {
        f.render_widget(Paragraph::new(Line::styled("// none yet", fg(DIM))), inner);
    }
    app.hits.texture.push(inner);
}

// ───────────────────────────── signals + input ─────────────────────────────

fn render_signals(f: &mut Frame, app: &mut App, area: Rect) {
    let cache = if app.tokens_in > 0 {
        app.tokens_cached as f64 / app.tokens_in as f64 * 100.0
    } else {
        0.0
    };
    let metrics = Line::from(vec![
        Span::styled(format!(" calls {} ", app.model_calls), fg(DIM)),
        Span::styled("╱", fg(GRID)),
        Span::styled(format!(" cache {cache:.0}% "), fg(DIM)),
        Span::styled("╱", fg(GRID)),
        Span::styled(format!(" events {} ", app.events.len()), fg(DIM)),
    ]);
    let alarm = !app.pending().is_empty();
    let inner = hud(f, area, "SIGNALS", AMBER, alarm, Some(metrics));

    let mut y = 0;
    let first_pending = app
        .pending()
        .first()
        .map(|t| (t.id, t.tool.clone(), t.args.clone()));
    if let Some((id, tool, args)) = first_pending {
        let r = row(inner, 0);
        let head = chip(" ▲ APPROVAL PENDING ", AMBER, blink(app));
        let head_w: u16 = head.iter().map(|s| s.width() as u16).sum();
        let buttons = [
            (" A ▸ ACCEPT ", NEON, Button::Accept),
            (" D ▸ DENY ", RED, Button::Deny),
            (" ⏎ INSPECT ", CYAN, Button::OpenPending),
        ];
        let buttons_w: u16 = buttons.iter().map(|(l, _, _)| chip_width(l) + 1).sum();
        let what = trunc(
            &format!(
                " #{id} {tool} {}",
                args.lines()
                    .next()
                    .unwrap_or("")
                    .trim_start_matches("path: ")
            ),
            (inner.width as usize).saturating_sub((head_w + buttons_w + 3) as usize),
        );
        let wlen = width(&what) as u16;
        let mut spans = head;
        spans.push(Span::styled(what, bold(TEXT)));
        spans.push(Span::raw("  "));
        for (label, c, _) in &buttons {
            spans.extend(chip(label, *c, true));
            spans.push(Span::raw(" "));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), r);
        // Button hit areas, in drawing order.
        let mut x = r.x + head_w + wlen + 2;
        for (label, _, b) in buttons {
            let w = chip_width(label);
            app.hits.buttons.push((
                Rect {
                    x,
                    y: r.y,
                    width: w,
                    height: 1,
                },
                b,
            ));
            x += w + 1;
        }
        app.hits.buttons.push((
            Rect {
                width: head_w + wlen,
                ..r
            },
            Button::OpenPending,
        ));
        y = 1;
    }
    for (at, sev, text) in app
        .signals
        .iter()
        .rev()
        .take((inner.height.saturating_sub(y)) as usize)
    {
        let (g, c) = severity_glyph(*sev);
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("▍", fg(c)),
                Span::styled(format!("{g} "), bold(c)),
                Span::styled(format!("{} ", t(*at)), fg(DIM)),
                Span::styled(
                    trunc(text, (inner.width as usize).saturating_sub(12)),
                    fg(c),
                ),
            ])),
            row(inner, y),
        );
        y += 1;
    }
}

fn render_input(f: &mut Frame, app: &mut App, area: Rect) {
    let focused = app.focus == Panel::Input;
    app.hits.panels.push((area, Panel::Input));
    let bg = if focused { SELECT } else { BAND };
    // Says where the message will go before it is sent.
    let target = match app.input_task.filter(|_| focused) {
        Some(id) => format!(" REPLY IN BRANCH #{id} "),
        None => " TO MAIN ".to_string(),
    };
    let mut spans = chip(
        &target,
        if target.starts_with(" REPLY") {
            CYAN
        } else {
            NEON
        },
        focused,
    );
    spans.push(Span::styled(" ❯ ", bold(if focused { NEON } else { DIM })));
    if app.input.is_empty() && !focused {
        spans.push(Span::styled(
            "i to type · it goes to main, which branches when it is real work",
            fg(DIM),
        ));
    } else {
        spans.push(Span::styled(app.input.clone(), fg(TEXT)));
        if focused {
            spans.push(Span::styled(if blink(app) { "█" } else { " " }, fg(NEON)));
        }
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::new().bg(bg)),
        area,
    );
    let hint = if focused {
        "⏎ send  ╱  esc leave "
    } else {
        "0 main ╱ b branches ╱ [ ] tabs ╱ 1-3 views ╱ a/d approve ╱ X stop ╱ m memory ╱ ? help ╱ q quit "
    };
    if area.width as usize > width(hint) + 44 {
        f.render_widget(
            Paragraph::new(Line::styled(hint, fg(DIM))).alignment(Alignment::Right),
            area,
        );
    }
}

// ───────────────────────────── modals ─────────────────────────────

fn kv(k: &str, v: String, vs: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!(" {k:<9}"), fg(DIM)),
        Span::styled(v, vs),
    ])
}

fn section(title: &str, w: u16) -> Line<'static> {
    // `w` is the modal's outer width: leave room for both borders.
    let fill = (w as usize).saturating_sub(width(title) + 10);
    Line::from(vec![
        Span::styled(" ▍", fg(MAGENTA)),
        Span::styled(format!("{title} "), bold(MAGENTA)),
        Span::styled("─".repeat(fill), fg(GRID)),
    ])
}

fn code_lines(text: &str) -> Vec<Line<'static>> {
    text.lines()
        .enumerate()
        .map(|(i, l)| {
            let style = if l.starts_with('+') {
                fg(NEON)
            } else if l.starts_with('-') {
                fg(RED)
            } else if l.starts_with("@@") {
                fg(VIOLET)
            } else if l.contains("FAILED") {
                bold(RED)
            } else if l.contains("ok.") || l.contains("0 warnings") {
                bold(NEON)
            } else {
                fg(TEXT)
            };
            Line::from(vec![
                Span::styled(format!("  {:>3} │ ", i + 1), fg(GRID)),
                Span::styled(l.to_string(), style),
            ])
        })
        .collect()
}

fn node_color(kind: &str) -> Color {
    match kind {
        "passage" => CYAN,
        "note" => NEON,
        "term" => AMBER,
        "document" => VIOLET,
        _ => DIM,
    }
}

/// A node of the lookup graph, placed for drawing.
struct GraphNode {
    kind: String,
    title: String,
    /// How many links from the question: 1 is a node whose words matched.
    layer: usize,
    x: f64,
    y: f64,
    /// The passage number, if this node was given to the model.
    pick: Option<usize>,
}

/// A link of the lookup graph. `from` is `None` for the question itself.
struct GraphEdge {
    from: Option<usize>,
    to: usize,
    /// The passages whose route runs over this link.
    routes: Vec<usize>,
}

/// Lays the routes of a lookup out as one graph: the question on the left,
/// the nodes whose words matched next, then whatever was reached over links.
/// A node that several routes pass through appears once.
fn lookup_graph(picks: &[LookupPick]) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut ids: Vec<u64> = Vec::new();
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut edges: Vec<GraphEdge> = Vec::new();
    for (p, pick) in picks.iter().enumerate() {
        let mut previous: Option<usize> = None;
        for (depth, step) in pick.path.iter().enumerate() {
            let at = match ids.iter().position(|id| *id == step.id) {
                Some(at) => at,
                None => {
                    ids.push(step.id);
                    nodes.push(GraphNode {
                        kind: step.kind.clone(),
                        title: step.title.clone(),
                        layer: depth + 1,
                        x: 0.0,
                        y: 0.0,
                        pick: None,
                    });
                    nodes.len() - 1
                }
            };
            if depth + 1 == pick.path.len() {
                nodes[at].pick = Some(p);
            }
            match edges.iter_mut().find(|e| e.from == previous && e.to == at) {
                Some(edge) => edge.routes.push(p),
                None => edges.push(GraphEdge {
                    from: previous,
                    to: at,
                    routes: vec![p],
                }),
            }
            previous = Some(at);
        }
    }
    // Columns left to right by distance from the question, rows spread evenly.
    let layers = nodes.iter().map(|n| n.layer).max().unwrap_or(1);
    for layer in 1..=layers {
        let members: Vec<usize> = (0..nodes.len())
            .filter(|&i| nodes[i].layer == layer)
            .collect();
        let count = members.len();
        for (row, &i) in members.iter().enumerate() {
            nodes[i].x = QUESTION_X + (LAST_X - QUESTION_X) * layer as f64 / layers as f64;
            nodes[i].y = if count == 1 {
                // A lone node sits off the centre line so its links curve.
                if layer % 2 == 0 { 0.32 } else { -0.18 }
            } else {
                0.78 - 1.56 * row as f64 / (count - 1) as f64
            };
        }
    }
    (nodes, edges)
}

/// Where the question sits, and the rightmost column of nodes (which leaves
/// room for labels to its right).
const QUESTION_X: f64 = -1.82;
const LAST_X: f64 = 1.05;

/// Which neighbours a line cell joins: bits for up, right, down, left.
const UP: u8 = 1;
const RIGHT: u8 = 2;
const DOWN: u8 = 4;
const LEFT: u8 = 8;

/// The box-drawing character that joins the given neighbours, so that lines
/// meet, turn and branch without gaps.
fn line_char(joins: u8) -> char {
    match joins {
        j if j == UP | DOWN => '│',
        j if j == DOWN | RIGHT => '╭',
        j if j == DOWN | LEFT => '╮',
        j if j == UP | RIGHT => '╰',
        j if j == UP | LEFT => '╯',
        j if j == UP | DOWN | RIGHT => '├',
        j if j == UP | DOWN | LEFT => '┤',
        j if j == LEFT | RIGHT | DOWN => '┬',
        j if j == LEFT | RIGHT | UP => '┴',
        j if j == UP | DOWN | LEFT | RIGHT => '┼',
        j if j == UP || j == DOWN => '│',
        _ => '─',
    }
}

/// Draws the lookup graph with solid, connected lines: each link leaves its
/// node to the right, turns once up or down with a rounded corner, and runs
/// into the next node. Links that share a stretch merge and branch. The
/// selected passage's route is bright, with a pulse moving along it.
fn draw_lookup_graph(
    buf: &mut Buffer,
    area: Rect,
    nodes: &[GraphNode],
    edges: &[GraphEdge],
    sel: usize,
    lit: &[usize],
    tick: u64,
) {
    if area.width < 20 || area.height < 4 {
        return;
    }
    let (w, h) = (area.width as usize, area.height as usize);
    // Graph coordinates (x in -2..2, y in -1.1..1.1) to cells. The top row is
    // kept free for labels, which sit just above their node.
    let cell = |x: f64, y: f64| -> (usize, usize) {
        let col = ((x + 2.0) / 4.0 * (w - 1) as f64).round() as usize;
        let row = ((1.1 - y) / 2.2 * (h - 1) as f64).round() as usize;
        (col.min(w - 1), row.clamp(1, h - 1))
    };
    let question = cell(QUESTION_X, 0.0);
    let place = |at: Option<usize>| at.map_or(question, |i| cell(nodes[i].x, nodes[i].y));

    // Every link as the run of cells it passes through, in order.
    let mut joins = vec![0u8; w * h];
    let mut bright = vec![false; w * h];
    let mut route: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for e in edges {
        let (from, to) = (place(e.from), place(Some(e.to)));
        if to.0 <= from.0 + 1 {
            continue;
        }
        let turn = from.0 + (to.0 - from.0) / 2;
        let mut cells: Vec<(usize, usize)> = (from.0 + 1..=turn).map(|c| (c, from.1)).collect();
        if to.1 > from.1 {
            cells.extend((from.1 + 1..=to.1).map(|r| (turn, r)));
        } else {
            cells.extend((to.1..from.1).rev().map(|r| (turn, r)));
        }
        cells.extend((turn + 1..to.0).map(|c| (c, to.1)));
        // Each cell joins the one before and after it; the ends join the nodes.
        let mut previous = from;
        for (n, &(c, r)) in cells.iter().enumerate() {
            let next = cells.get(n + 1).copied().unwrap_or(to);
            for other in [previous, next] {
                joins[r * w + c] |= match (other.0.cmp(&c), other.1.cmp(&r)) {
                    (std::cmp::Ordering::Less, _) => LEFT,
                    (std::cmp::Ordering::Greater, _) => RIGHT,
                    (_, std::cmp::Ordering::Less) => UP,
                    _ => DOWN,
                };
            }
            previous = (c, r);
        }
        if e.routes.contains(&sel) {
            for &(c, r) in &cells {
                bright[r * w + c] = true;
            }
            route.push((nodes[e.to].layer, cells));
        }
    }
    // The pulse: a short bright run travelling the selected route outwards.
    route.sort_by_key(|(layer, _)| *layer);
    let along: Vec<(usize, usize)> = route.into_iter().flat_map(|(_, cells)| cells).collect();
    let head = if along.is_empty() {
        0
    } else {
        (tick as usize * 2) % (along.len() + 6)
    };
    let pulsing = |c: usize, r: usize| {
        along
            .iter()
            .position(|p| *p == (c, r))
            .is_some_and(|at| at <= head && head - at < 4)
    };

    for r in 0..h {
        for c in 0..w {
            let j = joins[r * w + c];
            if j == 0 {
                continue;
            }
            let style = if pulsing(c, r) {
                bold(NEON)
            } else if bright[r * w + c] {
                fg(GRID_HOT)
            } else {
                fg(lerp(GRID_HOT, PANEL, 0.68))
            };
            buf[(area.x + c as u16, area.y + r as u16)]
                .set_char(line_char(j))
                .set_style(style);
        }
    }

    // Nodes and their labels go on top of the lines.
    let mut put = |c: usize, r: usize, text: &str, style: Style| {
        for (n, ch) in text.chars().enumerate() {
            if c + n >= w {
                break;
            }
            buf[(area.x + (c + n) as u16, area.y + r as u16)]
                .set_char(ch)
                .set_style(style);
        }
    };
    put(question.0, question.1, "◉", bold(NEON));
    put(
        question.0.saturating_sub(1),
        question.1 - 1,
        "YOUR QUESTION",
        bold(NEON),
    );
    for (i, n) in nodes.iter().enumerate() {
        let (c, r) = cell(n.x, n.y);
        let on_route = lit.contains(&i);
        let color = node_color(&n.kind);
        let style = match (n.pick == Some(sel), on_route) {
            (true, _) => bold(NEON),
            (false, true) => bold(color),
            (false, false) => fg(lerp(color, PANEL, 0.4)),
        };
        // A ringed dot marks a passage that was given to the model.
        put(c, r, if n.pick.is_some() { "◉" } else { "●" }, style);
        let room = w.saturating_sub(c + 1).min(34);
        let label = match n.pick {
            Some(p) => trunc(&format!("#{} {}", p + 1, n.title), room),
            None => trunc(&n.title, room),
        };
        put(c, r - 1, &label, style);
    }
}

/// The lookup view: how a question travelled through the knowledge graph.
/// The question is on the left; solid lines fan out to the nodes whose words
/// matched and on to what those led to. Passages given to the model are
/// ringed and numbered; the selected one's route is lit, with a pulse
/// running along it, and its details are shown underneath.
fn render_lookup(f: &mut Frame, app: &mut App, idx: usize, area: Rect) {
    let Some(EventKind::KnowledgeLookup {
        query,
        words,
        seeds,
        reached,
        dropped,
        tokens,
        limits,
        picks,
        ..
    }) = app.events.get(idx).map(|e| e.kind.clone())
    else {
        app.modal = None;
        return;
    };
    let r = centered(area, 94, 92);
    app.hits.modal = Some(r);
    f.render_widget(Clear, r);
    let inner = hud(f, r, "◈ KNOWLEDGE LOOKUP", VIOLET, true, None);
    let [summary, graph_area, detail, keys] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Percentage(50),
        Constraint::Min(6),
        Constraint::Length(1),
    ])
    .areas(inner);

    // What was asked, what was searched for, and how it narrowed down.
    let sw = (summary.width as usize).saturating_sub(12);
    let left_out = if dropped > 0 {
        format!(" · {dropped} more did not fit the limits")
    } else {
        String::new()
    };
    f.render_widget(
        Paragraph::new(vec![
            kv("QUESTION", trunc(&format!("\"{query}\""), sw), bold(TEXT)),
            kv("SEARCHED", trunc(&words.join(" · "), sw), fg(AMBER)),
            kv("LIMITS", trunc(&limits, sw), fg(DIM)),
            Line::from(vec![
                Span::styled(format!(" {:<9}", "FUNNEL"), fg(DIM)),
                Span::styled(
                    format!(
                        "{seeds} node{} matched your words",
                        if seeds == 1 { "" } else { "s" }
                    ),
                    bold(NEON),
                ),
                Span::styled(" → ", fg(GRID_HOT)),
                Span::styled(format!("{reached} reached by following links"), bold(CYAN)),
                Span::styled(" → ", fg(GRID_HOT)),
                Span::styled(
                    format!(
                        "{} passage{} given to the model ({tokens} tokens)",
                        picks.len(),
                        if picks.len() == 1 { "" } else { "s" }
                    ),
                    bold(VIOLET),
                ),
                Span::styled(left_out, fg(DIM)),
            ]),
        ]),
        summary,
    );

    let sel = app.lookup_sel.min(picks.len().saturating_sub(1));
    let (nodes, edges) = lookup_graph(&picks);
    // Nodes on the selected passage's route.
    let lit: Vec<usize> = edges
        .iter()
        .filter(|e| e.routes.contains(&sel))
        .map(|e| e.to)
        .collect();
    draw_lookup_graph(
        f.buffer_mut(),
        graph_area,
        &nodes,
        &edges,
        sel,
        &lit,
        app.tick,
    );

    // Underneath: the selected passage, in words.
    let Some(pick) = picks.get(sel) else {
        f.render_widget(
            Paragraph::new(Line::styled(
                " nothing was retrieved for this question",
                fg(DIM),
            )),
            detail,
        );
        return;
    };
    let dw = (detail.width as usize).saturating_sub(12);
    let mut route = vec!["your question".to_string()];
    route.extend(pick.path.iter().map(|s| s.title.clone()));
    let how = match pick.hops {
        0 => "its own words matched your question".to_string(),
        n => format!(
            "reached {n} link{} away from the words that matched",
            if n == 1 { "" } else { "s" }
        ),
    };
    let mut v = vec![
        section(
            &format!(
                "PASSAGE {} OF {} · {}",
                sel + 1,
                picks.len(),
                trunc(&pick.title, dw.saturating_sub(24))
            ),
            detail.width + 8,
        ),
        kv("ROUTE", trunc(&route.join(" → "), dw), fg(TEXT)),
        kv(
            "WHY",
            trunc(
                &if pick.matched.is_empty() {
                    how
                } else {
                    format!("{how} (words found: {})", pick.matched.join(", "))
                },
                dw,
            ),
            fg(TEXT),
        ),
        kv(
            "FROM",
            format!(
                "{} · {} · relevance {:.2} · {} tokens",
                pick.source, pick.trust, pick.score, pick.tokens
            ),
            fg(TEXT),
        ),
        section("WHAT THE MODEL WAS SHOWN", detail.width + 8),
    ];
    for line in pick.text.lines() {
        for part in wrap(line, (detail.width as usize).saturating_sub(2)) {
            v.push(Line::from(vec![
                Span::raw(" "),
                Span::styled(part, fg(lerp(TEXT, DIM, 0.25))),
            ]));
        }
    }
    f.render_widget(Paragraph::new(v).scroll((app.modal_scroll, 0)), detail);

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ↑↓", bold(NEON)),
            Span::styled(" next passage   ", fg(DIM)),
            Span::styled("PgUp PgDn", bold(NEON)),
            Span::styled(" scroll text   ", fg(DIM)),
            Span::styled("ESC", bold(NEON)),
            Span::styled(" close      ", fg(DIM)),
            Span::styled("● ", fg(CYAN)),
            Span::styled("passage  ", fg(DIM)),
            Span::styled("● ", fg(AMBER)),
            Span::styled("shared term  ", fg(DIM)),
            Span::styled("● ", fg(NEON)),
            Span::styled("memory  ", fg(DIM)),
            Span::styled("◯ ", fg(NEON)),
            Span::styled("given to the model", fg(DIM)),
        ])),
        keys,
    );
}

fn render_modal(f: &mut Frame, app: &mut App, modal: Modal, area: Rect) {
    if let Modal::Lookup(idx) = modal {
        return render_lookup(f, app, idx, area);
    }
    let r = centered(area, 70, 72);
    app.hits.modal = Some(r);

    let (title, accent, lines, pending): (String, Color, Vec<Line<'static>>, bool) = match modal {
        Modal::Tool(id) => {
            let Some(tc) = app.tool(id).cloned() else {
                app.modal = None;
                return;
            };
            let (b, bs) = tool_badge(app, &tc);
            let pending = tc.status == ToolStatus::AwaitingApproval;
            let mut v = vec![
                kv("TOOL", tc.tool.clone(), bold(CYAN)),
                kv("AGENT", app.agent_name(tc.agent).to_uppercase(), fg(TEXT)),
                kv("RISK", tc.risk.label().into(), bold(risk_color(tc.risk))),
                kv("POLICY", tc.policy.clone(), fg(TEXT)),
                Line::from(vec![
                    Span::styled(format!(" {:<9}", "STATUS"), fg(DIM)),
                    Span::styled(b, bs),
                ]),
                kv(
                    "DECIDED",
                    tc.decided_by.clone().unwrap_or_else(|| "—".into()),
                    fg(TEXT),
                ),
                kv(
                    "TIME",
                    format!(
                        "{}  ·  {}",
                        t(tc.requested_at),
                        tc.duration_ms
                            .map(|d| format!("{d}ms"))
                            .unwrap_or_else(|| "—".into())
                    ),
                    fg(TEXT),
                ),
                Line::from(""),
                section("ARGS", r.width),
            ];
            v.extend(code_lines(&tc.args));
            v.push(Line::from(""));
            v.push(section("OUTPUT", r.width));
            match &tc.output {
                Some(o) => v.extend(code_lines(o)),
                None => v.push(Line::styled("   // no output yet", fg(DIM))),
            }
            let accent = if pending { AMBER } else { CYAN };
            (format!("λ TOOL CALL #{}", tc.id), accent, v, pending)
        }
        // Drawn by `render_lookup`, above.
        Modal::Lookup(_) => return,
        Modal::Event(idx) => {
            let Some(ev) = app.events.get(idx) else {
                app.modal = None;
                return;
            };
            // A reply is easier to read rendered than as a debug dump.
            if let EventKind::Message {
                agent: Some(a),
                text,
            } = &ev.kind
            {
                let mut v = vec![
                    kv("FROM", app.agent_name(*a), bold(CYAN)),
                    kv("TIME", t(ev.at), fg(TEXT)),
                    Line::from(""),
                ];
                v.extend(
                    markdown::render(text, (r.width as usize).saturating_sub(6))
                        .into_iter()
                        .map(|mut row| {
                            row.insert(0, Span::raw(" "));
                            Line::from(row)
                        }),
                );
                (format!("◆ REPLY 0x{:04X}", ev.seq), CYAN, v, false)
            } else {
                let mut v = vec![
                    kv("SEQ", format!("0x{:04X}", ev.seq), fg(TEXT)),
                    kv("TIME", t(ev.at), fg(TEXT)),
                    kv("KIND", ev.kind.name().into(), bold(MAGENTA)),
                    Line::from(""),
                    section("PAYLOAD", r.width),
                ];
                v.extend(code_lines(&format!("{:#?}", ev.kind)));
                (format!("◇ EVENT 0x{:04X}", ev.seq), MAGENTA, v, false)
            }
        }
        Modal::Context(i) => {
            let Some(s) = app.context.get(i) else {
                app.modal = None;
                return;
            };
            let used = app.context_used().max(1);
            let v = vec![
                kv(
                    "SECTION",
                    s.name.to_uppercase(),
                    bold(ACCENTS[i % ACCENTS.len()]),
                ),
                kv("TOKENS", s.tokens.to_string(), fg(TEXT)),
                kv(
                    "SHARE",
                    format!(
                        "{:.0}% of context · {:.1}% of window",
                        s.tokens as f64 / used as f64 * 100.0,
                        s.tokens as f64 / app.window.max(1) as f64 * 100.0
                    ),
                    fg(TEXT),
                ),
                Line::from(""),
                section("WHY IT IS IN CONTEXT", r.width),
                Line::from(vec![
                    Span::raw("   "),
                    Span::styled(s.why.clone(), fg(TEXT)),
                ]),
            ];
            ("▤ CONTEXT SECTION".into(), CYAN, v, false)
        }
        Modal::Memory(id) => {
            let shown = app.shown_memories();
            let Some(at) = shown.iter().position(|m| m.node.id == id) else {
                app.modal = None;
                return;
            };
            let m = shown[at];
            let kind = match m.node.kind {
                MemKind::Episodic => "● something that happened",
                MemKind::Semantic => "◆ a fact",
                MemKind::Procedural => {
                    "■ always kept in mind (who you are, how you want things done)"
                }
            };
            let mut v = vec![
                kv("MEMORY", m.node.label.clone(), bold(TEXT)),
                kv("KIND", kind.into(), fg(mem_color(m.node.kind))),
            ];
            if !m.node.about.is_empty() {
                v.push(kv("ABOUT IT", m.node.about.clone(), fg(TEXT)));
            }
            if m.recalls > 0 {
                v.push(kv(
                    "USED",
                    format!(
                        "{} time{} since this screen opened",
                        m.recalls,
                        if m.recalls == 1 { "" } else { "s" }
                    ),
                    fg(TEXT),
                ));
            }
            v.push(Line::from(""));
            v.push(section("WHAT IS REMEMBERED", r.width));
            let body = if m.node.text.is_empty() {
                "(only its title was sent to this screen)".to_string()
            } else {
                m.node.text.clone()
            };
            for line in body.lines() {
                for l in wrap(line, (r.width as usize).saturating_sub(8)) {
                    v.push(Line::from(vec![
                        Span::raw("   "),
                        Span::styled(l, fg(TEXT)),
                    ]));
                }
            }
            v.push(Line::from(""));
            v.push(Line::styled(
                format!(
                    "   {} of {} · ↑↓ next / previous · esc closes",
                    at + 1,
                    shown.len()
                ),
                fg(DIM),
            ));
            ("◈ MEMORY".into(), VIOLET, v, false)
        }
        Modal::Agent(i) => {
            let Some(a) = app.agents.get(i).cloned() else {
                app.modal = None;
                return;
            };
            let mut v = vec![
                kv("NAME", a.name.to_uppercase(), bold(TEXT)),
                kv("ID", a.id.to_string(), fg(TEXT)),
                kv(
                    "PARENT",
                    a.parent
                        .map(|p| app.agent_name(p))
                        .unwrap_or_else(|| "— (core)".into()),
                    fg(TEXT),
                ),
                kv("STATE", a.state.label().into(), bold(state_color(a.state))),
                kv("TOKENS", tok(a.tokens), fg(TEXT)),
                kv("PURPOSE", a.purpose.clone(), fg(TEXT)),
                Line::from(""),
                section("TOOL CALLS", r.width),
            ];
            for tc in app.tools.iter().filter(|tc| tc.agent == a.id) {
                let (b, bs) = tool_badge(app, tc);
                v.push(Line::from(vec![
                    Span::styled(format!("   #{:<3}", tc.id), fg(DIM)),
                    Span::styled(format!("{} ", tc.tool), bold(CYAN)),
                    Span::styled(trunc(tc.args.lines().next().unwrap_or(""), 40), fg(TEXT)),
                    Span::raw(" "),
                    Span::styled(b, bs),
                ]));
            }
            v.push(Line::from(""));
            v.push(Line::styled(
                " press f in the AGENTS panel to filter the timeline to this agent",
                fg(DIM),
            ));
            ("◉ AGENT".into(), VIOLET, v, false)
        }
        Modal::Help => {
            let keys = [
                ("tab / shift-tab", "move focus between panels"),
                ("0", "main: the conversation that keeps going"),
                ("K", "type a secret (a key or token) into its own window"),
                ("b", "the branches: each one and where it stands"),
                (
                    "e",
                    "list the branches that have ended too, or hide them again",
                ),
                ("] [ / → ←", "next / previous tab"),
                ("1 2 3", "in a tab: activity · raw events · knowledge"),
                ("n", "say something in main (it decides whether to branch)"),
                ("X", "stop what the open tab is doing"),
                ("↑↓ / j k", "select in the focused panel"),
                (
                    "enter / click",
                    "inspect selection (tool, event, agent, context)",
                ),
                ("a / d", "accept / deny the pending tool call"),
                ("f / click agent twice", "filter timeline by selected agent"),
                (
                    "pgup pgdn / wheel",
                    "scroll the timeline · G jumps back to live",
                ),
                (
                    "i",
                    "reply in the open branch once it has answered, else talk to main",
                ),
                ("g", "show how the last knowledge lookup found its passages"),
                (
                    "+ / -",
                    "rate the open task's answer good / bad (teaches routing)",
                ),
                ("r", "replay the demo"),
                ("q / ctrl-c", "quit"),
            ];
            let mut v = vec![Line::from("")];
            for (k, d) in keys {
                v.push(Line::from(vec![
                    Span::styled(format!("   {k:<24}"), bold(NEON)),
                    Span::styled(d, fg(TEXT)),
                ]));
            }
            v.push(Line::from(""));
            v.push(Line::styled(
                "   // all data in this build is simulated by src/demo.rs",
                fg(DIM),
            ));
            ("? KEYS".into(), NEON, v, false)
        }
    };

    // Drop shadow, offset one cell down and two right.
    {
        let buf = f.buffer_mut();
        let right = (r.x + r.width + 2).min(area.x + area.width);
        let bottom = (r.y + r.height + 1).min(area.y + area.height);
        for y in r.y + 1..bottom {
            for x in r.x + 2..right {
                if x >= r.x + r.width || y >= r.y + r.height {
                    buf[(x, y)].set_bg(SHADOW).set_fg(GRID);
                }
            }
        }
    }
    f.render_widget(Clear, r);
    let inner = hud(f, r, &title, accent, true, None);

    let [content, _, buttons] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((app.modal_scroll, 0)),
        content,
    );

    let list: Vec<(&str, Color, Button)> = if pending {
        vec![
            (" A ▸ ACCEPT ", NEON, Button::Accept),
            (" D ▸ DENY ", RED, Button::Deny),
            (" ESC ▸ LATER ", DIM, Button::Close),
        ]
    } else {
        vec![(" ESC ▸ CLOSE ", DIM, Button::Close)]
    };
    let mut x = buttons.x + 1;
    for (label, color, b) in list {
        let w = chip_width(label);
        let br = Rect {
            x,
            y: buttons.y,
            width: w,
            height: 1,
        };
        f.render_widget(
            Paragraph::new(Line::from(chip(label, color, color != DIM))),
            br,
        );
        app.hits.buttons.push((br, b));
        x += w + 2;
    }
    if pending {
        f.render_widget(
            Paragraph::new(Line::styled(
                "guardrail ▸ this call needs your decision ",
                fg(AMBER),
            ))
            .alignment(Alignment::Right),
            buttons,
        );
    }
}
