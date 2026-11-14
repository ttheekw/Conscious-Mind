//! Colour palette and small style helpers for the neon terminal look.

use ratatui::style::{Color, Modifier, Style};

pub const BG: Color = Color::Rgb(5, 8, 13);
pub const PANEL: Color = Color::Rgb(8, 12, 19);
pub const SELECT: Color = Color::Rgb(16, 34, 44);
pub const GRID: Color = Color::Rgb(30, 48, 60);
pub const GRID_HOT: Color = Color::Rgb(0, 210, 180);
pub const TEXT: Color = Color::Rgb(182, 206, 216);
pub const DIM: Color = Color::Rgb(82, 104, 116);
pub const NEON: Color = Color::Rgb(0, 255, 156);
pub const CYAN: Color = Color::Rgb(0, 220, 255);
pub const MAGENTA: Color = Color::Rgb(255, 46, 170);
pub const VIOLET: Color = Color::Rgb(156, 116, 255);
pub const AMBER: Color = Color::Rgb(255, 190, 40);
pub const RED: Color = Color::Rgb(255, 64, 96);

/// Header band, slightly lifted from the page background.
pub const BAND: Color = Color::Rgb(9, 16, 25);
/// Dot-grid texture drawn in empty panel rows.
pub const TEXTURE: Color = Color::Rgb(22, 35, 46);
/// Drop shadow under modals.
pub const SHADOW: Color = Color::Rgb(2, 3, 6);

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn fg(c: Color) -> Style {
    Style::new().fg(c)
}

pub fn bold(c: Color) -> Style {
    Style::new().fg(c).add_modifier(Modifier::BOLD)
}

pub fn badge(fg_c: Color, bg_c: Color) -> Style {
    Style::new().fg(fg_c).bg(bg_c).add_modifier(Modifier::BOLD)
}

/// Linear blend between two RGB colours, `t` in 0..=1.
pub fn lerp(a: Color, b: Color, t: f32) -> Color {
    match (a, b) {
        (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
            let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
            Color::Rgb(mix(r1, r2), mix(g1, g2), mix(b1, b2))
        }
        _ => a,
    }
}

/// Cycling accent colours for context sections and similar lists.
pub const ACCENTS: [Color; 5] = [CYAN, MAGENTA, NEON, VIOLET, AMBER];
