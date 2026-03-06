use bevy::prelude::*;

use crate::ai::effects::Tags;

/// Maps a tag string (case-insensitive) to an emissive Color.
pub fn tag_to_color(tag: &str) -> Color {
    match tag.to_ascii_lowercase().as_str() {
        "fire" => Color::rgb(1.0, 0.5, 0.0),
        "ice" | "frost" => Color::rgb(0.3, 0.9, 1.0),
        "holy" | "light" => Color::rgb(1.0, 0.85, 0.2),
        "poison" | "nature" => Color::rgb(0.2, 0.9, 0.2),
        "dark" | "shadow" | "void" => Color::rgb(0.6, 0.2, 0.9),
        "physical" | "melee" => Color::rgb(0.9, 0.9, 0.9),
        "lightning" | "electric" => Color::rgb(0.9, 0.9, 0.3),
        "arcane" => Color::rgb(0.8, 0.3, 1.0),
        "water" => Color::rgb(0.2, 0.5, 1.0),
        _ => Color::rgb(0.7, 0.7, 0.7),
    }
}

/// Returns the color for the highest-weight tag, or light grey fallback.
pub fn primary_tag_color(tags: &Tags) -> Color {
    tags.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(tag, _)| tag_to_color(tag))
        .unwrap_or(Color::rgb(0.7, 0.7, 0.7))
}
