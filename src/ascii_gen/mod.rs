//! ASCII art generation module.
//!
//! Produces 2D character grids with color data for visual game assets.
//! Output is constrained to the game's glyph atlas vocabulary and color
//! palette, ensuring compatibility with the existing ASCII viewport renderer.
//!
//! Supports model-backed generation (via [`ModelClient`]) with procedural
//! template fallback when no model is available.

pub mod grid;
pub mod palette;
pub mod vocab;
pub mod generator;
pub mod templates;

#[allow(unused_imports)]
pub use grid::{AsciiCell, AsciiGrid};
#[allow(unused_imports)]
pub use generator::{AsciiArtGenerator, AsciiArtRequest, AsciiArtStyle, AsciiGenError};
#[allow(unused_imports)]
pub use palette::{snap_to_palette, Rgb, GAME_PALETTE};
#[allow(unused_imports)]
pub use vocab::GlyphVocab;
