use serde::{Deserialize, Serialize};

use super::grid::{AsciiCell, AsciiGrid};
use super::palette::{snap_to_palette, Rgb};
use super::templates;
use super::vocab::GlyphVocab;
use crate::model_backend::{ModelClient, ModelError, format_json_generation_prompt};

/// Style of ASCII art to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsciiArtStyle {
    Environment,
    CharacterPortrait,
    ItemIcon,
    UiDecoration,
}

/// Request for ASCII art generation.
#[derive(Debug, Clone)]
pub struct AsciiArtRequest {
    pub prompt: String,
    pub style: AsciiArtStyle,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    /// Optional color constraint — if provided, only these colors may appear.
    pub palette_constraint: Option<Vec<Rgb>>,
}

/// Errors from the ASCII art generator.
#[derive(Debug)]
pub enum AsciiGenError {
    Model(ModelError),
    ParseFailed(String),
}

impl std::fmt::Display for AsciiGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsciiGenError::Model(e) => write!(f, "model error: {e}"),
            AsciiGenError::ParseFailed(msg) => write!(f, "parse failed: {msg}"),
        }
    }
}

impl std::error::Error for AsciiGenError {}

/// ASCII art generator with model backend and procedural fallback.
pub struct AsciiArtGenerator {
    model: Option<ModelClient>,
    vocab: GlyphVocab,
}

impl AsciiArtGenerator {
    pub fn new(model: Option<ModelClient>) -> Self {
        Self {
            model,
            vocab: GlyphVocab::game_default(),
        }
    }

    /// Generate ASCII art for the given request.
    ///
    /// Tries the model backend first; falls back to procedural templates.
    pub fn generate(&self, request: &AsciiArtRequest) -> Result<AsciiGrid, AsciiGenError> {
        // Try model-backed generation if available.
        if let Some(ref model) = self.model {
            if model.is_available() {
                match self.generate_with_model(model, request) {
                    Ok(grid) => return Ok(grid),
                    Err(_) => {
                        // Fall through to procedural.
                    }
                }
            }
        }

        // Procedural fallback.
        Ok(self.generate_procedural(request))
    }

    fn generate_with_model(
        &self,
        model: &ModelClient,
        request: &AsciiArtRequest,
    ) -> Result<AsciiGrid, AsciiGenError> {
        let system_ctx = format!(
            "You are an ASCII art generator. Produce a {}x{} character grid. \
             Use only standard printable ASCII characters, box-drawing characters (U+2500-U+257F), \
             and block elements (U+2580-U+259F). Style: {:?}.",
            request.width, request.height, request.style
        );

        let schema = r#"{
  "grid": [["char", ...]],
  "fg_colors": [[[r,g,b], ...]],
  "bg_colors": [[[r,g,b] | null, ...]]
}"#;

        let prompt = format_json_generation_prompt(&system_ctx, &request.prompt, schema);

        let text = model
            .generate(&prompt, request.seed)
            .map_err(AsciiGenError::Model)?;

        self.parse_model_output(&text, request.width, request.height, &request.palette_constraint)
    }

    fn parse_model_output(
        &self,
        text: &str,
        width: usize,
        height: usize,
        palette_constraint: &Option<Vec<Rgb>>,
    ) -> Result<AsciiGrid, AsciiGenError> {
        let val: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| AsciiGenError::ParseFailed(format!("invalid JSON: {e}")))?;

        let grid_arr = val["grid"]
            .as_array()
            .ok_or_else(|| AsciiGenError::ParseFailed("missing 'grid' array".to_string()))?;

        let mut grid = AsciiGrid::new(width, height);

        for (row, row_val) in grid_arr.iter().enumerate().take(height) {
            let row_arr = row_val
                .as_array()
                .ok_or_else(|| AsciiGenError::ParseFailed("row is not an array".to_string()))?;

            for (col, cell_val) in row_arr.iter().enumerate().take(width) {
                let ch_str = cell_val.as_str().unwrap_or(" ");
                let ch = ch_str.chars().next().unwrap_or(' ');
                let ch = self.vocab.clamp(ch);

                let fg = self.extract_color(&val["fg_colors"], row, col)
                    .unwrap_or([0xAA, 0xAA, 0xAA]);
                let bg = self.extract_color(&val["bg_colors"], row, col);

                let fg = self.apply_palette_constraint(fg, palette_constraint);
                let bg = bg.map(|c| self.apply_palette_constraint(c, palette_constraint));

                grid.set(col, row, AsciiCell { ch, fg, bg });
            }
        }

        Ok(grid)
    }

    fn extract_color(
        &self,
        colors: &serde_json::Value,
        row: usize,
        col: usize,
    ) -> Option<Rgb> {
        let row_arr = colors.as_array()?.get(row)?.as_array()?;
        let cell = row_arr.get(col)?;
        if cell.is_null() {
            return None;
        }
        let arr = cell.as_array()?;
        if arr.len() >= 3 {
            Some([
                arr[0].as_u64()? as u8,
                arr[1].as_u64()? as u8,
                arr[2].as_u64()? as u8,
            ])
        } else {
            None
        }
    }

    fn apply_palette_constraint(&self, color: Rgb, constraint: &Option<Vec<Rgb>>) -> Rgb {
        if let Some(ref pal) = constraint {
            // Snap to the provided palette.
            let mut best = pal[0];
            let mut best_dist = u32::MAX;
            for &p in pal {
                let dr = (color[0] as i32 - p[0] as i32).unsigned_abs();
                let dg = (color[1] as i32 - p[1] as i32).unsigned_abs();
                let db = (color[2] as i32 - p[2] as i32).unsigned_abs();
                let dist = dr * dr + dg * dg + db * db;
                if dist < best_dist {
                    best_dist = dist;
                    best = p;
                }
            }
            best
        } else {
            // Snap to default game palette.
            snap_to_palette(color)
        }
    }

    fn generate_procedural(&self, request: &AsciiArtRequest) -> AsciiGrid {
        match request.style {
            AsciiArtStyle::Environment => {
                templates::generate_environment(request.width, request.height, request.seed)
            }
            AsciiArtStyle::CharacterPortrait => {
                templates::generate_character_portrait(request.width, request.height, request.seed)
            }
            AsciiArtStyle::ItemIcon => {
                templates::generate_item_icon(request.width, request.height, request.seed)
            }
            AsciiArtStyle::UiDecoration => {
                // Reuse environment template with a different flavor.
                templates::generate_environment(request.width, request.height, request.seed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_without_model_produces_grid() {
        let gen = AsciiArtGenerator::new(None);
        let request = AsciiArtRequest {
            prompt: "dark cave entrance".to_string(),
            style: AsciiArtStyle::Environment,
            width: 20,
            height: 10,
            seed: 42,
            palette_constraint: None,
        };
        let grid = gen.generate(&request).unwrap();
        assert_eq!(grid.width, 20);
        assert_eq!(grid.height, 10);
    }

    #[test]
    fn generate_portrait_produces_grid() {
        let gen = AsciiArtGenerator::new(None);
        let request = AsciiArtRequest {
            prompt: "brave warrior".to_string(),
            style: AsciiArtStyle::CharacterPortrait,
            width: 9,
            height: 9,
            seed: 77,
            palette_constraint: None,
        };
        let grid = gen.generate(&request).unwrap();
        assert_eq!(grid.width, 9);
        assert_eq!(grid.height, 9);
    }

    #[test]
    fn generate_item_icon() {
        let gen = AsciiArtGenerator::new(None);
        let request = AsciiArtRequest {
            prompt: "enchanted sword".to_string(),
            style: AsciiArtStyle::ItemIcon,
            width: 7,
            height: 7,
            seed: 123,
            palette_constraint: None,
        };
        let grid = gen.generate(&request).unwrap();
        assert_eq!(grid.width, 7);
    }

    #[test]
    fn deterministic_generation() {
        let gen = AsciiArtGenerator::new(None);
        let req = AsciiArtRequest {
            prompt: "test".to_string(),
            style: AsciiArtStyle::Environment,
            width: 15,
            height: 8,
            seed: 42,
            palette_constraint: None,
        };
        let a = gen.generate(&req).unwrap();
        let b = gen.generate(&req).unwrap();
        assert_eq!(a.to_plain_text(), b.to_plain_text());
    }
}
