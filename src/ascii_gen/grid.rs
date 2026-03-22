use serde::{Deserialize, Serialize};

use super::palette::Rgb;

/// A single cell in an ASCII art grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsciiCell {
    /// The character to display.
    pub ch: char,
    /// Foreground color (RGB).
    pub fg: Rgb,
    /// Optional background color (RGB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bg: Option<Rgb>,
}

impl Default for AsciiCell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: [0xAA, 0xAA, 0xAA],
            bg: None,
        }
    }
}

/// A 2D grid of ASCII cells with color data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsciiGrid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<AsciiCell>>,
}

impl AsciiGrid {
    /// Create a blank grid filled with spaces.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![vec![AsciiCell::default(); width]; height],
        }
    }

    /// Get cell at (col, row).
    pub fn get(&self, col: usize, row: usize) -> Option<&AsciiCell> {
        self.cells.get(row).and_then(|r| r.get(col))
    }

    /// Set cell at (col, row).
    pub fn set(&mut self, col: usize, row: usize, cell: AsciiCell) {
        if row < self.height && col < self.width {
            self.cells[row][col] = cell;
        }
    }

    /// Render the grid as a plain-text string (no color info).
    pub fn to_plain_text(&self) -> String {
        let mut out = String::with_capacity(self.width * self.height + self.height);
        for row in &self.cells {
            for cell in row {
                out.push(cell.ch);
            }
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid_has_correct_dimensions() {
        let grid = AsciiGrid::new(10, 5);
        assert_eq!(grid.width, 10);
        assert_eq!(grid.height, 5);
        assert_eq!(grid.cells.len(), 5);
        assert_eq!(grid.cells[0].len(), 10);
    }

    #[test]
    fn set_and_get_cell() {
        let mut grid = AsciiGrid::new(3, 3);
        grid.set(1, 2, AsciiCell { ch: '#', fg: [255, 0, 0], bg: None });
        assert_eq!(grid.get(1, 2).unwrap().ch, '#');
    }

    #[test]
    fn plain_text_output() {
        let mut grid = AsciiGrid::new(3, 2);
        grid.set(0, 0, AsciiCell { ch: 'A', ..Default::default() });
        grid.set(1, 0, AsciiCell { ch: 'B', ..Default::default() });
        grid.set(2, 0, AsciiCell { ch: 'C', ..Default::default() });
        let text = grid.to_plain_text();
        assert!(text.starts_with("ABC\n"));
    }
}
