//! Ability DSL parser — brace-based syntax for hero ability definitions.
//!
//! Parses `.ability` files into `AbilityDef` / `PassiveDef` structs,
//! replacing verbose TOML arrays with a compact, readable format.
//!
//! # Example
//! ```text
//! ability Fireball {
//!     target: enemy, range: 5.0
//!     cooldown: 5s, cast: 300ms
//!     hint: damage
//!
//!     deliver projectile { speed: 8.0, width: 0.3 } {
//!         on_hit { damage 55 [FIRE: 60] }
//!         on_arrival { damage 15 in circle(2.0) }
//!     }
//! }
//! ```

pub mod ast;
pub mod error;
pub mod lower;
pub mod parser;

#[cfg(test)]
mod tests;

use crate::ai::effects::defs::{AbilityDef, PassiveDef};
use error::DslError;

/// Parse a `.ability` file string into ability and passive definitions.
pub fn parse_abilities(input: &str) -> Result<(Vec<AbilityDef>, Vec<PassiveDef>), DslError> {
    let mut remaining = input;
    let file = parser::parse_ability_file(&mut remaining).map_err(|e| {
        let offset = input.len() - remaining.len();
        let (line, col) = error::offset_to_line_col(input, offset);
        DslError {
            message: format!("parse error: {e}"),
            line,
            col,
            context: error::extract_line(input, line),
        }
    })?;

    lower::lower_file(&file).map_err(|msg| DslError {
        message: msg,
        line: 0,
        col: 0,
        context: None,
    })
}
