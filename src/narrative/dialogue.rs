//! Dialogue system — dialogue lines and exchanges for NPC interactions.

use serde::{Deserialize, Serialize};

/// A single line of dialogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueLine {
    pub speaker: String,
    pub text: String,
}

/// A dialogue exchange — a sequence of lines between characters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueExchange {
    pub id: String,
    pub lines: Vec<DialogueLine>,
    /// Optional choices presented after the exchange.
    #[serde(default)]
    pub choices: Vec<DialogueChoice>,
}

/// A player choice within a dialogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueChoice {
    pub label: String,
    /// Tag used to determine consequences.
    pub consequence_tag: String,
    /// Optional next dialogue exchange to chain.
    #[serde(default)]
    pub next_exchange_id: Option<String>,
}

/// State of an ongoing dialogue interaction.
#[derive(Debug, Clone, Default)]
pub struct DialogueState {
    pub current_exchange_id: Option<String>,
    pub current_line_index: usize,
    pub is_complete: bool,
}

impl DialogueState {
    pub fn start(exchange_id: String) -> Self {
        Self {
            current_exchange_id: Some(exchange_id),
            current_line_index: 0,
            is_complete: false,
        }
    }

    /// Advance to the next line. Returns true if there are more lines.
    pub fn advance(&mut self, exchange: &DialogueExchange) -> bool {
        if self.current_line_index + 1 < exchange.lines.len() {
            self.current_line_index += 1;
            true
        } else {
            self.is_complete = true;
            false
        }
    }

    /// Get the current dialogue line from the exchange.
    pub fn current_line<'a>(&self, exchange: &'a DialogueExchange) -> Option<&'a DialogueLine> {
        exchange.lines.get(self.current_line_index)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialogue_state_advance() {
        let exchange = DialogueExchange {
            id: "test".to_string(),
            lines: vec![
                DialogueLine {
                    speaker: "A".to_string(),
                    text: "Hello".to_string(),
                },
                DialogueLine {
                    speaker: "B".to_string(),
                    text: "Hi!".to_string(),
                },
            ],
            choices: Vec::new(),
        };

        let mut state = DialogueState::start("test".to_string());
        assert_eq!(state.current_line(&exchange).unwrap().text, "Hello");

        let more = state.advance(&exchange);
        assert!(more);
        assert_eq!(state.current_line(&exchange).unwrap().text, "Hi!");

        let more = state.advance(&exchange);
        assert!(!more);
        assert!(state.is_complete);
    }
}
