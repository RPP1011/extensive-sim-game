//! The `UiModel` types double as the deserialization target for the
//! compiler-emitted `ui_descriptor()` JSON (Plan A/C). The serde reprs are
//! the **default externally-tagged** form (`{"Bar":{...}}`, `"Restart"`,
//! `{"Increment":"bolt_level"}`) — the `dsl_compiler::cg::emit::ui_model`
//! emitter hand-writes JSON in exactly this shape, validated by the
//! `from_json_roundtrip` test below + the compiler-side compile gate.

use serde::{Deserialize, Serialize};

/// A value lookup key into UiData (a named sim-readback value).
pub type BindKey = String;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Widget {
    /// Horizontal bar: value/max fraction, labelled, RGB color.
    Bar {
        label: String,
        value: BindKey,
        max: BindKey,
        color: [u8; 3],
    },
    /// Text with `{key}` placeholders substituted from UiData (formatted as ints).
    Text { template: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum UiAction {
    /// Increment a named host-side counter (e.g. "bolt_level"). Applied by the caller.
    Increment(String),
    /// Restart the run.
    Restart,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Card {
    pub label: String,
    pub action: UiAction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Screen {
    /// Modal upgrade menu — pauses; cards are buttons returning their action.
    Menu { title: String, cards: Vec<Card> },
    /// Modal end screen — summary rows (label, BindKey) + a restart button.
    End {
        title: String,
        summary: Vec<(String, BindKey)>,
        restart_label: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NamedScreen {
    pub name: String,
    pub screen: Screen,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct UiModel {
    pub hud: Vec<Widget>,
    pub screens: Vec<NamedScreen>,
}

impl UiModel {
    pub fn screen(&self, name: &str) -> Option<&Screen> {
        self.screens
            .iter()
            .find(|s| s.name == name)
            .map(|s| &s.screen)
    }

    /// Parse a compiler-emitted `ui_descriptor()` JSON string. Fallible so
    /// the viewer can fall back to a hand-built model on parse error (P10).
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_roundtrip() {
        let m = UiModel {
            hud: vec![
                Widget::Bar {
                    label: "HP".into(),
                    value: "hp".into(),
                    max: "hp_max".into(),
                    color: [220, 40, 40],
                },
                Widget::Text { template: "Lv {level}".into() },
            ],
            screens: vec![
                NamedScreen {
                    name: "level_up".into(),
                    screen: Screen::Menu {
                        title: "Level Up!".into(),
                        cards: vec![Card {
                            label: "Bolt +".into(),
                            action: UiAction::Increment("bolt_level".into()),
                        }],
                    },
                },
                NamedScreen {
                    name: "dead".into(),
                    screen: Screen::End {
                        title: "You Died".into(),
                        summary: vec![("time".into(), "time".into())],
                        restart_label: "Restart".into(),
                    },
                },
            ],
        };
        let j = serde_json::to_string(&m).unwrap();
        let back = UiModel::from_json(&j).unwrap();
        assert_eq!(back, m);
    }
}
