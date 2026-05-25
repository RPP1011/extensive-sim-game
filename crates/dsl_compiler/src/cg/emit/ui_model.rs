//! Plan A (absorbs playable-VS Plan 5) — lower a parsed `ui {}` block to a
//! JSON descriptor string matching `engine_ui::UiModel`'s serde shape
//! (externally-tagged enums, serde default):
//!
//! ```json
//! {"hud":[{"Bar":{"label":"HP","value":"hp","max":"hp_max","color":[220,40,40]}},
//!         {"Text":{"template":"Lv {level} Kills {kills}"}}],
//!  "screens":[{"name":"level_up","screen":{"Menu":{"title":"Level Up!",
//!         "cards":[{"label":"Bolt Damage +","action":{"Increment":"bolt_level"}}]}}},
//!            {"name":"dead","screen":{"End":{"title":"You Died",
//!         "summary":[["time","time"],["level","level"]],"restart_label":"Restart (R)"}}}]}
//! ```

use dsl_ast::ast::{UiDecl, UiScreen, UiWidget};

use super::json::json_escape;

fn color_json(c: &[u8; 3]) -> String {
    format!("[{},{},{}]", c[0], c[1], c[2])
}

/// Serialize a `UiDecl` into the `UiModel` JSON string.
pub fn ui_decl_to_json(d: &UiDecl) -> String {
    let mut hud = String::from("[");
    for (i, w) in d.hud.iter().enumerate() {
        if i > 0 {
            hud.push(',');
        }
        match w {
            UiWidget::Bar { label, value, max, color } => {
                hud.push_str(&format!(
                    "{{\"Bar\":{{\"label\":\"{label}\",\"value\":\"{value}\",\"max\":\"{max}\",\"color\":{color}}}}}",
                    label = json_escape(label),
                    value = json_escape(value),
                    max = json_escape(max),
                    color = color_json(color),
                ));
            }
            UiWidget::Text { template } => {
                hud.push_str(&format!(
                    "{{\"Text\":{{\"template\":\"{template}\"}}}}",
                    template = json_escape(template),
                ));
            }
        }
    }
    hud.push(']');

    let mut screens = String::from("[");
    for (i, s) in d.screens.iter().enumerate() {
        if i > 0 {
            screens.push(',');
        }
        match s {
            UiScreen::Menu { name, title, cards } => {
                let mut cards_json = String::from("[");
                for (j, card) in cards.iter().enumerate() {
                    if j > 0 {
                        cards_json.push(',');
                    }
                    cards_json.push_str(&format!(
                        "{{\"label\":\"{label}\",\"action\":{{\"Increment\":\"{field}\"}}}}",
                        label = json_escape(&card.label),
                        field = json_escape(&card.action_field),
                    ));
                }
                cards_json.push(']');
                screens.push_str(&format!(
                    "{{\"name\":\"{name}\",\"screen\":{{\"Menu\":{{\"title\":\"{title}\",\"cards\":{cards_json}}}}}}}",
                    name = json_escape(name),
                    title = json_escape(title),
                ));
            }
            UiScreen::End { name, title, summary, restart_label } => {
                let mut summary_json = String::from("[");
                for (j, (label, key)) in summary.iter().enumerate() {
                    if j > 0 {
                        summary_json.push(',');
                    }
                    summary_json.push_str(&format!(
                        "[\"{label}\",\"{key}\"]",
                        label = json_escape(label),
                        key = json_escape(key),
                    ));
                }
                summary_json.push(']');
                screens.push_str(&format!(
                    "{{\"name\":\"{name}\",\"screen\":{{\"End\":{{\"title\":\"{title}\",\"summary\":{summary_json},\"restart_label\":\"{restart}\"}}}}}}",
                    name = json_escape(name),
                    title = json_escape(title),
                    restart = json_escape(restart_label),
                ));
            }
        }
    }
    screens.push(']');

    format!("{{\"hud\":{hud},\"screens\":{screens}}}")
}

/// The empty `ui {}` default — a `UiModel` with no widgets or screens.
pub fn empty_ui_json() -> String {
    "{\"hud\":[],\"screens\":[]}".to_string()
}
