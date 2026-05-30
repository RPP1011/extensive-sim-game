//! Plan A — lower a parsed `controls {}` block to a JSON descriptor string
//! matching `engine_play_api::ControlsDescriptor`'s serde shape:
//!
//! ```json
//! {"bindings":[{"key":"w","field":"ctl.move_y","value":1.0,"mode":"Hold"}]}
//! ```
//!
//! `BindMode` is an externally-tagged fieldless enum → serde emits the bare
//! variant string (`"Hold"` / `"Press"`). The `field` is the
//! `"<block>.<field>"` form the `PlayableRuntime::set_input` contract keys on.

use dsl_ast::ast::ControlsDecl;

use super::json::{json_escape, json_f32};

/// Serialize a `ControlsDecl` into the `ControlsDescriptor` JSON string.
pub fn controls_decl_to_json(d: &ControlsDecl) -> String {
    let mut out = String::from("{\"bindings\":[");
    for (i, b) in d.bindings.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let mode = if b.press { "Press" } else { "Hold" };
        out.push_str(&format!(
            "{{\"key\":\"{key}\",\"field\":\"{block}.{field}\",\"value\":{value},\"mode\":\"{mode}\"}}",
            key = json_escape(&b.key),
            block = json_escape(&b.block),
            field = json_escape(&b.field),
            value = json_f32(b.value as f32),
        ));
    }
    out.push_str("]}");
    out
}

/// The empty `controls {}` default — a `ControlsDescriptor` with no bindings.
pub fn empty_controls_json() -> String {
    "{\"bindings\":[]}".to_string()
}
