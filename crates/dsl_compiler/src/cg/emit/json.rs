//! Plan A — tiny hand-rolled JSON helpers shared by the player-facing
//! descriptor emitters (`controls` / `render` / `ui_model`). The compiler
//! emits descriptor JSON as `&'static str` literals; these helpers keep the
//! output valid JSON that round-trips through the `engine_play_api` /
//! `engine_ui` serde `from_json` parsers.

/// Escape a string for embedding inside a JSON double-quoted literal.
/// Handles the JSON-mandatory escapes (`"`, `\`, control chars). The
/// descriptor labels/templates are author-controlled text, so this is the
/// minimal correct set.
pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Format an `f32` as a JSON number that round-trips through `serde_json`.
/// Whole numbers get a trailing `.0` so the literal is unambiguously a float
/// (and matches serde's own float rendering); non-finite values clamp to 0.0
/// (descriptors carry no NaN/Inf).
pub fn json_f32(v: f32) -> String {
    if !v.is_finite() {
        return "0.0".to_string();
    }
    // `{:?}` on f32 is round-trip-safe and always includes a decimal point
    // for whole values (e.g. `1.0`, not `1`). Mirrors the WGSL-emit float
    // policy used elsewhere in cg::emit.
    format!("{v:?}")
}
