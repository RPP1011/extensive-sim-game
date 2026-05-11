//! Replaces the previous ~110-line per-fixture build pipeline with the
//! shared `dsl_compiler::build_helper::emit()` (Plan E-A1). The helper
//! does parse → resolve → CG lower → schedule → WGSL/Rust emit →
//! `OUT_DIR/generated.rs` concatenation, including the `cargo:warning`
//! emit-stats lines this fixture used to print by hand.
fn main() {
    dsl_compiler::build_helper::emit("dodger_probe");
}
