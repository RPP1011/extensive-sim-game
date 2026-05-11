//! Replaces the previous per-fixture build pipeline with the shared
//! `dsl_compiler::build_helper::emit()` helper (Plan E-A1). The helper
//! does parse → resolve → CG lower → schedule → WGSL/Rust emit →
//! `OUT_DIR/generated.rs` concatenation.
fn main() {
    dsl_compiler::build_helper::emit("predator_prey_real");
}
