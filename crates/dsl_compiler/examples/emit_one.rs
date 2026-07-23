//! Isolated full-emit runner for one fixture: parse -> resolve -> lower ->
//! WGSL/Rust emit via `build_helper::emit_namespaced`, without rebuilding
//! the whole `sims` mega-crate (~5s vs ~60-90s). Companion to `parse_one`
//! / `resolve_one`, which stop before the CG layer where most authoring
//! errors actually surface (EmitError::Unsupported etc.).
//!
//! Usage (from the workspace root; OUT_DIR must be set by the caller):
//!   OUT_DIR=$(mktemp -d) SIM_REQUIRE_ALL_RULES=1 \
//!     cargo run -p dsl_compiler --example emit_one <fixture>

fn main() {
    let name = std::env::args().nth(1).expect("usage: emit_one <fixture>");
    if std::env::var_os("OUT_DIR").is_none() {
        eprintln!("emit_one: set OUT_DIR (e.g. OUT_DIR=$(mktemp -d)) before running");
        std::process::exit(2);
    }
    dsl_compiler::build_helper::emit_namespaced(&name);
    println!("{name}: EMIT OK");
}
