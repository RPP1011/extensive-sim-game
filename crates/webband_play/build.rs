//! Give the `webband_play` binary a big MAIN-THREAD stack.
//!
//! The same finding S8 wrote up for `engine_play`'s `play`, copied verbatim
//! for the same reason: `RUST_MIN_STACK` only sizes threads Rust *spawns*, so
//! a binary's main-thread stack is whatever the linker reserved (2 MiB on
//! Windows) — and `webband_colony`'s `GeneratedRuntime::try_new` builds ~120
//! kernels' worth of descriptors as stack locals and blows straight through
//! it before the window ever opens. `crates/sims`' tests never see this
//! because they run their bodies on an explicit 64 MiB thread; a binary has
//! no such seam before `main`.
//!
//! `rustc-link-arg-bin` touches the final link of ONE binary, so nothing in
//! the dependency tree is rebuilt or reflagged (a `RUSTFLAGS` stack setting
//! would rebuild the workspace).

const STACK_BYTES: usize = 64 * 1024 * 1024;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("windows") {
        // ELF/Mach-O main-thread stacks grow dynamically up to the rlimit.
        return;
    }
    if target.contains("msvc") {
        println!("cargo:rustc-link-arg-bin=webband_play=/STACK:{STACK_BYTES}");
    } else {
        println!("cargo:rustc-link-arg-bin=webband_play=-Wl,--stack,{STACK_BYTES}");
    }
}
