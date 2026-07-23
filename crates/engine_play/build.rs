//! Give the `play` binary a big MAIN-THREAD stack.
//!
//! Why this exists: `RUST_MIN_STACK` only sizes threads Rust *spawns* — the
//! process main thread's stack is fixed by the linker at build time (2 MiB by
//! default on Windows). Constructing a compiled-`.sim` `GeneratedRuntime` blows
//! straight through that: `webband_colony`'s `try_new` builds ~120 kernels and
//! several hundred wgpu buffer/bind-group descriptors as stack locals, and
//! `play webband_colony` died with `thread 'main' has overflowed its stack`
//! before the window ever opened. (The `crates/sims` tests do not hit this only
//! because they run their bodies on an explicitly-spawned 64 MiB thread; a
//! binary has no such seam before `main`.)
//!
//! A per-binary link arg is the narrow fix: `rustc-link-arg-bin` applies to the
//! final link of ONE binary, so nothing in the dependency tree is rebuilt or
//! reflagged (setting `RUSTFLAGS` instead would rebuild the whole workspace).
//! 64 MiB matches the `RUST_MIN_STACK` the test suites use.

const STACK_BYTES: usize = 64 * 1024 * 1024;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("windows") {
        // ELF/Mach-O main-thread stacks grow dynamically up to the rlimit; no
        // link-time reservation is needed (or expressible the same way).
        return;
    }
    if target.contains("msvc") {
        println!("cargo:rustc-link-arg-bin=play=/STACK:{STACK_BYTES}");
    } else {
        // windows-gnu: ld's --stack takes the reserve size.
        println!("cargo:rustc-link-arg-bin=play=-Wl,--stack,{STACK_BYTES}");
    }
}
