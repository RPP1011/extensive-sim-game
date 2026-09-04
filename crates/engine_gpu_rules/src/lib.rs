// Placeholder lib.rs. The DSL inputs this crate's kernel modules were once
// emitted from no longer exist, and there is currently no live `compile-dsl
// --cg-canonical` command to regenerate them (xtask itself is gone). The
// crate keeps this empty placeholder so the workspace continues to build;
// restoring that emission path would write `pub mod ...` / `pub use ...`
// declarations back into this file.
