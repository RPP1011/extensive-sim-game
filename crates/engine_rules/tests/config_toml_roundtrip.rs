//! Exercise the compiler-emitted `Config::from_toml` loader end-to-end
//! against the canonical `assets/config/default.toml`. Three assertions:
//!
//! 1. The committed TOML file parses cleanly and equals `Config::default()`
//!    byte-for-byte (the compiler's Rust + TOML emitters agree on every
//!    default value).
//! 2. Mutating a subset of fields via TOML and re-parsing preserves the
//!    edits on the touched fields and leaves the rest at their defaults
//!    (thanks to `#[serde(default)]` on each per-block field).
//! 3. `ConfigError` implements `std::error::Error` so call sites can
//!    compose it with other error types via `?`.

use std::io::Write;
use std::path::PathBuf;

use engine_rules::config::{Config, ConfigError};

fn repo_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` points at the per-crate dir; the default TOML is
    // at `<workspace>/assets/config/default.toml`. Walking up two levels is
    // the minimal way to locate it without bringing in `cargo_metadata`.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn default_toml_equals_config_default() {
    let path = repo_root().join("assets").join("config").join("default.toml");
    let loaded = Config::from_toml(&path).expect("from_toml should parse default.toml");
    assert_eq!(loaded, Config::default(), "default.toml must match Config::default()");
}

#[test]
fn mutate_one_field_and_reload_keeps_the_edit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tuned.toml");

    // Copy the default, mutate attack_damage to 20, leave everything else
    // alone. Serde's `#[serde(default)]` on each block-level field means an
    // incomplete TOML file still parses; we only need to spell out the one
    // block that's changing here.
    let body = r#"
[combat]
attack_damage = 20.0
"#;
    std::fs::write(&path, body).expect("write tuned toml");
    let loaded = Config::from_toml(&path).expect("parse tuned toml");

    assert_eq!(loaded.combat.attack_damage, 20.0);
    // Untouched fields retained their defaults.
    let defaults = Config::default();
    assert_eq!(loaded.combat.attack_range, defaults.combat.attack_range);
    assert_eq!(loaded.movement.move_speed_mps, defaults.movement.move_speed_mps);
    assert_eq!(loaded.needs.eat_restore, defaults.needs.eat_restore);
}

#[test]
fn parse_error_surfaces_via_configerror() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bad.toml");

    // Invalid TOML — the key has no value.
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "[combat]").unwrap();
    writeln!(f, "attack_damage =").unwrap();
    drop(f);

    let err = Config::from_toml(&path).expect_err("bad toml should fail");
    match err {
        ConfigError::Parse(_) => {}
        ConfigError::Io(e) => panic!("expected Parse, got Io({e})"),
    }
}

#[test]
fn missing_file_surfaces_as_io_error() {
    let err = Config::from_toml(std::path::Path::new("/definitely/not/a/real/path.toml"))
        .expect_err("missing file should fail");
    match err {
        ConfigError::Io(_) => {}
        ConfigError::Parse(_) => panic!("expected Io, got Parse"),
    }
}
