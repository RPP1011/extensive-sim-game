use dsl_compiler::parse;

#[test]
fn parses_minimum_terrain_block() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 0.5
  seed_purpose: 0xBA5E_7E55
}
"#;
    let program = parse(src).expect("parse");
    let t = program.terrain.expect("terrain block");
    assert_eq!(t.extent, 64);
    assert!((t.cell_size - 0.5).abs() < 1e-6);
    assert_eq!(t.seed_purpose, 0xBA5E_7E55);
}

#[test]
fn rejects_zero_seed_purpose() {
    let src = r#"
terrain {
  extent: 64
  cell_size: 0.5
  seed_purpose: 0
}
"#;
    let err = parse(src).err().expect("must fail");
    let msg = format!("{err}");
    assert!(msg.contains("seed_purpose") && msg.contains("non-zero"),
            "expected seed_purpose non-zero error, got: {msg}");
}
