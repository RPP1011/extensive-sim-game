//! Asserts every emitted WGSL file under `engine_gpu_rules/src/` parses
//! cleanly as WGSL via naga's WGSL frontend. Catches missing-symbol
//! errors at test time instead of at first device-instantiation.
//!
//! `megakernel.wgsl` is excluded — it's intentional T14 scaffold owned
//! by the gpu_megakernel_plan follow-up.

use std::path::PathBuf;

#[test]
fn every_wgsl_parses() {
    let src_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "src"].iter().collect();
    let mut failures: Vec<(String, String)> = Vec::new();

    for entry in std::fs::read_dir(&src_dir).expect("read src dir") {
        let entry = entry.expect("entry");
        let path = entry.path();
        let Some(ext) = path.extension() else { continue };
        if ext != "wgsl" {
            continue;
        }
        let name = path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        if name == "megakernel.wgsl" {
            continue;
        }

        let src = std::fs::read_to_string(&path).expect("read wgsl");
        match naga::front::wgsl::parse_str(&src) {
            Ok(_) => {}
            Err(e) => failures.push((name, e.emit_to_string(&src))),
        }
    }

    if !failures.is_empty() {
        let mut msg = String::from("naga parse failures:\n");
        for (name, err) in &failures {
            msg.push_str(&format!("==== {name} ====\n{err}\n"));
        }
        panic!("{msg}");
    }
}
