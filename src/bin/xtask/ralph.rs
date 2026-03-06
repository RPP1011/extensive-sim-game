use std::fs;
use std::process::Command as ProcessCommand;
use std::process::ExitCode;

use super::cli::RalphStatusArgs;

pub fn run_ralph_status(args: RalphStatusArgs) -> ExitCode {
    // 1. Read and parse the PRD JSON.
    let prd_raw = match fs::read_to_string(&args.prd) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to read PRD {}: {err}", args.prd.display());
            return ExitCode::from(2);
        }
    };

    let mut prd: serde_json::Value = match serde_json::from_str(&prd_raw) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("Failed to parse PRD {}: {err}", args.prd.display());
            return ExitCode::from(2);
        }
    };

    // 2. Extract stories array.
    let stories = match prd.get("stories").and_then(|s| s.as_array()) {
        Some(s) => s.clone(),
        None => {
            eprintln!("PRD has no 'stories' array.");
            return ExitCode::from(2);
        }
    };

    // 3. Print table header.
    println!();
    println!(
        "{:<8}  {:<52}  {:<11}  {}",
        "ID", "Title", "Status", "Acceptance Criteria (first)"
    );
    println!("{}", "-".repeat(130));

    for story in &stories {
        let id = story
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let title = story
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("(untitled)");
        let status = story
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        // Summarise acceptance criteria: use the first criterion, truncated.
        let ac_summary = story
            .get("acceptanceCriteria")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("(none)");
        let ac_truncated = if ac_summary.len() > 55 {
            format!("{}...", &ac_summary[..52])
        } else {
            ac_summary.to_string()
        };

        // Colour-code status for clarity (ANSI, works in most terminals).
        let status_display = match status {
            "done" => format!("\x1b[32m{:<11}\x1b[0m", status),
            "in-progress" => format!("\x1b[33m{:<11}\x1b[0m", status),
            _ => format!("{:<11}", status),
        };

        println!(
            "{:<8}  {:<52}  {}  {}",
            id,
            if title.len() > 52 {
                format!("{}...", &title[..49])
            } else {
                title.to_string()
            },
            status_display,
            ac_truncated
        );
    }
    println!();

    // 4. Run cargo test and report result.
    println!("Running `cargo test`...");
    let test_result = ProcessCommand::new("cargo")
        .arg("test")
        .status();

    let tests_passed = match test_result {
        Ok(status) => {
            if status.success() {
                println!("\x1b[32mcargo test: PASSED\x1b[0m");
                true
            } else {
                println!("\x1b[31mcargo test: FAILED (exit code {})\x1b[0m",
                    status.code().unwrap_or(-1));
                false
            }
        }
        Err(err) => {
            eprintln!("Failed to run cargo test: {err}");
            return ExitCode::from(1);
        }
    };

    // 5. Optionally update in-progress stories to done when tests pass.
    if args.update {
        if !tests_passed {
            println!("Skipping --update: tests did not pass.");
            return ExitCode::from(1);
        }

        // Pre-compute top-level quality gate check before taking a mutable
        // borrow on `prd` to avoid a simultaneous borrow conflict.
        let toplevel_gates_has_cargo_test: bool = prd
            .get("qualityGates")
            .and_then(|v| v.as_array())
            .map(|gates| {
                gates
                    .iter()
                    .any(|g| g.as_str().map(|s| s.contains("cargo test")).unwrap_or(false))
            })
            .unwrap_or(false);

        let stories_arr = match prd.get_mut("stories").and_then(|s| s.as_array_mut()) {
            Some(s) => s,
            None => {
                eprintln!("Cannot mutate PRD stories array.");
                return ExitCode::from(1);
            }
        };

        let mut updated_count = 0usize;
        for story in stories_arr.iter_mut() {
            let status = story
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Only promote in-progress stories; never regress done stories.
            if status == "in-progress" {
                // Check quality gates: we treat any story with "cargo test" in
                // its quality_gates (or the top-level qualityGates) as
                // satisfied when cargo test passes.
                let gates_satisfied = {
                    let story_gates = story
                        .get("quality_gates")
                        .or_else(|| story.get("qualityGates"))
                        .and_then(|v| v.as_array());

                    match story_gates {
                        Some(gates) => gates.iter().any(|g| {
                            g.as_str()
                                .map(|s| s.contains("cargo test"))
                                .unwrap_or(false)
                        }),
                        // Fall back to pre-computed top-level qualityGates check.
                        None => toplevel_gates_has_cargo_test,
                    }
                };

                if gates_satisfied {
                    let id = story
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();
                    if let Some(obj) = story.as_object_mut() {
                        obj.insert("status".to_string(), serde_json::Value::String("done".to_string()));
                    }
                    println!("  Marked {id} as done.");
                    updated_count += 1;
                }
            }
        }

        if updated_count > 0 {
            // Write updated PRD back to disk.
            let updated = match serde_json::to_string_pretty(&prd) {
                Ok(v) => v,
                Err(err) => {
                    eprintln!("Failed to serialize updated PRD: {err}");
                    return ExitCode::from(1);
                }
            };
            if let Err(err) = fs::write(&args.prd, format!("{updated}\n")) {
                eprintln!("Failed to write updated PRD {}: {err}", args.prd.display());
                return ExitCode::from(1);
            }
            println!("Updated {} story/stories in {}.", updated_count, args.prd.display());
        } else {
            println!("No in-progress stories with satisfied quality gates found; nothing updated.");
        }
    }

    ExitCode::SUCCESS
}
