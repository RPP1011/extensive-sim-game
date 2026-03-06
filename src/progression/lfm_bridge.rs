use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use super::types::LfmProgressionResult;

/// Spawn an LFM subprocess to generate a narrative progression reward.
///
/// Returns a shared handle that will be populated when the subprocess completes.
/// Follows the same `Arc<Mutex<Option<T>>>` pattern as `BackstoryNarrativeGenState`.
pub fn spawn_lfm_progression_request(
    journal_json: String,
    hero_id: u32,
) -> Arc<Mutex<Option<LfmProgressionResult>>> {
    let shared = Arc::new(Mutex::new(None));
    let shared_clone = Arc::clone(&shared);

    std::thread::spawn(move || {
        let result = run_lfm_progression(&journal_json, hero_id);
        if let Ok(mut slot) = shared_clone.lock() {
            *slot = Some(result);
        }
    });

    shared
}

fn run_lfm_progression(journal_json: &str, hero_id: u32) -> LfmProgressionResult {
    let child = Command::new("python")
        .args(["-m", "lfm_agent.progression"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .current_dir(
            std::env::var("LFM_AGENT_DIR")
                .unwrap_or_else(|_| {
                    // Default: sibling directory of game project
                    let mut path = std::env::current_dir().unwrap_or_default();
                    path.pop();
                    path.push("lfm-agent");
                    path.to_string_lossy().to_string()
                }),
        )
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            return LfmProgressionResult {
                hero_id,
                reward_json: String::new(),
                narrative_text: String::new(),
                success: false,
                error: Some(format!("Failed to spawn LFM subprocess: {e}")),
            };
        }
    };

    if let Some(ref mut stdin) = child.stdin {
        let _ = stdin.write_all(journal_json.as_bytes());
    }
    // Close stdin so the subprocess knows input is complete.
    drop(child.stdin.take());

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => {
            return LfmProgressionResult {
                hero_id,
                reward_json: String::new(),
                narrative_text: String::new(),
                success: false,
                error: Some(format!("LFM subprocess wait failed: {e}")),
            };
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return LfmProgressionResult {
            hero_id,
            reward_json: String::new(),
            narrative_text: String::new(),
            success: false,
            error: Some(format!(
                "LFM subprocess exited with {}: {}",
                output.status,
                stderr.chars().take(500).collect::<String>()
            )),
        };
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    // Parse the JSON output to extract narrative_text and the reward.
    match serde_json::from_str::<serde_json::Value>(&stdout) {
        Ok(val) => {
            let narrative_text = val
                .get("narrative_text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            LfmProgressionResult {
                hero_id,
                reward_json: stdout,
                narrative_text,
                success: true,
                error: None,
            }
        }
        Err(e) => LfmProgressionResult {
            hero_id,
            reward_json: stdout,
            narrative_text: String::new(),
            success: false,
            error: Some(format!("Failed to parse LFM JSON output: {e}")),
        },
    }
}
