Run the full game test suite and report results.

Steps to follow:

1. **Cargo unit tests** — run `cargo test 2>&1` and check for failures. If any fail, show which test failed and the error message.

2. **Scenario runner** — run `cargo run --bin xtask -- scenario run scenarios/` to execute all scenario TOML files. Parse the output: list which scenarios passed (✓) and which failed (✗) with the assertion that failed.

3. **Headless sim smoke test** — run `cargo run -- --headless --steps 300 2>&1 | tail -20` and check for panics or unexpected output.

4. **Summary** — after all three steps, produce a concise summary:
   - Total tests: N passed, M failed
   - Which scenarios failed and why (wrong outcome, too slow, etc.)
   - Any panics or compile errors found
   - Suggest which source file to look at for each failure

5. **On failure** — if anything failed:
   - Read the relevant source file to understand the system
   - Identify the likely root cause
   - Propose a specific fix (but do not apply it unless asked)

Do not skip steps. If a command fails to compile, stop and report the compile error.
