Run BFS state-space exploration on the headless campaign simulator and report results.

## Steps

1. **Build** — run `cargo build 2>&1 | grep "^error"` to ensure the project compiles. Stop on errors.

2. **Run BFS exploration** — run the BFS explorer to completion:
   ```
   cargo run --bin xtask -- bfs-explore \
     --max-waves 0 \
     --clusters 50 \
     --initial-roots 200 \
     --ticks-per-branch 200 \
     --trajectory-ticks 15000 \
     --root-interval 200 \
     --output generated/bfs_explore.jsonl
   ```
   If `--config <path>` is provided as an argument to this skill, pass it through.

3. **Analyze results** — after BFS completes, analyze the output:
   ```python
   import json
   from collections import Counter
   with open('generated/bfs_explore.jsonl') as f:
       samples = [json.loads(line) for line in f]
   # Report: total samples, action type distribution, value ranges,
   # terminal outcomes, waves completed
   ```

4. **Summary** — report:
   - Total samples generated
   - Waves completed
   - Action types explored (and any missing)
   - Terminal outcome distribution (victories vs defeats)
   - Mean leaf value by action type (which actions lead to best outcomes)
   - Output file size
   - Time taken

5. **Validate** — run `cargo test -p bevy_game --lib "headless" -- --test-threads=1 2>&1 | tail -3` to ensure no regressions.

## Notes
- BFS runs to campaign completion (all leaves terminal) unless `--max-waves` is set
- Each wave expands all roots in parallel via rayon
- Leaves are clustered and median states become next wave's roots
- The output JSONL has one sample per line with root_tokens, action_type, leaf_tokens, leaf_value
