Run HvH combat simulations on generated room layouts and report win rate results.

## Usage

```
/simulate-rooms [max_matches] [heroes_dir] [rooms_file]
```

All arguments are optional with sensible defaults:
- `max_matches` — number of matches to run (default: 1000)
- `heroes_dir` — path to hero template directory (default: `assets/hero_templates`)
- `rooms_file` — path to rooms JSONL (default: `generated/rooms.jsonl`)

## Steps

1. **Check prerequisites** — verify `generated/rooms.jsonl` exists (or the specified rooms file). If not, generate it:
   ```bash
   cargo run --release --bin xtask -- roomgen export --count-per-type 5000 --output generated/rooms.jsonl
   ```

2. **Build release binary** if needed:
   ```bash
   cargo build --release --bin xtask
   ```

3. **Run simulations** — execute the simulate command with the specified parameters:
   ```bash
   ./target/release/xtask roomgen simulate <rooms_file> <heroes_dir> --max-matches <N> --output generated/room_sim_results.jsonl
   ```

4. **Parse results** — read the output JSONL and compute:
   - Overall win/loss/timeout rates
   - Win rates broken down by room type
   - Win rates broken down by hero (which heroes appear most in winning teams)
   - Average match duration (ticks) by room type
   - Rooms with extreme outcomes (always win or always lose across all matches)

5. **Report** — present a formatted summary table with:
   - Per-room-type breakdown: win%, loss%, timeout%, avg ticks
   - Top 5 heroes by win contribution
   - Bottom 5 heroes by win contribution
   - Any rooms that seem degenerate (100% one outcome)

## Notes

- Each match is a random 4v4 HvH with heroes sampled from the template directory
- Hero team is always Team A; enemy team is always Team B; "Victory" means Team A won
- The sim uses GOAP/squad AI with inferred personalities
- Matches time out at 5000 ticks (~8.3 minutes of game time at 100ms/tick)
- Results are written to `generated/room_sim_results.jsonl` as NDJSON
- Use `--max-matches` to control how long the run takes (~0.4s/match)
