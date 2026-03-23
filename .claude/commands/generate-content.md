Generate game content (abilities, classes) using the local LLM via Ollama and report results.

## Prerequisites
- Ollama must be running: `ollama serve` (or check with `curl -s http://localhost:11434/api/tags`)
- Model must be pulled: `ollama pull qwen35-9b`
- For best results: `OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=true OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`

## Steps

1. **Check Ollama** — verify the server is running and the model is available:
   ```
   curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['name'] for m in d['models']])"
   ```
   If not running, start it: `OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=true ollama serve &`

2. **Generate content** — use the generation script or direct API calls.

   For batch ability generation from a campaign trace:
   ```
   uv run --with httpx python3 << 'PYEOF'
   import httpx, json, time, re, concurrent.futures

   SPEC = '''Output ONLY one ability block in this exact format, nothing else:

   ability shield_bash {
       type: active
       cooldown: 10s
       effect: stun 2s + damage 30
       tag: melee
       description: "A devastating shield strike"
   }

   Valid tags: ranged, nature, stealth, tracking, survival, melee, defense, leadership, fortification, honor, arcane, elemental, ritual, knowledge, enchantment, healing, divine, protection, purification, restoration, assassination, agility, deception, sabotage
   Valid stats: attack, defense, speed, max_hp, ability_power

   Generate ONE ability for: '''

   # Build context from campaign trace or arguments
   # ... (see scripts/generate_content.py for full context assembly)
   PYEOF
   ```

   Or use the generation script directly:
   ```
   uv run --with transformers --with torch --with accelerate \
     python3 scripts/generate_content.py \
     --type ability \
     --context "level 10 ranger, 15 solo dungeons" \
     --tags "stealth, tracking" \
     --count 5
   ```

3. **For full campaign content generation** — load a trace file and generate for all trigger points:
   - Load trace from `generated/traces/campaign_*.trace.json`
   - Identify trigger points (level milestones, quest completions, fame thresholds)
   - Build rich context prompts using adventurer stats, guild state, world state, backstory
   - Generate abilities/classes via Ollama in parallel (4 workers)
   - Validate output format (must have `ability name { ... }` or `class Name { ... }`)
   - Report: total generated, valid %, time per item, examples

4. **Validate DSL** — for any generated class, run it through the parser:
   ```
   cargo test -p bevy_game --lib "class_dsl::tests" -- --nocapture
   ```

5. **Summary** — report:
   - Items generated (abilities, classes, quest hooks)
   - Format validity rate
   - Average generation time per item
   - Examples of the best/most interesting generated content
   - Any failures or format issues

## Content quality checklist
- [ ] Uses only valid stats (attack, defense, speed, max_hp, ability_power)
- [ ] Uses only valid tags from the tag list
- [ ] Ability names are snake_case
- [ ] Class names are PascalCase
- [ ] Descriptions are contextually appropriate to the adventurer/situation
- [ ] Scaling sources match the class's intended role
- [ ] No absurd stat growth (total per level ≤ 50)
- [ ] No duplicate abilities for the same adventurer

## Performance targets
- Ollama Q4 9B: ~0.8 items/s with 4 parallel workers
- Format validity: >90%
- Rich context prompt: full adventurer + guild + world + backstory
