#!/usr/bin/env python3
"""Generate ability/class DSL content using Gemini API from sweep contexts.

Reads coarse-deduped contexts from a VAE sweep, calls Gemini to generate
DSL content, scores/validates, and writes results to a content store JSONL.

Usage:
    # Set API key
    export GEMINI_API_KEY=your_key_here

    # Generate from sweep contexts
    uv run --with google-genai --with pandas python3 scripts/gemini_generate.py \
        --contexts generated/vae_v6/vae_contexts.jsonl \
        --output generated/gemini_content_store.jsonl \
        --workers 8 \
        --limit 100  # for testing
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time

ABILITY_SPEC = '''Output ONLY one ability block in this exact format, nothing else.

ability shield_bash {
    type: active
    cooldown: 10s
    effect: stun 2s + damage 30
    tag: melee
    description: "A devastating shield strike"
}

ability battle_sense {
    type: passive
    trigger: on_damage_taken
    effect: buff defense 15% 5s
    tag: defense
    description: "Pain sharpens focus"
}

Valid tags: ranged, nature, stealth, tracking, survival, melee, defense, leadership, fortification, honor, arcane, elemental, ritual, knowledge, enchantment, healing, divine, protection, purification, restoration, assassination, agility, deception, sabotage
Valid stats: attack, defense, speed, max_hp, ability_power
Valid triggers (passive only): on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged, on_death, on_ability_used, on_hp_below, on_hp_above, on_shield_broken, periodic
Valid effects: damage <N>, heal <N>, shield <N>, stun <N>s, slow <factor> <N>s, knockback <N>, dash, buff <stat> <N>% <N>s, debuff <stat> <N>% <N>s, stealth <N>s, evasion <N>%, tenacity <N>, teleport, aura <stat> +<N>

Generate ONE ability for: '''

CLASS_SPEC = '''Output ONLY one class definition, nothing else.

class Ranger {
    stat_growth: +2 attack, +1 defense, +3 speed, +1 ability_power per level
    tags: ranged, nature, stealth
    scaling party_alive_count {
        when party_members >= 2: +10% speed
        always: aura morale +1
    }
    abilities {
        level 1: keen_eye "Improved scouting range"
        level 5: multishot "Hit multiple targets"
        level 10: camouflage "Escape losing battles"
        level 20: deadeye "Double critical hit chance"
    }
    requirements: level 1
}

Rules:
- stat_growth uses ONLY: attack, defense, speed, max_hp, ability_power (total per level <= 10)
- tags from: ranged, nature, stealth, melee, defense, leadership, arcane, elemental, healing, divine, assassination, agility, deception, sabotage
- scaling sources: party_alive_count, faction_allied_count, crisis_active, threat_level, fame
- requirements: level <N> or quest_count <N> or fame <N>
- ability names must be snake_case, 3-5 abilities spread across levels 1-40

Generate ONE class for: '''

VALID_TAGS = {
    "ranged", "nature", "stealth", "tracking", "survival",
    "melee", "defense", "leadership", "fortification", "honor",
    "arcane", "elemental", "ritual", "knowledge", "enchantment",
    "healing", "divine", "protection", "purification", "restoration",
    "assassination", "agility", "deception", "sabotage",
    "crisis", "legendary", "inspiration", "sacrifice",
}


def extract_ability(text):
    m = re.search(r'ability\s+\w+\s*\{[^}]+\}', text, re.DOTALL)
    return m.group(0) if m else None


def extract_class(text):
    m = re.search(r'class\s+\w+\s*\{.*?\n\}', text, re.DOTALL)
    return m.group(0) if m else None


def score_ability(text):
    score = 0.0
    if re.search(r'type:\s*(active|passive)', text): score += 1
    if re.search(r'type:\s*active', text) and 'cooldown:' in text: score += 1
    if re.search(r'type:\s*passive', text) and 'trigger:' in text: score += 1
    tag_m = re.search(r'tag:\s*(\S+)', text)
    if tag_m and tag_m.group(1).strip().rstrip(',') in VALID_TAGS: score += 3
    if 'effect:' in text: score += 1
    if re.search(r'description:\s*"[^"]+"', text): score += 1
    name_m = re.search(r'ability\s+(\w+)', text)
    if name_m and name_m.group(1) == name_m.group(1).lower() and '_' in name_m.group(1): score += 2
    return score


def score_class(text):
    score = 0.0
    if 'stat_growth:' in text: score += 1
    if re.search(r'scaling\s+\w+\s*\{', text): score += 1
    levels = re.findall(r'level\s+\d+:', text)
    if 3 <= len(levels) <= 6: score += 2
    elif levels: score += 1
    if 'requirements:' in text: score += 1
    name_m = re.search(r'class\s+(\w+)', text)
    if name_m and name_m.group(1)[0].isupper() and '_' not in name_m.group(1): score += 2
    return score


def coarse_key(ctx):
    level_bucket = (ctx.get('level', 0) // 5) * 5
    return f"{ctx.get('archetype', '')}_{level_bucket}_{ctx.get('trigger', '')}_{ctx.get('content_type', '')}"


def hash_prompt(prompt):
    return int(hashlib.md5(prompt.encode()).hexdigest()[:16], 16)


def generate_one(client, model_name, context_text, content_type, temperature=0.7):
    """Call Gemini API for one generation."""
    spec = ABILITY_SPEC if content_type == "ability" else CLASS_SPEC
    prompt = spec + context_text
    extractor = extract_ability if content_type == "ability" else extract_class
    scorer = score_ability if content_type == "ability" else score_class

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": 500 if content_type == "class" else 300,
            },
        )
        raw = response.text or ""
        elapsed = time.time() - t0

        block = extractor(raw)
        if block:
            s = scorer(block)
            return {"valid": True, "text": block, "raw": raw, "score": s, "time_s": elapsed}
        else:
            return {"valid": False, "text": None, "raw": raw, "score": 0, "time_s": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        return {"valid": False, "text": None, "raw": str(e), "score": 0, "time_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Generate content via Gemini API")
    parser.add_argument("--contexts", default="generated/vae_v6/vae_contexts.jsonl")
    parser.add_argument("--output", "-o", default="generated/gemini_content_store.jsonl")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--workers", "-j", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0, help="Max keys to generate (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip keys already in output")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)

    # Load contexts and deduplicate by coarse key
    print(f"Loading contexts from {args.contexts}...", file=sys.stderr)
    contexts = []
    with open(args.contexts) as f:
        for line in f:
            try:
                contexts.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    llm_contexts = [c for c in contexts if c.get("content_type") in ("ability", "class")]

    seen_keys = set()
    unique = []
    for ctx in llm_contexts:
        key = coarse_key(ctx)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(ctx)

    # Resume support: skip keys already generated
    existing_keys = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("selected"):
                        existing_keys.add(r.get("coarse_key", ""))
                except json.JSONDecodeError:
                    continue
        unique = [c for c in unique if coarse_key(c) not in existing_keys]
        print(f"Resuming: {len(existing_keys)} already done, {len(unique)} remaining", file=sys.stderr)

    if args.limit > 0:
        unique = unique[:args.limit]

    print(f"Generating {len(unique)} unique keys ({args.workers} workers, model={args.model})", file=sys.stderr)

    # Open output file
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_file = open(args.output, "a")

    completed = 0
    valid_count = 0
    start = time.time()

    def process_one(ctx):
        return ctx, generate_one(client, args.model, ctx.get("context_text", ""), ctx["content_type"], args.temperature)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, ctx): ctx for ctx in unique}

        for future in concurrent.futures.as_completed(futures):
            ctx, result = future.result()
            completed += 1
            key = coarse_key(ctx)

            record = {
                "coarse_key": key,
                "prompt_hash": hash_prompt(ctx.get("context_text", "")),
                "gen_type": ctx["content_type"],
                "context": ctx.get("context_text", "")[:200],
                "selected": result["text"],
                "score": result["score"],
                "raw": result["raw"][:500],
                "valid": result["valid"],
                "time_s": result["time_s"],
                "archetype": ctx.get("archetype", ""),
                "level": ctx.get("level", 0),
                "trigger": ctx.get("trigger", ""),
            }

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()

            if result["valid"]:
                valid_count += 1
                name = result["text"].split()[1] if result["text"] else "?"
                status = f"OK {name}"
            else:
                status = "FAIL"

            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0

            if completed % 10 == 0 or completed == len(unique):
                print(
                    f"[{completed}/{len(unique)}] {rate:.1f}/s "
                    f"valid={valid_count} ({valid_count/completed*100:.0f}%) "
                    f"{ctx['content_type']:7s} {status}",
                    file=sys.stderr,
                )

    out_file.close()
    elapsed = time.time() - start

    print(f"\n=== Done in {elapsed:.1f}s ({elapsed/60:.1f}min) ===", file=sys.stderr)
    print(f"Generated: {completed}, Valid: {valid_count} ({valid_count/completed*100:.0f}%)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
