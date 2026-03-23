#!/usr/bin/env python3
"""Generate ability/class DSL from sweep contexts via Ollama.

Reads sweep JSONL, deduplicates by coarse key, generates content in parallel,
writes to content store JSONL. Supports --resume for interrupted runs.

Usage:
    uv run --with httpx --with pandas python3 scripts/ollama_generate_from_sweep.py \
        --contexts generated/vae_v7/vae_contexts.jsonl \
        --output generated/ollama_content_store.jsonl \
        --workers 4
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

import httpx

# ---------------------------------------------------------------------------
# Prompts (same as generate_content.py / llm.rs)
# ---------------------------------------------------------------------------

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


def strip_think(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def extract_ability(text):
    m = re.search(r'ability\s+\w+\s*\{[^}]+\}', text, re.DOTALL)
    return m.group(0) if m else None


def extract_class(text):
    m = re.search(r'class\s+\w+\s*\{.*?\n\}', text, re.DOTALL)
    return m.group(0) if m else None


def score_block(text, gen_type):
    score = 0.0
    if gen_type == "ability":
        if re.search(r'type:\s*(active|passive)', text): score += 1
        if re.search(r'type:\s*active', text) and 'cooldown:' in text: score += 1
        if re.search(r'type:\s*passive', text) and 'trigger:' in text: score += 1
        tag_m = re.search(r'tag:\s*(\S+)', text)
        if tag_m and tag_m.group(1).strip().rstrip(',') in VALID_TAGS: score += 3
        if 'effect:' in text: score += 1
        if re.search(r'description:\s*"[^"]+"', text): score += 1
        name_m = re.search(r'ability\s+(\w+)', text)
        if name_m and name_m.group(1) == name_m.group(1).lower() and '_' in name_m.group(1): score += 2
    else:
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


def generate_one(client, base_url, model, ctx, temperature=0.7):
    gen_type = ctx["content_type"]
    context_text = ctx.get("context_text", "")
    spec = ABILITY_SPEC if gen_type == "ability" else CLASS_SPEC
    max_tokens = 300 if gen_type == "ability" else 500
    extractor = extract_ability if gen_type == "ability" else extract_class

    # For classes, use multi-turn
    if gen_type == "class":
        words = [w.capitalize() for w in context_text.split()[:5]
                 if w.isalpha() and w.lower() not in ("a", "an", "the", "who", "that", "and", "or", "with")]
        seed_name = "".join(words[:3]) or "Custom"
        messages = [
            {"role": "user", "content": f"Generate a class definition for: {context_text[:200]}. Use the exact DSL format I'll show you."},
            {"role": "assistant", "content": f"class {seed_name} {{"},
            {"role": "user", "content": spec + context_text},
        ]
    else:
        messages = [{"role": "user", "content": spec + context_text}]

    t0 = time.time()
    try:
        resp = client.post(f"{base_url}/api/chat", json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=90)
        raw = strip_think(resp.json()["message"]["content"])
        elapsed = time.time() - t0

        block = extractor(raw)
        if block:
            s = score_block(block, gen_type)
            return {"valid": True, "selected": block, "raw": raw, "score": s, "time_s": elapsed}
        return {"valid": False, "selected": None, "raw": raw, "score": 0, "time_s": elapsed}
    except Exception as e:
        return {"valid": False, "selected": None, "raw": str(e), "score": 0, "time_s": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="generated/vae_v7/vae_contexts.jsonl")
    parser.add_argument("--output", "-o", default="generated/ollama_content_store.jsonl")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--workers", "-j", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Check server
    try:
        httpx.get(f"{args.url}/api/tags", timeout=3)
    except Exception:
        print(f"Ollama not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)

    # Load and deduplicate
    print(f"Loading {args.contexts}...", file=sys.stderr)
    contexts = []
    with open(args.contexts) as f:
        for line in f:
            try:
                contexts.append(json.loads(line))
            except:
                continue

    llm = [c for c in contexts if c.get("content_type") in ("ability", "class")]
    seen = set()
    unique = []
    for ctx in llm:
        k = coarse_key(ctx)
        if k not in seen:
            seen.add(k)
            unique.append(ctx)

    # Resume
    existing = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("selected"):
                        existing.add(r.get("coarse_key", ""))
                except:
                    continue
        unique = [c for c in unique if coarse_key(c) not in existing]
        print(f"Resume: {len(existing)} done, {len(unique)} remaining", file=sys.stderr)

    if args.limit > 0:
        unique = unique[:args.limit]

    print(f"{len(unique)} unique keys, {args.workers} workers, model={args.model}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = open(args.output, "a")
    client = httpx.Client()

    completed = 0
    valid_count = 0
    start = time.time()

    def do_one(ctx):
        return ctx, generate_one(client, args.url, args.model, ctx, args.temperature)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_one, c): c for c in unique}
        for f in concurrent.futures.as_completed(futures):
            ctx, result = f.result()
            completed += 1
            key = coarse_key(ctx)

            record = {
                "coarse_key": key,
                "gen_type": ctx["content_type"],
                "selected": result["selected"],
                "score": result["score"],
                "valid": result["valid"],
                "time_s": result["time_s"],
                "archetype": ctx.get("archetype", ""),
                "level": ctx.get("level", 0),
                "trigger": ctx.get("trigger", ""),
                "raw": result["raw"][:500],
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

            if result["valid"]:
                valid_count += 1

            elapsed = time.time() - start
            if completed % 10 == 0 or completed == len(unique):
                rate = completed / elapsed
                eta = (len(unique) - completed) / rate / 60 if rate > 0 else 0
                print(
                    f"[{completed}/{len(unique)}] {rate:.1f}/s "
                    f"valid={valid_count} ({valid_count/completed*100:.0f}%) "
                    f"ETA={eta:.0f}min",
                    file=sys.stderr,
                )

    out.close()
    elapsed = time.time() - start
    print(f"\nDone: {completed} generated, {valid_count} valid ({valid_count/completed*100:.0f}%) in {elapsed/60:.1f}min", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
