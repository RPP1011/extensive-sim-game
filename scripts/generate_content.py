#!/usr/bin/env python3
"""Generate game content (abilities, classes) using Ollama.

Usage:
    # Single ability:
    uv run --with httpx python3 scripts/generate_content.py --type ability --context "level 10 stealth ranger"

    # Single class:
    uv run --with httpx python3 scripts/generate_content.py --type class --context "a battle medic who heals while fighting"

    # Batch from JSONL:
    uv run --with httpx python3 scripts/generate_content.py --batch requests.jsonl --output generated/content.jsonl

    # Generate N items with varied contexts:
    uv run --with httpx python3 scripts/generate_content.py --type ability --context "level 5 ranger" --count 5

    # Custom model/server:
    uv run --with httpx python3 scripts/generate_content.py --model qwen35-9b --url http://localhost:11434 ...
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
# Prompts — everything in user message for best instruction following
# ---------------------------------------------------------------------------

ABILITY_PROMPT = '''Output ONLY one ability block in this exact format, nothing else.

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

ability shadow_step {
    type: active
    cooldown: 15s
    effect: teleport + damage 40 + stealth 3s
    tag: stealth
    description: "Vanish and reappear behind your prey"
}

Valid tags: ranged, nature, stealth, tracking, survival, melee, defense, leadership, fortification, honor, arcane, elemental, ritual, knowledge, enchantment, healing, divine, protection, purification, restoration, assassination, agility, deception, sabotage
Valid stats: attack, defense, speed, max_hp, ability_power
Valid triggers (passive only): on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged, on_death, on_ability_used, on_hp_below, on_hp_above, on_shield_broken, periodic
Valid effects: damage <N>, heal <N>, shield <N>, stun <N>s, slow <factor> <N>s, knockback <N>, dash, buff <stat> <N>% <N>s, debuff <stat> <N>% <N>s, stealth <N>s, evasion <N>%, tenacity <N>, teleport, aura <stat> +<N>

Generate ONE ability for: '''

CLASS_PROMPT = '''Output ONLY one class definition, nothing else.

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

class Sentinel {
    stat_growth: +1 attack, +3 defense, +3 max_hp per level
    tags: melee, defense, leadership
    scaling party_alive_count {
        when party_members > 0: +10% defense
        when party_members >= 3: tenacity 0.5
        always: aura defense +2
    }
    abilities {
        level 1: shield_wall "Reduces incoming damage to party"
        level 5: taunt "Forces enemies to target this unit"
        level 10: iron_will "Immune to morale effects"
    }
    requirements: level 5, fame 50
}

Rules:
- stat_growth uses ONLY: attack, defense, speed, max_hp, ability_power (total per level <= 10)
- tags from: ranged, nature, stealth, melee, defense, leadership, arcane, elemental, healing, divine, assassination, agility, deception, sabotage
- scaling sources: party_alive_count, faction_allied_count, crisis_active, threat_level, fame
- requirements: level <N> or quest_count <N> or fame <N>
- ability names must be snake_case, 3-5 abilities spread across levels 1-40

Generate ONE class for: '''

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


VALID_TAGS = {
    "ranged", "nature", "stealth", "tracking", "survival",
    "melee", "defense", "leadership", "fortification", "honor",
    "arcane", "elemental", "ritual", "knowledge", "enchantment",
    "healing", "divine", "protection", "purification", "restoration",
    "assassination", "agility", "deception", "sabotage",
    "crisis", "legendary", "inspiration", "sacrifice",
}

VALID_STATS = {"attack", "defense", "speed", "max_hp", "ability_power"}

VALID_TRIGGERS = {
    "on_damage_dealt", "on_damage_taken", "on_kill", "on_ally_damaged",
    "on_death", "on_ability_used", "on_hp_below", "on_hp_above",
    "on_shield_broken", "periodic",
}

VALID_EFFECTS = {
    "damage", "heal", "shield", "stun", "slow", "knockback", "dash",
    "buff", "debuff", "stealth", "evasion", "tenacity", "teleport", "aura",
}


def extract_ability(text: str) -> str | None:
    """Extract an ability block from raw LLM output."""
    m = re.search(r'ability\s+\w+\s*\{[^}]+\}', text, re.DOTALL)
    return m.group(0) if m else None


def extract_class(text: str) -> str | None:
    """Extract a class block from raw LLM output."""
    m = re.search(r'class\s+\w+\s*\{.*?\n\}', text, re.DOTALL)
    return m.group(0) if m else None


def score_ability(text: str) -> float:
    """Score an ability block for quality. Higher is better."""
    score = 0.0

    # Has snake_case name
    name_m = re.search(r'ability\s+(\w+)', text)
    if name_m:
        name = name_m.group(1)
        if name == name.lower() and "_" in name:
            score += 2.0  # proper snake_case with underscore
        elif name == name.lower():
            score += 1.0  # lowercase but no underscore

    # Uses valid tag
    tag_m = re.search(r'tag:\s*(\S+)', text)
    if tag_m:
        tag = tag_m.group(1).strip().rstrip(",")
        if tag in VALID_TAGS:
            score += 3.0

    # Has valid type
    if re.search(r'type:\s*(active|passive)', text):
        score += 1.0

    # Active has cooldown, passive has trigger
    if re.search(r'type:\s*active', text) and re.search(r'cooldown:', text):
        score += 1.0
    if re.search(r'type:\s*passive', text) and re.search(r'trigger:', text):
        score += 1.0
        trigger_m = re.search(r'trigger:\s*(\S+)', text)
        if trigger_m and trigger_m.group(1) in VALID_TRIGGERS:
            score += 1.0

    # Effect uses known keywords
    effect_m = re.search(r'effect:\s*(.+)', text)
    if effect_m:
        effect_line = effect_m.group(1).lower()
        for kw in VALID_EFFECTS:
            if kw in effect_line:
                score += 0.5
                break
        # Bonus for effects that reference valid stats
        for stat in VALID_STATS:
            if stat in effect_line:
                score += 0.5
                break

    # Has description
    if re.search(r'description:\s*"[^"]+"', text):
        score += 1.0

    return score


def score_class(text: str) -> float:
    """Score a class block for quality. Higher is better."""
    score = 0.0

    # Has PascalCase name
    name_m = re.search(r'class\s+(\w+)', text)
    if name_m:
        name = name_m.group(1)
        if name[0].isupper() and "_" not in name:
            score += 2.0

    # stat_growth uses valid stats
    growth_m = re.search(r'stat_growth:\s*(.+?)per level', text)
    if growth_m:
        growth_line = growth_m.group(1).lower()
        valid_stat_count = sum(1 for s in VALID_STATS if s in growth_line)
        score += valid_stat_count * 0.5
        # Check total isn't absurd
        nums = re.findall(r'\+(\d+)', growth_line)
        total = sum(int(n) for n in nums)
        if 3 <= total <= 15:
            score += 2.0
        elif total <= 25:
            score += 1.0

    # Tags are valid
    tags_m = re.search(r'tags:\s*(.+)', text)
    if tags_m:
        tags = [t.strip().rstrip(",") for t in tags_m.group(1).split(",")]
        valid = sum(1 for t in tags if t in VALID_TAGS)
        score += valid * 0.5

    # Has scaling block
    if re.search(r'scaling\s+\w+\s*\{', text):
        score += 1.0

    # Has abilities block with levels
    abilities = re.findall(r'level\s+\d+:', text)
    if 3 <= len(abilities) <= 6:
        score += 2.0
    elif abilities:
        score += 1.0

    # Ability names are snake_case
    ability_names = re.findall(r'level\s+\d+:\s+(\w+)', text)
    snake_count = sum(1 for n in ability_names if n == n.lower())
    if ability_names:
        score += (snake_count / len(ability_names)) * 2.0

    # Has requirements
    if re.search(r'requirements:', text):
        score += 1.0

    return score


def _call_ollama(
    client: httpx.Client,
    messages: list[dict],
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    """Single Ollama chat call. Returns stripped text or None on error."""
    try:
        resp = client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=90,
        )
        return strip_think(resp.json()["message"]["content"])
    except Exception:
        return None


def generate_one(
    client: httpx.Client,
    context: str,
    gen_type: str,
    model: str,
    base_url: str,
    temperature: float = 0.7,
    retries: int = 1,
    candidates: int = 1,
) -> dict:
    """Generate a single ability or class via Ollama chat API.

    When candidates > 1, generates N variants in parallel and picks the
    best valid one by DSL quality score.

    Returns dict with keys: valid, text, raw, time_s, score, candidates_tried.
    """
    prompt = (ABILITY_PROMPT if gen_type == "ability" else CLASS_PROMPT) + context
    max_tokens = 300 if gen_type == "ability" else 500
    extractor = extract_ability if gen_type == "ability" else extract_class
    scorer = score_ability if gen_type == "ability" else score_class

    # Build messages — classes use multi-turn to avoid prompt continuation
    if gen_type == "class":
        words = [w.capitalize() for w in context.strip("a ").split()[:3]
                 if w.isalpha() and w not in ("a", "an", "the", "who", "that")]
        seed_name = "".join(words) or "Custom"
        messages = [
            {"role": "user", "content": f"Generate a class definition for {context}. "
                                        f"Use the exact DSL format I'll show you."},
            {"role": "assistant", "content": f"class {seed_name} {{"},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    total_attempts = candidates * (1 + retries)
    t0 = time.time()
    valid_candidates: list[tuple[str, str, float]] = []  # (block, raw, score)
    last_raw = ""

    for attempt in range(total_attempts):
        raw = _call_ollama(client, messages, model, base_url, temperature, max_tokens)
        if raw is None:
            continue
        last_raw = raw
        block = extractor(raw)
        if block:
            s = scorer(block)
            valid_candidates.append((block, raw, s))
            # Early exit: if we already have enough good candidates
            if len(valid_candidates) >= candidates:
                break

    elapsed = time.time() - t0

    if valid_candidates:
        # Pick the highest-scoring candidate
        valid_candidates.sort(key=lambda x: x[2], reverse=True)
        best_block, best_raw, best_score = valid_candidates[0]
        return {
            "valid": True, "text": best_block, "raw": best_raw,
            "time_s": elapsed, "score": best_score,
            "candidates_tried": len(valid_candidates),
        }
    else:
        return {
            "valid": False, "text": None, "raw": last_raw,
            "time_s": elapsed, "score": 0.0,
            "candidates_tried": 0,
        }


def generate_batch(
    contexts: list[tuple[str, str]],
    model: str,
    base_url: str,
    workers: int = 4,
    temperature: float = 0.7,
    retries: int = 1,
    candidates: int = 1,
) -> list[dict]:
    """Generate a batch of items in parallel.

    contexts: list of (gen_type, context_string) tuples.
    candidates: generate N variants per item, pick best valid one.
    Returns list of result dicts with added 'type' and 'context' keys.
    """
    client = httpx.Client()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for gen_type, ctx in contexts:
            f = pool.submit(
                generate_one, client, ctx, gen_type, model, base_url,
                temperature, retries, candidates,
            )
            futures[f] = (gen_type, ctx)

        done_count = 0
        total = len(futures)
        for f in concurrent.futures.as_completed(futures):
            gen_type, ctx = futures[f]
            r = f.result()
            r["type"] = gen_type
            r["context"] = ctx
            results.append(r)
            done_count += 1
            status = "OK" if r["valid"] else "FAIL"
            score_str = f" score={r.get('score', 0):.1f}" if r["valid"] else ""
            tried = r.get("candidates_tried", 0)
            tried_str = f" [{tried}/{candidates}]" if candidates > 1 else ""
            print(
                f"  [{done_count}/{total}] {gen_type:7s} {status}{tried_str}{score_str}  "
                f"{ctx[:45]}  ({r['time_s']:.1f}s)",
                file=sys.stderr,
            )

    client.close()
    return results


def print_summary(results: list[dict], total_time: float):
    """Print a summary of generation results."""
    valid_abilities = [r for r in results if r["type"] == "ability" and r["valid"]]
    valid_classes = [r for r in results if r["type"] == "class" and r["valid"]]
    n_abilities = sum(1 for r in results if r["type"] == "ability")
    n_classes = sum(1 for r in results if r["type"] == "class")
    total = len(results)
    valid = len([r for r in results if r["valid"]])

    print(f"\n{'='*60}")
    print(f"RESULTS: {valid}/{total} valid ({valid/total*100:.0f}%)")
    if n_abilities:
        print(f"  Abilities: {len(valid_abilities)}/{n_abilities} "
              f"({len(valid_abilities)/n_abilities*100:.0f}%)")
    if n_classes:
        print(f"  Classes:   {len(valid_classes)}/{n_classes} "
              f"({len(valid_classes)/n_classes*100:.0f}%)")
    print(f"  Time: {total_time:.1f}s ({total/total_time:.2f} items/s)")
    print(f"{'='*60}")

    if valid_abilities:
        print(f"\n--- Abilities ({len(valid_abilities)}) ---")
        for r in valid_abilities[:5]:
            print(f"\n{r['text']}")
        if len(valid_abilities) > 5:
            print(f"\n  ... and {len(valid_abilities) - 5} more")

    if valid_classes:
        print(f"\n--- Classes ({len(valid_classes)}) ---")
        for r in valid_classes[:3]:
            print(f"\n{r['text']}")
        if len(valid_classes) > 3:
            print(f"\n  ... and {len(valid_classes) - 3} more")

    failures = [r for r in results if not r["valid"]]
    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        for r in failures[:5]:
            raw = r["raw"][:150].replace("\n", " ")
            print(f"  [{r['type']}] {r['context'][:50]}...")
            print(f"    Raw: {raw}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate game content via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--type", choices=["ability", "class"], default="ability",
        help="Content type to generate (default: ability)",
    )
    parser.add_argument(
        "--context", default="a level 5 ranger",
        help="Context string describing the adventurer/class concept",
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of items to generate with the same context",
    )
    parser.add_argument(
        "--batch", metavar="FILE",
        help='JSONL file with batch requests (each line: {"type":"ability","context":"..."})',
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE",
        help="Output JSONL file (default: stdout for single, generated/content.jsonl for batch)",
    )
    parser.add_argument(
        "--model", default=os.environ.get("OLLAMA_MODEL", "qwen35-9b"),
        help="Ollama model name (default: qwen35-9b or $OLLAMA_MODEL)",
    )
    parser.add_argument(
        "--url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434 or $OLLAMA_URL)",
    )
    parser.add_argument(
        "--workers", "-j", type=int, default=4,
        help="Parallel workers for batch generation (default: 4)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--retries", type=int, default=1,
        help="Retries per item on format failure (default: 1)",
    )
    parser.add_argument(
        "--candidates", "-n", type=int, default=3,
        help="Generate N variants per item, pick best valid one (default: 3)",
    )
    args = parser.parse_args()

    # Check server
    try:
        resp = httpx.get(f"{args.url}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json()["models"]]
        if not any(args.model in m for m in models):
            print(f"Model '{args.model}' not found. Available: {models}", file=sys.stderr)
            sys.exit(1)
    except Exception:
        print(f"Cannot connect to Ollama at {args.url}. Is it running?", file=sys.stderr)
        print("Start with: OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=true ollama serve", file=sys.stderr)
        sys.exit(1)

    start = time.time()

    if args.batch:
        # Batch mode: read JSONL, generate all
        with open(args.batch) as f:
            requests = [json.loads(line) for line in f if line.strip()]
        contexts = [(r.get("type", "ability"), r.get("context", "")) for r in requests]
        print(f"Batch: {len(contexts)} items, {args.workers} workers", file=sys.stderr)
        results = generate_batch(
            contexts, args.model, args.url, args.workers,
            args.temperature, args.retries, args.candidates,
        )
    else:
        # Single/count mode
        contexts = [(args.type, args.context)] * args.count
        if args.count > 1:
            print(f"Generating {args.count} {args.type}(s)...", file=sys.stderr)
            results = generate_batch(
                contexts, args.model, args.url, args.workers,
                args.temperature, args.retries, args.candidates,
            )
        else:
            client = httpx.Client()
            r = generate_one(
                client, args.context, args.type, args.model, args.url,
                args.temperature, args.retries, args.candidates,
            )
            r["type"] = args.type
            r["context"] = args.context
            results = [r]
            client.close()

    total_time = time.time() - start

    # Output
    if args.count == 1 and not args.batch and not args.output:
        # Single item: print to stdout
        r = results[0]
        if r["valid"]:
            print(r["text"])
        else:
            print(f"Generation failed. Raw output:\n{r['raw']}", file=sys.stderr)
            sys.exit(1)
    else:
        # Batch/multi: write JSONL
        out_path = args.output or "generated/content.jsonl"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")
        print(f"\nSaved to {out_path}", file=sys.stderr)

    # Summary for multi-item runs
    if len(results) > 1:
        print_summary(results, total_time)


if __name__ == "__main__":
    main()
