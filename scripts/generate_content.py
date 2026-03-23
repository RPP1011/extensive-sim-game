#!/usr/bin/env python3
"""Generate game content (abilities, classes) using a local LLM.

Usage:
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --type ability --context "level 10 stealth ranger"
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --type class --context "level 15 knight, defended 3 regions"
    uv run --with transformers --with torch --with accelerate python3 scripts/generate_content.py --batch abilities.jsonl
"""

import argparse
import json
import sys
import time

MODEL_ID = "Qwen/Qwen3.5-9B"

ABILITY_SYSTEM = '''You generate ability definitions in a custom DSL. Output ONLY the ability block. No thinking. No explanation. No markdown.

## Grammar

ability <snake_case_name> {
    type: active|passive
    cooldown: <N>s                    (required for active, omit for passive)
    trigger: <trigger_type>           (required for passive, omit for active)
    effect: <effect_description>
    tag: <single_tag>
    description: "<flavor text>"
}

## Valid stats (ONLY these exist)
attack, defense, speed, max_hp, ability_power

## Valid tags (pick ONE)
ranged, nature, stealth, tracking, survival,
melee, defense, leadership, fortification, honor,
arcane, elemental, ritual, knowledge, enchantment,
healing, divine, protection, purification, restoration,
assassination, agility, deception, sabotage,
crisis, legendary, inspiration, sacrifice

## Valid triggers (passive only)
on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged,
on_death, on_ability_used, on_hp_below, on_hp_above,
on_shield_broken, periodic

## Effect types
damage <N>, heal <N>, shield <N>, stun <N>s, slow <factor> <N>s,
knockback <N>, dash, buff <stat> <N>% <N>s, debuff <stat> <N>% <N>s,
stealth <N>s, evasion <N>%, tenacity <N>, teleport, aura <stat> +<N>

## Examples

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
}'''

CLASS_SYSTEM = '''You generate class definitions in a custom DSL. Output ONLY the class block. No thinking. No explanation. No markdown.

## Grammar

class <PascalCaseName> {
    stat_growth: +<N> <stat>, +<N> <stat>, ... per level
    tags: <tag>, <tag>, ...
    scaling <source> {
        when <condition>: <bonus>
        always: <bonus>
    }
    abilities {
        level <N>: <snake_case_name> "<description>"
    }
    requirements: <req>, <req>, ...
    consolidates_at: <N>              (optional, for prestige classes)
}

## Valid stats (ONLY these, no others)
attack, defense, speed, max_hp, ability_power

## Stat growth rules
- Total per level should be 5-15 for normal classes, up to 25 for hero classes
- Use "+N all" as shorthand for equal growth in all stats

## Valid tags
ranged, nature, stealth, tracking, survival,
melee, defense, leadership, fortification, honor,
arcane, elemental, ritual, knowledge, enchantment,
healing, divine, protection, purification, restoration,
assassination, agility, deception, sabotage,
crisis, legendary, inspiration, sacrifice

## Valid scaling sources
party_alive_count, party_size, faction_strength, coalition_strength,
crisis_severity, fame, territory_control, adventurer_count,
gold, reputation, threat_level

## Valid conditions in "when" clauses
party_members > N, party_members >= N, faction_alive,
faction_territory >= N, crisis_active, crisis_severity > N,
solo (party_members == 1)

## Valid bonuses
+N% <stat>          (percentage stat boost)
+N <stat>           (flat stat boost)
tenacity <N>        (CC reduction, 0-1)
escape <N>          (disengage chance)
aura <stat> +<N>    (buff nearby allies)
last_stand below <N>% max_hp +<N>% attack
inspire nearby +<N> <stat>

## Valid requirements
level <N>, fame <N>, quests <N>, trait <name>,
active_crisis, gold <N>, group_size <N>, allies <N>

## Examples

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

class Shadowmaster {
    stat_growth: +3 attack, +3 speed, +1 ability_power per level
    tags: stealth, assassination, agility
    scaling party_alive_count {
        when party_members == 1: +25% attack
        always: +5% speed
    }
    abilities {
        level 1: ambush "Bonus damage on first strike"
        level 5: evasion "Chance to dodge attacks"
        level 10: assassinate "Attempt to one-shot a target"
    }
    requirements: level 10, fame 100
}'''


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map='cuda')
    return tokenizer, model


def generate(tokenizer, model, system, user, prefix, max_tokens=250):
    import torch
    raw = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefix}'
    ids = tokenizer(raw, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_tokens,
            temperature=0.7, do_sample=True, repetition_penalty=1.1
        )
    txt = prefix + tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

    # Find matching closing brace
    depth = 0
    for i, c in enumerate(txt):
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth <= 0:
                return txt[:i+1].strip()
    return txt.strip()


def generate_ability(tokenizer, model, context, tags=""):
    prompt = f"Write an ability for {context}."
    if tags:
        prompt += f" Tags: {tags}"
    return generate(tokenizer, model, ABILITY_SYSTEM, prompt, "ability ")


def generate_class(tokenizer, model, context, tags=""):
    prompt = f"Write a class for {context}."
    if tags:
        prompt += f" Tags: {tags}"
    return generate(tokenizer, model, CLASS_SYSTEM, prompt, "class ", max_tokens=350)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["ability", "class"], default="ability")
    parser.add_argument("--context", default="a level 5 ranger")
    parser.add_argument("--tags", default="")
    parser.add_argument("--batch", help="JSONL file with batch requests")
    parser.add_argument("--count", type=int, default=1, help="Number to generate")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...", file=sys.stderr)
    tokenizer, model = load_model()
    print("Ready.", file=sys.stderr)

    out = open(args.output, 'w') if args.output else sys.stdout

    if args.batch:
        with open(args.batch) as f:
            requests = [json.loads(line) for line in f if line.strip()]
        for i, req in enumerate(requests):
            t0 = time.time()
            gen_type = req.get("type", "ability")
            context = req.get("context", "a level 5 ranger")
            tags = req.get("tags", "")
            if gen_type == "ability":
                result = generate_ability(tokenizer, model, context, tags)
            else:
                result = generate_class(tokenizer, model, context, tags)
            elapsed = time.time() - t0
            print(json.dumps({"type": gen_type, "content": result, "context": context, "time": elapsed}), file=out)
            out.flush()
            print(f"  [{i+1}/{len(requests)}] {gen_type} in {elapsed:.1f}s", file=sys.stderr)
    else:
        for i in range(args.count):
            t0 = time.time()
            if args.type == "ability":
                result = generate_ability(tokenizer, model, args.context, args.tags)
            else:
                result = generate_class(tokenizer, model, args.context, args.tags)
            elapsed = time.time() - t0
            print(result)
            print()
            print(f"Generated in {elapsed:.1f}s", file=sys.stderr)

    if args.output:
        out.close()


if __name__ == "__main__":
    main()
