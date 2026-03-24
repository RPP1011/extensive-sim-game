#!/usr/bin/env python3
"""Map simplified content-gen ability DSL to combat-sim ability DSL.

Content gen format:
    ability shield_bash {
        type: active
        cooldown: 10s
        effect: stun 2s + damage 30
        tag: melee
        description: "A devastating shield strike"
    }

Combat sim format:
    ability ShieldBash {
        target: enemy, range: 5.0
        cooldown: 10s, cast: 0ms
        hint: crowd_control
        stun 2000
        damage 30
    }

Usage:
    uv run python3 scripts/map_content_to_combat_dsl.py \
        --input generated/vae_dataset_final.jsonl \
        --output generated/vae_dataset_combat_dsl.jsonl
"""

import argparse
import json
import re
import sys


# Tag → hint mapping
TAG_TO_HINT = {
    "melee": "damage", "ranged": "damage", "assassination": "damage",
    "defense": "defense", "fortification": "defense", "protection": "defense",
    "stealth": "utility", "deception": "utility", "agility": "utility",
    "healing": "heal", "divine": "heal", "purification": "heal", "restoration": "heal",
    "arcane": "damage", "elemental": "damage", "ritual": "utility",
    "nature": "utility", "tracking": "utility", "survival": "defense",
    "leadership": "utility", "honor": "defense", "knowledge": "utility",
    "enchantment": "utility", "sabotage": "utility", "sacrifice": "utility",
    "inspiration": "utility", "crisis": "utility", "legendary": "damage",
}

# Tag → default range
TAG_TO_RANGE = {
    "melee": 2.0, "defense": 2.0, "fortification": 2.0, "honor": 2.0,
    "ranged": 6.0, "arcane": 6.0, "elemental": 6.0, "nature": 5.0,
    "healing": 5.0, "divine": 5.0, "purification": 5.0, "restoration": 5.0,
    "stealth": 3.0, "assassination": 3.0, "agility": 3.0,
}


def parse_content_ability(text):
    """Parse the simplified content-gen DSL into fields."""
    fields = {}

    name_m = re.search(r'ability\s+(\w+)', text)
    fields["name"] = name_m.group(1) if name_m else "unnamed"

    type_m = re.search(r'type:\s*(active|passive)', text)
    fields["type"] = type_m.group(1) if type_m else "active"

    cd_m = re.search(r'cooldown:\s*(\d+)s', text)
    fields["cooldown"] = int(cd_m.group(1)) if cd_m else 10

    trigger_m = re.search(r'trigger:\s*(\S+)', text)
    fields["trigger"] = trigger_m.group(1) if trigger_m else None

    tag_m = re.search(r'tag:\s*(\w+)', text)
    fields["tag"] = tag_m.group(1) if tag_m else "melee"

    effect_m = re.search(r'effect:\s*(.+)', text)
    fields["effect_line"] = effect_m.group(1).strip() if effect_m else ""

    return fields


def parse_effect_tokens(effect_line):
    """Parse 'damage 30 + stun 2s + buff defense 15% 5s' into combat DSL lines."""
    lines = []
    # Split on '+'
    parts = [p.strip() for p in effect_line.split("+")]

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # damage <N>
        m = re.match(r'damage\s+(\d+)', part)
        if m:
            lines.append(f"    damage {m.group(1)}")
            continue

        # heal <N>
        m = re.match(r'heal\s+(\d+)', part)
        if m:
            lines.append(f"    heal {m.group(1)}")
            continue

        # shield <N>
        m = re.match(r'shield\s+(\d+)', part)
        if m:
            lines.append(f"    shield {m.group(1)} for 5s")
            continue

        # stun <N>s
        m = re.match(r'stun\s+(\d+(?:\.\d+)?)s', part)
        if m:
            ms = int(float(m.group(1)) * 1000)
            lines.append(f"    stun {ms}")
            continue

        # slow <factor> <N>s
        m = re.match(r'slow\s+([\d.]+)\s+(\d+(?:\.\d+)?)s', part)
        if m:
            ms = int(float(m.group(2)) * 1000)
            lines.append(f"    slow {m.group(1)} for {ms}ms")
            continue

        # knockback <N>
        m = re.match(r'knockback\s+(\d+)', part)
        if m:
            lines.append(f"    knockback {m.group(1)}")
            continue

        # dash
        if part.strip() == "dash":
            lines.append("    dash")
            continue

        # teleport
        if part.strip() == "teleport":
            lines.append("    dash blink")
            continue

        # buff <stat> <N>% <N>s
        m = re.match(r'buff\s+(\w+)\s+(\d+)%\s+(\d+)s', part)
        if m:
            factor = int(m.group(2)) / 100.0
            ms = int(m.group(3)) * 1000
            lines.append(f"    buff {m.group(1)} {factor} for {ms}ms")
            continue

        # debuff <stat> <N>% <N>s
        m = re.match(r'debuff\s+(\w+)\s+(\d+)%\s+(\d+)s', part)
        if m:
            factor = int(m.group(2)) / 100.0
            ms = int(m.group(3)) * 1000
            lines.append(f"    debuff {m.group(1)} {factor} for {ms}ms")
            continue

        # stealth <N>s
        m = re.match(r'stealth\s+(\d+)s', part)
        if m:
            ms = int(m.group(1)) * 1000
            lines.append(f"    stealth {ms}")
            continue

        # evasion <N>%
        m = re.match(r'evasion\s+(\d+)%', part)
        if m:
            factor = int(m.group(1)) / 100.0
            lines.append(f"    buff evasion {factor} for 5000ms")
            continue

        # tenacity <N>
        m = re.match(r'tenacity\s+(\d+)', part)
        if m:
            lines.append(f"    buff tenacity {int(m.group(1))/100.0} for 5000ms")
            continue

        # aura <stat> +<N>
        m = re.match(r'aura\s+(\w+)\s+\+?(\d+)', part)
        if m:
            lines.append(f"    buff {m.group(1)} {int(m.group(2))/100.0} for 10000ms")
            continue

        # Fallback: treat as raw
        # Skip unrecognized

    return lines


def to_pascal_case(snake):
    """shield_bash → ShieldBash"""
    return "".join(w.capitalize() for w in snake.split("_"))


def convert_ability(text):
    """Convert content-gen DSL to combat-sim DSL."""
    fields = parse_content_ability(text)
    name = to_pascal_case(fields["name"])
    tag = fields["tag"]
    hint = TAG_TO_HINT.get(tag, "utility")
    range_val = TAG_TO_RANGE.get(tag, 4.0)

    effect_lines = parse_effect_tokens(fields["effect_line"])
    if not effect_lines:
        effect_lines = ["    damage 10"]  # fallback

    if fields["type"] == "passive":
        trigger = fields["trigger"] or "on_damage_taken"
        # Fix trigger format for combat DSL parser
        if trigger == "periodic":
            trigger = "periodic(5000ms)"
        elif trigger == "on_hp_below":
            trigger = "on_hp_below(50%)"
        elif trigger == "on_hp_above":
            trigger = "on_hp_above(80%)"
        elif trigger == "on_ally_damaged":
            trigger = "on_ally_damaged(5.0)"
        elif trigger == "on_ally_killed":
            trigger = "on_ally_killed(5.0)"
        lines = [
            f"passive {name} {{",
            f"    trigger: {trigger}",
            f"    cooldown: {fields['cooldown']}s",
            "",
        ]
        lines.extend(effect_lines)
        lines.append("}")
    else:
        lines = [
            f"ability {name} {{",
            f"    target: enemy, range: {range_val}",
            f"    cooldown: {fields['cooldown']}s, cast: 0ms",
            f"    hint: {hint}",
            "",
        ]
        lines.extend(effect_lines)
        lines.append("}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated/vae_dataset_final.jsonl")
    parser.add_argument("--output", "-o", default="generated/vae_dataset_combat_dsl.jsonl")
    args = parser.parse_args()

    total = 0
    converted = 0
    failed = 0

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            if r["content_type"] == "ability":
                try:
                    combat_dsl = convert_ability(r["raw_dsl"])
                    r["combat_dsl"] = combat_dsl
                    converted += 1
                except Exception as e:
                    r["combat_dsl"] = ""
                    failed += 1
            else:
                r["combat_dsl"] = r["raw_dsl"]  # classes already parse fine
                converted += 1

            fout.write(json.dumps(r) + "\n")

    print(f"Total: {total:,}, Converted: {converted:,}, Failed: {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
