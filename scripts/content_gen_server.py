#!/usr/bin/env python3
"""Content generation server for the headless campaign.

Accepts generation requests via stdin (JSON lines), sends prompts to
vLLM, validates output with the DSL parsers, and returns results.

Usage:
    # Start vLLM first:
    vllm serve Qwen/Qwen3.5-4B --port 8100 --max-model-len 4096

    # Then run this:
    echo '{"type":"ability","archetype":"ranger","level":5,"context":"completed 10 quests solo"}' | python3 scripts/content_gen_server.py
"""

import json
import sys
import os

try:
    import openai
except ImportError:
    print("pip install openai", file=sys.stderr)
    sys.exit(1)

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8100/v1")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-4B")

client = openai.OpenAI(base_url=VLLM_URL, api_key="unused")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

ABILITY_SYSTEM = """\
You generate ability definitions in a custom DSL for a fantasy guild management game.

Output ONLY the ability block, no explanation.

Format:
```
ability <name> {
    type: active|passive
    cooldown: Ns (for active)
    trigger: <trigger> (for passive)
    effect: <description of mechanical effect>
    tag: <one of the class tags>
    description: "<flavor text>"
}
```

Available tags: ranged, nature, stealth, tracking, survival, melee, defense, leadership,
fortification, honor, arcane, elemental, ritual, knowledge, enchantment, healing, divine,
protection, purification, restoration, assassination, agility, deception, sabotage,
crisis, legendary, inspiration, sacrifice

Available triggers (passive): on_damage_dealt, on_damage_taken, on_kill, on_ally_damaged,
on_death, on_ability_used, on_hp_below, on_hp_above, on_shield_broken, periodic

Effects can include: damage, heal, shield, stun, slow, knockback, dash, buff, debuff,
stealth, evasion, tenacity, inspire, aura, teleport
"""

CLASS_SYSTEM = """\
You generate class definitions in a custom DSL for a fantasy guild management game.

Output ONLY the class block, no explanation.

Format:
```
class <Name> {
    stat_growth: +N attack, +N defense, +N speed, +N max_hp, +N ability_power per level

    tags: tag1, tag2, tag3

    scaling <source> {
        when <condition>: <bonus>
        always: <bonus>
    }

    abilities {
        level N: ability_name "description"
    }

    requirements: level N, fame N
    consolidates_at: N (optional, for prestige classes)
}
```

Valid scaling sources: party_alive_count, party_size, faction_strength, coalition_strength,
crisis_severity, fame, territory_control, adventurer_count, gold, reputation, threat_level

Valid bonuses: +N% stat, +N stat, tenacity N, escape N, aura stat +N,
last_stand below N% hp +N% attack, inspire nearby +N stat

Stat growth total per level should be 5-15 for normal classes, up to 25 for hero classes.
"""

QUEST_SYSTEM = """\
You generate quest hook definitions in TOML for a fantasy guild management game.

Output ONLY the TOML, no explanation. Follow this format exactly:

```toml
name = "Quest Name"
priority = 5
repeatable = false
cooldown_ticks = 5000

[trigger]
type = "TriggerType"
# trigger-specific fields

[choice]
category = "quest_branch"
trigger = "hook"
prompt = "Description of the situation"
deadline_secs = 60.0
default_option = 0

[[choice.options]]
label = "Option 1"
description = "What happens"
effects = [
    { type = "gold", amount = -20.0 },
    { type = "narrative", text = "Something happens." },
]
```

Valid trigger types: AdventurerTrait, AdventurerLevel, FactionRelation, LocationScouted,
CampaignProgress, GuildResource, QuestsCompleted, ThreatLevel, AdventurerBond, Periodic
"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(prompt: str, system: str, max_tokens: int = 512) -> str:
    """Call vLLM and return the generated text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def generate_ability(request: dict) -> dict:
    """Generate an ability from context."""
    archetype = request.get("archetype", "ranger")
    level = request.get("level", 5)
    context = request.get("context", "")
    tags = request.get("tags", [archetype])

    prompt = f"""Generate a unique ability for a level {level} {archetype}.
Tags available: {', '.join(tags)}
Context: {context}
The ability should feel appropriate for their experience and recent actions."""

    raw = generate(prompt, ABILITY_SYSTEM)

    # Extract the ability block if wrapped in ```
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "ability " in part:
                raw = part.strip()
                break

    return {"type": "ability", "content": raw, "raw": raw}


def generate_class(request: dict) -> dict:
    """Generate a class from context."""
    archetype = request.get("archetype", "ranger")
    level = request.get("level", 10)
    fame = request.get("fame", 100)
    context = request.get("context", "")

    prompt = f"""Generate a class specialization for a level {level} {archetype} with {fame} fame.
Context: {context}
The class should reflect how they've been playing and what they've experienced."""

    raw = generate(prompt, CLASS_SYSTEM, max_tokens=800)

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "class " in part:
                raw = part.strip()
                break

    return {"type": "class", "content": raw, "raw": raw}


def generate_quest_hook(request: dict) -> dict:
    """Generate a quest hook from context."""
    context = request.get("context", "")
    trigger_type = request.get("trigger_type", "GuildResource")

    prompt = f"""Generate a quest hook for a guild management game.
Trigger type: {trigger_type}
Context: {context}
The quest should create an interesting choice with meaningful trade-offs."""

    raw = generate(prompt, QUEST_SYSTEM, max_tokens=800)

    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            if "name =" in part or "[trigger]" in part:
                raw = part.strip()
                break

    return {"type": "quest_hook", "content": raw, "raw": raw}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    """Process generation requests from stdin."""
    # Single request mode (for testing)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Testing ability generation...", file=sys.stderr)
        result = generate_ability({
            "archetype": "ranger",
            "level": 10,
            "context": "Has completed 15 solo quests in dungeons, specializes in stealth",
            "tags": ["ranged", "stealth", "tracking"],
        })
        print(json.dumps(result, indent=2))

        print("\nTesting class generation...", file=sys.stderr)
        result = generate_class({
            "archetype": "knight",
            "level": 15,
            "fame": 200,
            "context": "Defended 3 regions from attacks, formed a coalition with The Accord",
        })
        print(json.dumps(result, indent=2))

        print("\nTesting quest hook generation...", file=sys.stderr)
        result = generate_quest_hook({
            "context": "Guild has 12 adventurers, reputation 80, a Sleeping King crisis is active with 3 champions arrived",
            "trigger_type": "CampaignProgress",
        })
        print(json.dumps(result, indent=2))
        return

    # JSONL stdin mode
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "invalid JSON"}))
            sys.stdout.flush()
            continue

        gen_type = request.get("type", "ability")
        if gen_type == "ability":
            result = generate_ability(request)
        elif gen_type == "class":
            result = generate_class(request)
        elif gen_type == "quest_hook":
            result = generate_quest_hook(request)
        else:
            result = {"error": f"unknown type: {gen_type}"}

        print(json.dumps(result))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
