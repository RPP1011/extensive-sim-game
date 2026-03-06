#!/usr/bin/env python3
"""Generate diverse training scenarios for oracle dataset generation.

Creates scenarios across multiple axes of variation:
- Team compositions (mixed heroes, not just 3war+1)
- Team sizes (2v3 through 6v8)
- Difficulty levels (1-5)
- HP multipliers (1x-5x)
- Room types (Entry, Pressure, Recovery, Climax)
- Multiple seed variants per composition for trajectory diversity
"""

import random
from pathlib import Path

HEROES = [
    "warrior", "mage", "ranger", "rogue", "cleric", "paladin", "druid",
    "assassin", "bard", "berserker", "blood_mage", "cryomancer",
    "elementalist", "engineer", "knight", "monk", "necromancer",
    "pyromancer", "samurai", "shadow_dancer", "shaman", "templar",
    "warden", "warlock", "witch_doctor", "alchemist",
]

TANKS = ["warrior", "knight", "paladin", "warden", "templar"]
HEALERS = ["cleric", "druid", "bard", "shaman", "alchemist"]
MELEE_DPS = ["rogue", "assassin", "berserker", "samurai", "shadow_dancer", "monk"]
RANGED_DPS = ["mage", "ranger", "pyromancer", "cryomancer", "elementalist", "engineer"]
HYBRID = ["blood_mage", "necromancer", "warlock", "witch_doctor"]

OUT_DIR = Path("scenarios/training")
SEED_VARIANTS = 3  # Number of seed variants per composition


def make_scenario(name, heroes_str, hero_count, enemy_count, difficulty, hp_mult, room_type, seed):
    return '\n'.join([
        '[scenario]',
        f'name = "{name}"',
        f'seed = {seed}',
        f'hero_count = {hero_count}',
        f'enemy_count = {enemy_count}',
        f'difficulty = {difficulty}',
        f'max_ticks = 10000',
        f'room_type = "{room_type}"',
        f'hero_templates = {heroes_str}',
        f'hp_multiplier = {hp_mult}',
        '', '[assert]', 'outcome = "Either"',
    ]) + '\n'


def add_variants(scenarios, base_name, heroes, ec, diff, hp, room, base_seed):
    """Add SEED_VARIANTS copies with different seeds."""
    for v in range(SEED_VARIANTS):
        seed = base_seed + v * 10000
        suffix = f"_s{v}" if v > 0 else ""
        scenarios.append((f"{base_name}{suffix}", heroes, ec, diff, hp, room, seed))


def main():
    random.seed(2026)
    scenarios = []
    idx = 0

    # --- Category 1: Balanced party compositions (tank+healer+dps) ---
    balanced_parties = [
        (["warrior", "cleric", "mage", "rogue"], "classic"),
        (["paladin", "druid", "ranger", "assassin"], "nature"),
        (["knight", "bard", "pyromancer", "berserker"], "siege"),
        (["templar", "shaman", "engineer", "monk"], "tech"),
        (["warden", "alchemist", "elementalist", "samurai"], "mystic"),
        (["warrior", "cleric", "cryomancer", "shadow_dancer"], "frost"),
        (["paladin", "druid", "warlock", "rogue"], "shadow"),
        (["knight", "bard", "blood_mage", "ranger"], "blood"),
    ]
    for heroes, tag in balanced_parties:
        for diff in [1, 2, 3]:
            for ec in [4, 5, 6]:
                hp = 3.0 if ec > 4 else 1.0
                name = f"balanced_{tag}_{len(heroes)}v{ec}_d{diff}"
                add_variants(scenarios, name, heroes, ec, diff, hp, "Entry", 1000 + idx)
                idx += 1

    # --- Category 2: Dual healer compositions ---
    dual_healer = [
        (["warrior", "warrior", "cleric", "druid"], "war_cler_druid"),
        (["paladin", "knight", "bard", "shaman"], "tank_bard_sham"),
        (["berserker", "samurai", "cleric", "bard"], "melee_heal"),
        (["warrior", "warrior", "alchemist", "cleric"], "war_alch_cler"),
    ]
    for heroes, tag in dual_healer:
        for ec in [5, 6, 7]:
            for hp in [2.0, 4.0]:
                name = f"dual_healer_{tag}_v{ec}_hp{hp:.0f}x"
                add_variants(scenarios, name, heroes, ec, 2, hp, "Recovery", 2000 + idx)
                idx += 1

    # --- Category 3: All-ranged vs melee swarm ---
    ranged_parties = [
        ["mage", "ranger", "pyromancer", "engineer"],
        ["cryomancer", "elementalist", "ranger", "mage"],
        ["engineer", "warlock", "pyromancer", "necromancer"],
    ]
    for heroes in ranged_parties:
        for ec in [6, 8]:
            for diff in [1, 2]:
                name = f"ranged_v{ec}_d{diff}_{heroes[0]}"
                add_variants(scenarios, name, heroes, ec, diff, 2.0, "Entry", 3000 + idx)
                idx += 1

    # --- Category 4: Small skirmishes (2v3, 3v3, 2v4) ---
    small_parties = [
        (["warrior", "cleric"], 3),
        (["rogue", "mage"], 3),
        (["paladin", "ranger"], 3),
        (["berserker", "druid"], 4),
        (["assassin", "bard"], 4),
        (["warrior", "mage", "cleric"], 3),
        (["knight", "ranger", "druid"], 3),
        (["rogue", "pyromancer", "shaman"], 4),
        (["samurai", "engineer", "bard"], 4),
    ]
    for heroes, ec in small_parties:
        for diff in [1, 2, 3]:
            for hp in [1.0, 3.0]:
                name = f"small_{len(heroes)}v{ec}_d{diff}_hp{hp:.0f}x"
                add_variants(scenarios, name, heroes, ec, diff, hp, "Entry", 4000 + idx)
                idx += 1

    # --- Category 5: Large battles (5v5, 5v6, 6v6, 6v8) ---
    large_parties = [
        ["warrior", "paladin", "cleric", "mage", "rogue"],
        ["knight", "warden", "druid", "ranger", "assassin"],
        ["templar", "berserker", "bard", "pyromancer", "monk"],
        ["warrior", "warrior", "cleric", "druid", "mage", "ranger"],
        ["paladin", "knight", "bard", "shaman", "elementalist", "rogue"],
    ]
    for heroes in large_parties:
        for ec in [len(heroes), len(heroes) + 1, len(heroes) + 2]:
            for diff in [2, 3]:
                hp = 2.0 if ec > len(heroes) else 1.0
                name = f"large_{len(heroes)}v{ec}_d{diff}"
                add_variants(scenarios, name, heroes, ec, diff, hp, "Pressure", 5000 + idx)
                idx += 1

    # --- Category 6: Boss fights (few heroes vs few tough enemies) ---
    boss_parties = [
        ["warrior", "cleric", "mage", "rogue"],
        ["paladin", "druid", "ranger", "berserker"],
        ["knight", "bard", "engineer", "assassin"],
        ["templar", "shaman", "warlock", "samurai"],
    ]
    for heroes in boss_parties:
        for ec in [2, 3]:
            for diff in [4, 5]:
                name = f"boss_{len(heroes)}v{ec}_d{diff}"
                add_variants(scenarios, name, heroes, ec, diff, 1.0, "Climax", 6000 + idx)
                idx += 1

    # --- Category 7: Pressure room gauntlets ---
    gauntlet_parties = [
        ["warrior", "warrior", "cleric", "mage"],
        ["paladin", "druid", "ranger", "rogue"],
        ["knight", "bard", "engineer", "berserker"],
    ]
    for heroes in gauntlet_parties:
        for ec in [7, 8, 10]:
            name = f"gauntlet_4v{ec}_d1"
            add_variants(scenarios, name, heroes, ec, 1, 1.0, "Pressure", 7000 + idx)
            idx += 1

    # --- Category 8: Random diverse compositions (more of these) ---
    for i in range(100):
        party_size = random.choice([3, 4, 5])
        heroes = random.sample(HEROES, party_size)
        ec = random.randint(party_size - 1, party_size + 3)
        diff = random.randint(1, 4)
        hp = random.choice([1.0, 2.0, 3.0, 5.0])
        room = random.choice(["Entry", "Pressure", "Recovery", "Climax"])
        name = f"random_{i:03d}_{party_size}v{ec}_d{diff}"
        seed = 8000 + i
        scenarios.append((name, heroes, ec, diff, hp, room, seed))

    # Write them all out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old files first
    for f in OUT_DIR.glob("*.toml"):
        f.unlink()

    for name, heroes, ec, diff, hp, room, seed in scenarios:
        hero_str = str(heroes).replace("'", '"')
        content = make_scenario(name, hero_str, len(heroes), ec, diff, hp, room, seed)
        fname = name.replace(" ", "_") + ".toml"
        (OUT_DIR / fname).write_text(content)

    print(f"Generated {len(scenarios)} scenarios in {OUT_DIR}/")

    sizes = {}
    for _, heroes, ec, *_ in scenarios:
        key = f"{len(heroes)}v{ec}"
        sizes[key] = sizes.get(key, 0) + 1
    print("\nBy team size:")
    for k in sorted(sizes.keys()):
        print(f"  {k}: {sizes[k]}")

    hero_counts = {}
    for _, heroes, *_ in scenarios:
        for h in heroes:
            hero_counts[h] = hero_counts.get(h, 0) + 1
    print(f"\nHero appearances (total {sum(hero_counts.values())}):")
    for h in sorted(hero_counts.keys(), key=lambda x: -hero_counts[x]):
        print(f"  {h}: {hero_counts[h]}")


if __name__ == "__main__":
    main()
