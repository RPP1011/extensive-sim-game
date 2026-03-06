#!/usr/bin/env python3
"""Generate champion animation mapping: ability name -> animation clip name.

Reads assets/lol_champions/*.json to build a mapping from ability name to the
LoL .anm animation name (spell1, spell2, spell3, spell4).

Output: assets/lol_models/anim_map.json

Format:
{
  "Ahri": {
    "abilities": {
      "Orb of Deception": { "slot": "Q", "anim": "spell1" },
      "Fox-Fire":         { "slot": "W", "anim": "spell2" },
      "Charm":            { "slot": "E", "anim": "spell3" },
      "Spirit Rush":      { "slot": "R", "anim": "spell4" }
    },
    "passive": { "name": "Essence Theft", "anim": "idle1" },
    "basic_attack": "attack1",
    "move": "run",
    "death": "death1",
    "idle": "idle1"
  },
  ...
}
"""

import json
import os
import sys
from pathlib import Path

SLOT_TO_ANIM = {
    "Q": "spell1",
    "W": "spell2",
    "E": "spell3",
    "R": "spell4",
}

SLOT_ORDER = ["Q", "W", "E", "R"]


def process_champion(json_path: Path) -> dict | None:
    """Process a single champion JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    abilities_data = data.get("abilities", {})
    if not isinstance(abilities_data, dict):
        return None

    result = {
        "abilities": {},
        "basic_attack": "attack1",
        "move": "run",
        "death": "death1",
        "idle": "idle1",
    }

    # Process passive
    passive = abilities_data.get("passive", {})
    if passive and isinstance(passive, dict):
        result["passive"] = {
            "name": passive.get("name", ""),
            "anim": "idle1",  # Passives don't have unique animations typically
        }

    # Process Q/W/E/R
    for slot in SLOT_ORDER:
        ability = abilities_data.get(slot, {})
        if ability and isinstance(ability, dict):
            name = ability.get("name", "")
            if name:
                result["abilities"][name] = {
                    "slot": slot,
                    "anim": SLOT_TO_ANIM[slot],
                }

    return result


def main():
    champions_dir = Path("assets/lol_champions")
    output_dir = Path("assets/lol_models")

    if not champions_dir.exists():
        print(f"Error: {champions_dir} not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    anim_map = {}
    json_files = sorted(champions_dir.glob("*.json"))

    for json_path in json_files:
        name = json_path.stem
        result = process_champion(json_path)
        if result:
            anim_map[name] = result

    output_path = output_dir / "anim_map.json"
    with open(output_path, "w") as f:
        json.dump(anim_map, f, indent=2)

    print(f"Generated animation map for {len(anim_map)} champions -> {output_path}")

    # Verify coverage against TOML files
    toml_dir = Path("assets/lol_heroes")
    if toml_dir.exists():
        toml_files = list(toml_dir.glob("*.toml"))
        mapped = set(anim_map.keys())
        toml_names = {f.stem for f in toml_files}
        missing = toml_names - mapped
        if missing:
            print(f"Warning: {len(missing)} TOML heroes have no JSON mapping: {sorted(missing)[:10]}...")
        else:
            print(f"All {len(toml_names)} TOML heroes have animation mappings")

    # Quick stats
    total_abilities = sum(len(v["abilities"]) for v in anim_map.values())
    print(f"Total ability mappings: {total_abilities}")


if __name__ == "__main__":
    main()
