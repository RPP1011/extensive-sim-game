#!/usr/bin/env python3
"""
Convert LoL champion JSON files (from fetch_lol_champions.py) into hero TOML files.

Usage:
  python3 scripts/lol_to_toml.py                # convert all champions
  python3 scripts/lol_to_toml.py Ahri Garen     # convert specific champions
  python3 scripts/lol_to_toml.py --dry-run Ahri  # preview without writing

Reads:  assets/lol_champions/*.json
Writes: assets/lol_heroes/{name}.toml
        assets/lol_heroes/_manifest.json
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

IN_DIR = Path(__file__).parent.parent / "assets" / "lol_champions"
OUT_DIR = Path(__file__).parent.parent / "assets" / "lol_heroes"

# ---------------------------------------------------------------------------
# Stat templates by primary tag — used when Data Dragon stats aren't available
# ---------------------------------------------------------------------------

STAT_TEMPLATES = {
    "Fighter":   {"hp": 120, "move_speed": 3.4, "attack_damage": 18, "attack_range": 1.5, "attack_cooldown": 800, "armor": 35, "magic_resist": 32},
    "Tank":      {"hp": 150, "move_speed": 3.3, "attack_damage": 14, "attack_range": 1.5, "attack_cooldown": 900, "armor": 40, "magic_resist": 35},
    "Mage":      {"hp": 90,  "move_speed": 3.3, "attack_damage": 12, "attack_range": 5.5, "attack_cooldown": 1000, "armor": 20, "magic_resist": 30},
    "Assassin":  {"hp": 85,  "move_speed": 3.5, "attack_damage": 20, "attack_range": 1.5, "attack_cooldown": 750, "armor": 25, "magic_resist": 30},
    "Marksman":  {"hp": 80,  "move_speed": 3.3, "attack_damage": 18, "attack_range": 5.5, "attack_cooldown": 700, "armor": 22, "magic_resist": 28},
    "Support":   {"hp": 95,  "move_speed": 3.3, "attack_damage": 10, "attack_range": 5.0, "attack_cooldown": 1000, "armor": 25, "magic_resist": 30},
}
DEFAULT_STATS = STAT_TEMPLATES["Fighter"]

# ---------------------------------------------------------------------------
# Resource type detection
# ---------------------------------------------------------------------------

RESOURCELESS = {"None", "No Cost", "", "Fury", "Ferocity", "Heat", "Courage",
                "Grit", "Shield", "Bloodthirst", "Flow", "Rage"}

# ---------------------------------------------------------------------------
# Keyword patterns for effect detection
# ---------------------------------------------------------------------------

def _num(text: str) -> int | None:
    """Extract first number from text."""
    m = re.search(r'(\d+)', text)
    return int(m.group(1)) if m else None

def _float(text: str) -> float | None:
    m = re.search(r'([\d.]+)', text)
    return float(m.group(1)) if m else None

def _ap_first(text: str) -> int | None:
    """Extract first value from {{ap|X to Y}} or {{ap|X/Y/Z}} patterns."""
    m = re.search(r'\{\{ap\|(\d+)', text)
    return int(m.group(1)) if m else None

def _ap_range(text: str) -> tuple[int | None, int | None]:
    """Extract (min, max) from {{ap|X to Y}}."""
    m = re.search(r'\{\{ap\|(\d+)\s+to\s+(\d+)\}\}', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EffectDef:
    type: str
    fields: dict = field(default_factory=dict)
    condition: dict | None = None
    area: dict | None = None
    tags: dict | None = None

@dataclass
class AbilityOut:
    name: str
    targeting: str = "target_enemy"
    range: float = 0.0
    cooldown_ms: int = 8000
    cast_time_ms: int = 0
    ai_hint: str = "damage"
    effects: list = field(default_factory=list)
    resource_cost: int = 0
    # Advanced mechanics
    max_charges: int = 0
    charge_recharge_ms: int = 0
    recast_count: int = 0
    recast_window_ms: int = 0
    is_toggle: bool = False
    unstoppable: bool = False

@dataclass
class PassiveOut:
    name: str
    trigger_type: str = "on_damage_dealt"
    cooldown_ms: int = 0
    range: float = 0.0
    effects: list = field(default_factory=list)
    trigger_extra: dict = field(default_factory=dict)

@dataclass
class ConversionResult:
    name: str
    confidence: str  # "high", "medium", "low"
    warnings: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Stat extraction
# ---------------------------------------------------------------------------

def get_stats(champ: dict) -> dict:
    """Build stat dict from champion tags."""
    tags = champ.get("tags", [])
    primary = tags[0] if tags else "Fighter"
    base = dict(STAT_TEMPLATES.get(primary, DEFAULT_STATS))

    # Melee vs ranged heuristic: if any ability wiki says range > 400 and
    # tags include Marksman/Mage, use ranged stats
    if primary in ("Marksman", "Mage"):
        base["attack_range"] = 5.5
    elif primary == "Support":
        # Check if melee support
        base["attack_range"] = 5.0
    else:
        base["attack_range"] = 1.5

    # Secondary tag adjustments
    if len(tags) > 1:
        sec = tags[1]
        if sec == "Tank":
            base["hp"] = int(base["hp"] * 1.15)
            base["armor"] += 5
        elif sec == "Assassin":
            base["move_speed"] = max(base["move_speed"], 3.5)
            base["attack_damage"] += 3
        elif sec == "Mage":
            base["magic_resist"] += 3
        elif sec == "Marksman":
            base["attack_range"] = max(base["attack_range"], 5.0)

    # Resource
    resource_type = champ.get("resource_type", "")
    if resource_type and resource_type not in RESOURCELESS:
        base["max_resource"] = 60
        base["resource_regen_per_sec"] = 2.0
    else:
        base["max_resource"] = 0

    return base

# ---------------------------------------------------------------------------
# Targeting detection
# ---------------------------------------------------------------------------

def detect_targeting(ability: dict) -> str:
    wiki = ability.get("wiki_detail", {})
    targeting = wiki.get("targeting", "").lower()
    affects = wiki.get("affects", "").lower()
    desc = wiki.get("description", "").lower()

    if targeting in ("passive", "auto"):
        if "self" in affects:
            return "self_cast"
        return "self_cast"
    if targeting == "location":
        return "ground_target"
    if targeting == "direction":
        return "direction"
    if targeting == "unit":
        if "allies" in affects or "ally" in affects:
            return "target_ally"
        return "target_enemy"

    # Fallback from description
    if "target direction" in desc or "in a line" in desc:
        return "direction"
    if "target location" in desc or "target area" in desc:
        return "ground_target"
    if "an ally" in desc or "allied" in desc:
        return "target_ally"
    if "all enemies" in desc and ("on the map" in desc or "global" in desc):
        return "global"

    return "target_enemy"

# ---------------------------------------------------------------------------
# Delivery detection
# ---------------------------------------------------------------------------

def detect_delivery(ability: dict) -> dict | None:
    wiki = ability.get("wiki_detail", {})
    desc = (wiki.get("description", "") + " " + ability.get("description", "")).lower()
    projectile = wiki.get("projectile", "").lower()

    if "channel" in desc and ("seconds" in desc or "second" in desc):
        dur_match = re.search(r'channels?\s+(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?)', desc)
        dur = int(float(dur_match.group(1)) * 1000) if dur_match else 2000
        return {"method": "channel", "duration_ms": dur, "tick_interval_ms": 500}

    if any(w in desc for w in ["creates a zone", "creates an area", "creates a field",
                                 "leaves behind", "placed area", "persists for"]):
        return {"method": "zone", "duration_ms": 4000, "tick_interval_ms": 1000}

    if any(w in desc for w in ["places a trap", "lays a trap", "plant a"]):
        return {"method": "trap", "duration_ms": 60000, "trigger_radius": 1.5}

    if any(w in desc for w in ["tethers", "tether", "leashes to"]):
        return {"method": "tether", "max_range": 6.0}

    if any(w in desc for w in ["bounces to", "chains to", "jumps to"]):
        return {"method": "chain", "bounces": 3, "bounce_range": 4.0}

    if projectile == "true" or any(w in desc for w in
            ["fires a", "shoots", "launches", "hurls", "throws", "sends",
             "skillshot", "bolt", "missile"]):
        speed_str = wiki.get("speed", "")
        speed = 10.0
        if speed_str:
            s = _num(speed_str)
            if s and s > 100:
                speed = s / 100.0  # normalize LoL speed to our scale
        return {"method": "projectile", "speed": min(speed, 20.0)}

    return None  # instant

# ---------------------------------------------------------------------------
# Damage type detection
# ---------------------------------------------------------------------------

def detect_damage_type(ability: dict) -> str:
    wiki = ability.get("wiki_detail", {})
    dt = wiki.get("damagetype", "").lower()
    if "true" in dt:
        return "true"
    if "magic" in dt:
        return "magic"
    if "physical" in dt:
        return "physical"
    return "physical"

# ---------------------------------------------------------------------------
# Effect extraction from wiki descriptions
# ---------------------------------------------------------------------------

def normalize_damage(lol_dmg: int) -> int:
    """Normalize LoL damage values to our scale (roughly ÷4)."""
    return max(1, lol_dmg // 4)

def normalize_heal(lol_heal: int) -> int:
    return max(1, lol_heal // 4)

def normalize_shield(lol_shield: int) -> int:
    return max(1, lol_shield // 4)

def extract_effects(ability: dict, slot: str) -> tuple[list[EffectDef], str]:
    """Extract effects from ability descriptions. Returns (effects, ai_hint)."""
    wiki = ability.get("wiki_detail", {})
    desc = wiki.get("description", "") + " " + wiki.get("description2", "")
    leveling = wiki.get("leveling", "") + " " + wiki.get("leveling2", "")
    raw_desc = ability.get("description", "")
    full = (desc + " " + leveling + " " + raw_desc).lower()
    effects = []
    ai_hint = "damage"
    dmg_type = detect_damage_type(ability)

    # --- Damage ---
    has_damage = False
    dmg_val = None
    # Try to extract from leveling {{ap|X to Y}}
    lev_text = leveling
    dmg_match = re.search(r'damage\|?\s*\{\{ap\|(\d+)', lev_text, re.IGNORECASE)
    if dmg_match:
        dmg_val = int(dmg_match.group(1))
    elif "damage" in full or "deals" in full:
        dmg_match2 = re.search(r'(?:deals?|dealing)\s+\{\{ap\|(\d+)', desc, re.IGNORECASE)
        if dmg_match2:
            dmg_val = int(dmg_match2.group(1))
        else:
            # Try raw description numbers
            raw_dmg = re.search(r'dealing\s+(\d+)', raw_desc, re.IGNORECASE)
            if raw_dmg:
                dmg_val = int(raw_dmg.group(1))

    if dmg_val and dmg_val > 0:
        has_damage = True
        eff = EffectDef("damage", {"amount": normalize_damage(dmg_val), "damage_type": dmg_type})

        # Scaling detection
        if "max" in full and "health" in full and "%" in full:
            pct = _float(re.search(r'([\d.]+)%?\s*(?:of\s+)?(?:their\s+)?(?:target.s?\s+)?max', full).group(1)) if re.search(r'([\d.]+)%?\s*(?:of\s+)?(?:their\s+)?(?:target.s?\s+)?max', full) else None
            if pct:
                eff.fields["scaling_stat"] = "target_max_hp"
                eff.fields["scaling_percent"] = pct
        elif "missing" in full and "health" in full and "%" in full:
            pct_m = re.search(r'([\d.]+)%.*missing\s+health', full)
            if pct_m:
                eff.fields["scaling_stat"] = "target_missing_hp"
                eff.fields["scaling_percent"] = float(pct_m.group(1))

        # DoT detection
        if any(w in full for w in ["over", "per second", "each second"]):
            dur_m = re.search(r'(?:over|for)\s+([\d.]+)\s*seconds?', full)
            if dur_m:
                dur_s = float(dur_m.group(1))
                eff.fields["duration_ms"] = int(dur_s * 1000)
                eff.fields["tick_interval_ms"] = 1000
                eff.fields["amount_per_tick"] = max(1, eff.fields["amount"] // max(1, int(dur_s)))
                eff.fields["amount"] = 0  # DoT, not burst

        effects.append(eff)

    # --- Heal ---
    if any(w in full for w in ["heals", "healing", "restores health", "regenerat"]):
        heal_m = re.search(r'heal.*?\{\{ap\|(\d+)', full)
        if not heal_m:
            heal_m = re.search(r'heals?\s+(?:for\s+)?(\d+)', full)
        heal_val = int(heal_m.group(1)) if heal_m else 30
        effects.append(EffectDef("heal", {"amount": normalize_heal(heal_val)}))
        if not has_damage:
            ai_hint = "heal"

    # --- Shield ---
    if any(w in full for w in ["shields", "shield"]) and "spell shield" not in full:
        sh_m = re.search(r'shield.*?\{\{ap\|(\d+)', full)
        if not sh_m:
            sh_m = re.search(r'shield\s+(?:for|of|worth)?\s*(\d+)', full)
        sh_val = int(sh_m.group(1)) if sh_m else 40
        dur_m = re.search(r'shield.*?(?:for|lasts?)\s+([\d.]+)\s*seconds?', full)
        sh_dur = int(float(dur_m.group(1)) * 1000) if dur_m else 3000
        effects.append(EffectDef("shield", {"amount": normalize_shield(sh_val), "duration_ms": sh_dur}))
        if not has_damage:
            ai_hint = "defense"

    # --- CC Effects ---
    cc_found = False

    if any(w in full for w in ["stuns", "stunning", "stunned"]):
        dur_m = re.search(r'stun\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1000
        effects.append(EffectDef("stun", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["roots", "rooting", "rooted", "snares", "snaring", "immobiliz"]):
        dur_m = re.search(r'(?:root|snar|immobil)\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1000
        effects.append(EffectDef("root", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["silences", "silencing", "silenced"]):
        dur_m = re.search(r'silenc\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1500
        effects.append(EffectDef("silence", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["fears", "fearing", "feared", "terrif"]):
        dur_m = re.search(r'(?:fear|terrif)\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1500
        effects.append(EffectDef("fear", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["taunts", "taunting", "taunted"]):
        dur_m = re.search(r'taunt\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1500
        effects.append(EffectDef("taunt", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["suppress", "suppressing"]):
        dur_m = re.search(r'suppress\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1500
        effects.append(EffectDef("suppress", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["charm", "charming", "charmed"]) and "charm" in ability.get("name", "").lower():
        dur_m = re.search(r'charm\w*\s+(?:them\s+)?(?:for\s+|and\s+)?([\d.]+)\s*seconds?', full)
        if not dur_m:
            dur_m = re.search(r'(?:disable|duration)\s*\|?\s*\{\{ap\|([\d.]+)', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 1500
        effects.append(EffectDef("charm", {"duration_ms": dur}))
        cc_found = True

    if any(w in full for w in ["grounds", "grounding", "grounded"]):
        dur_m = re.search(r'ground\w*\s+(?:them\s+)?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 2000
        effects.append(EffectDef("grounded", {"duration_ms": dur}))
        cc_found = True

    # Slow
    if any(w in full for w in ["slows", "slowing", "slowed"]):
        pct_m = re.search(r'slow\w*\s+(?:them\s+)?(?:by\s+)?([\d.]+)%', full)
        pct = float(pct_m.group(1)) / 100 if pct_m else 0.3
        dur_m = re.search(r'slow\w*.*?(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 2000
        effects.append(EffectDef("slow", {"factor": pct, "duration_ms": dur}))

    # Knockback/knockup
    if any(w in full for w in ["knocks back", "knocking back", "knocked back",
                                 "knocks up", "knocking up", "airborne"]):
        effects.append(EffectDef("knockback", {"distance": 2.0}))
        cc_found = True

    # Pull (exclude "pulls back" which is projectile return, not a CC pull)
    if any(w in full for w in ["pulling enemy", "pulls enemy", "pulls the target",
                                 "pulls all enemies", "drags"]):
        effects.append(EffectDef("pull", {"distance": 3.0}))
        cc_found = True

    # --- Mobility ---
    if any(w in full for w in ["dashes", "dashing", "dash", "leaps", "lunges"]):
        eff = EffectDef("dash", {"to_target": "enemy" in full or "target" in full})
        if "blink" in full or "teleport" in full:
            eff.fields["is_blink"] = True
        effects.append(eff)
        if not has_damage and not cc_found:
            ai_hint = "utility"

    # Blink
    elif "blink" in full or "teleport" in full:
        effects.append(EffectDef("dash", {"is_blink": True}))
        if not has_damage:
            ai_hint = "utility"

    # --- Buffs ---
    if any(w in full for w in ["movement speed", "move speed", "bonus movement"]):
        if "gains" in full or "bonus" in full or "increased" in full:
            pct_m = re.search(r'([\d.]+)%\s*(?:bonus\s+)?(?:move|movement)\s*speed', full)
            pct = float(pct_m.group(1)) / 100 if pct_m else 0.3
            dur_m = re.search(r'(?:for|over|decaying\s+over)\s+([\d.]+)\s*seconds?', full)
            dur = int(float(dur_m.group(1)) * 1000) if dur_m else 3000
            effects.append(EffectDef("buff", {"stat": "move_speed", "factor": pct, "duration_ms": dur}))
            if not has_damage and not cc_found:
                ai_hint = "buff"

    if "attack speed" in full and ("gains" in full or "bonus" in full or "increased" in full):
        pct_m = re.search(r'([\d.]+)%\s*(?:bonus\s+)?attack\s*speed', full)
        pct = float(pct_m.group(1)) / 100 if pct_m else 0.3
        effects.append(EffectDef("buff", {"stat": "attack_speed", "factor": pct, "duration_ms": 5000}))
        if not has_damage and not cc_found:
            ai_hint = "buff"

    # --- Stealth ---
    if any(w in full for w in ["stealth", "invisible", "camouflage", "vanish"]):
        dur_m = re.search(r'(?:stealth|invisible|camouflage)\w*\s+(?:for\s+)?([\d.]+)\s*seconds?', full)
        dur = int(float(dur_m.group(1)) * 1000) if dur_m else 3000
        effects.append(EffectDef("stealth", {"duration_ms": dur, "break_on_damage": True, "break_on_ability": True}))
        if not has_damage:
            ai_hint = "utility"

    # --- Lifesteal / Reflect ---
    if "lifesteal" in full or "life steal" in full or "omnivamp" in full:
        pct_m = re.search(r'([\d.]+)%', full)
        pct = float(pct_m.group(1)) / 100 if pct_m else 0.2
        effects.append(EffectDef("lifesteal", {"percent": pct, "duration_ms": 5000}))

    if "reflect" in full and "damage" in full:
        effects.append(EffectDef("reflect", {"percent": 0.3, "duration_ms": 3000}))

    # --- Summon ---
    if any(w in full for w in ["summons", "spawns", "conjures", "raises"]):
        if any(w in full for w in ["soldier", "turret", "minion", "pet", "clone",
                                     "ghoul", "voidling", "tentacle", "plant"]):
            eff = EffectDef("summon", {"template": "minion", "count": 1})
            if any(w in full for w in ["soldier", "sand soldier"]):
                eff.fields["directed"] = True
                eff.fields["template"] = "soldier"
            if "clone" in full:
                eff.fields["clone"] = True
                eff.fields["template"] = "clone"
            effects.append(eff)

    # --- AI hint override ---
    if cc_found and not has_damage:
        ai_hint = "crowd_control"
    elif cc_found and has_damage:
        ai_hint = "damage"  # damage + CC = damage hint
    elif not effects:
        # Fallback: if we extracted nothing, add a basic damage effect
        ai_hint = "damage"

    return effects, ai_hint

# ---------------------------------------------------------------------------
# Special mechanics detection
# ---------------------------------------------------------------------------

def detect_special_mechanics(ability: dict) -> dict:
    """Detect charges, recast, toggle, unstoppable."""
    wiki = ability.get("wiki_detail", {})
    desc = (wiki.get("description", "") + " " + wiki.get("description2", "")).lower()
    tooltip = ability.get("tooltip", "").lower()
    mech = {}

    # Charges/ammo
    if "charges" in desc or "ammo" in tooltip or "stock" in desc or "recharge" in wiki.get("recharge", ""):
        recharge_str = wiki.get("recharge", "")
        recharge_val = _ap_first(recharge_str) or _num(recharge_str)
        if recharge_val:
            mech["max_charges"] = 2
            mech["charge_recharge_ms"] = recharge_val * 1000 if recharge_val < 100 else recharge_val
        else:
            mech["max_charges"] = 2
            mech["charge_recharge_ms"] = 10000

    # Recast
    if "recast" in desc or "can be recast" in desc or "can be cast again" in desc:
        count_m = re.search(r'(\d+)\s+(?:more\s+)?(?:additional\s+)?(?:times?|recasts?)', desc)
        count = int(count_m.group(1)) if count_m else 1
        mech["recast_count"] = count
        window_m = re.search(r'within\s+([\d.]+)\s*seconds?', desc)
        window = int(float(window_m.group(1)) * 1000) if window_m else 10000
        mech["recast_window_ms"] = window

    # Toggle
    if "toggle" in desc or "can be toggled" in desc:
        mech["is_toggle"] = True

    # Unstoppable
    if "unstoppable" in desc:
        mech["unstoppable"] = True

    return mech

# ---------------------------------------------------------------------------
# Area detection
# ---------------------------------------------------------------------------

def detect_area(ability: dict) -> dict | None:
    wiki = ability.get("wiki_detail", {})
    desc = (wiki.get("description", "") + " " + ability.get("description", "")).lower()
    effect_radius = wiki.get("effect radius", "")

    # Circle AoE
    if any(w in desc for w in ["around", "area", "nearby", "in a circle",
                                 "surrounding"]):
        radius = 2.5  # default
        r_m = re.search(r'(\d+)', effect_radius)
        if r_m:
            radius = int(r_m.group(1)) / 100  # normalize LoL radius
            radius = max(1.5, min(radius, 8.0))
        return {"shape": "circle", "radius": radius}

    # Cone
    if any(w in desc for w in ["cone", "in a cone"]):
        return {"shape": "cone", "radius": 4.0, "angle_deg": 60.0}

    # Line
    if any(w in desc for w in ["in a line", "line of", "straight line"]):
        width_str = wiki.get("width", "")
        w_m = re.search(r'(\d+)', width_str)
        width = int(w_m.group(1)) / 100 if w_m else 1.5
        return {"shape": "line", "length": 6.0, "width": max(0.5, min(width, 3.0))}

    return None

# ---------------------------------------------------------------------------
# Passive mapping
# ---------------------------------------------------------------------------

def map_passive(champ: dict) -> PassiveOut | None:
    passive = champ.get("abilities", {}).get("passive", {})
    if not passive:
        return None

    name = passive.get("name", "Passive")
    wiki = passive.get("wiki_detail", {})
    desc = (wiki.get("description", "") + " " + passive.get("description", "")).lower()

    trigger = "on_damage_dealt"
    trigger_extra = {}
    effects = []

    # Detect trigger type
    if any(w in desc for w in ["takes damage", "damaged", "hit by"]):
        trigger = "on_damage_taken"
    elif any(w in desc for w in ["kills", "takedown", "killing"]):
        trigger = "on_kill"
    elif any(w in desc for w in ["every", "periodically", "per second", "each second"]):
        trigger = "periodic"
        dur_m = re.search(r'every\s+([\d.]+)\s*seconds?', desc)
        interval = int(float(dur_m.group(1)) * 1000) if dur_m else 5000
        trigger_extra["interval_ms"] = interval
    elif "basic attack" in desc or "auto attack" in desc:
        trigger = "on_auto_attack"
    elif "ability" in desc and ("cast" in desc or "use" in desc):
        trigger = "on_ability_used"
    elif "death" in desc and ("upon" in desc or "on" in desc):
        trigger = "on_death"

    # Effects from passive
    if any(w in desc for w in ["heal", "regenerat", "restores health"]):
        heal_m = re.search(r'heal\w*\s+(?:for\s+)?(?:\{\{ap\|)?(\d+)', desc)
        val = int(heal_m.group(1)) // 4 if heal_m else 10
        effects.append(EffectDef("heal", {"amount": max(1, val)}))

    if any(w in desc for w in ["shield", "barrier"]) and "spell shield" not in desc:
        effects.append(EffectDef("shield", {"amount": 20, "duration_ms": 3000}))

    if any(w in desc for w in ["bonus damage", "extra damage", "additional damage",
                                 "deals.*damage", "magic damage", "physical damage"]):
        effects.append(EffectDef("damage", {"amount": 10, "damage_type": "physical"}))

    if any(w in desc for w in ["movement speed", "move speed"]):
        effects.append(EffectDef("buff", {"stat": "move_speed", "factor": 0.15, "duration_ms": 3000}))

    if any(w in desc for w in ["attack speed"]):
        effects.append(EffectDef("buff", {"stat": "attack_speed", "factor": 0.2, "duration_ms": 5000}))

    if any(w in desc for w in ["stacks", "stack"]):
        name_m = re.search(r"''(\w[\w\s]+)''", wiki.get("description", ""))
        stack_name = name_m.group(1).lower().replace(" ", "_") if name_m else "passive_stacks"
        effects.append(EffectDef("apply_stacks", {"name": stack_name, "count": 1, "max_stacks": 4}))

    # If we found no effects, add a generic one
    if not effects:
        effects.append(EffectDef("heal", {"amount": 10}))

    return PassiveOut(
        name=name,
        trigger_type=trigger,
        cooldown_ms=5000,
        range=0.0,
        effects=effects,
        trigger_extra=trigger_extra,
    )

# ---------------------------------------------------------------------------
# Ability mapping
# ---------------------------------------------------------------------------

def map_ability(ability: dict, slot: str) -> AbilityOut:
    name = ability.get("name", f"Ability_{slot}")
    wiki = ability.get("wiki_detail", {})

    # Targeting
    targeting = detect_targeting(ability)

    # Range
    range_str = wiki.get("target range", ability.get("range", "0"))
    range_val = _num(str(range_str)) or 0
    range_norm = range_val / 100.0 if range_val > 20 else float(range_val)
    range_norm = max(0.0, min(range_norm, 8.0))

    # If self-cast / self-aoe, range = 0
    if targeting in ("self_cast", "self_aoe"):
        range_norm = 0.0

    # Cooldown — normalize LoL cooldowns to our sim's time scale
    # LoL basic abilities: 4-20s -> keep as-is
    # LoL ults: 60-180s -> compress to 15-30s range
    cd_str = wiki.get("cooldown", ability.get("cooldown", "8"))
    cd_first = _num(str(cd_str)) or 8
    cd_ms = cd_first * 1000 if cd_first < 200 else cd_first
    if cd_ms > 30000:
        # Compress long cooldowns: 60s->18s, 90s->22s, 120s->26s, 180s->30s
        cd_ms = int(14000 + (cd_ms - 30000) * 0.11)
        cd_ms = max(15000, min(cd_ms, 30000))

    # Cast time
    ct_str = wiki.get("cast time", "0")
    if ct_str.lower() == "none" or "attack windup" in ct_str.lower():
        cast_time_ms = 0
    else:
        # Extract from {{fd|0.25}} or plain number
        fd_m = re.search(r'\{\{fd\|([\d.]+)\}\}', ct_str)
        if fd_m:
            cast_time_ms = int(float(fd_m.group(1)) * 1000)
        else:
            ct_val = _float(str(ct_str))
            cast_time_ms = int(ct_val * 1000) if ct_val and ct_val > 0.01 and ct_val < 10 else 0

    # Resource cost
    cost_str = wiki.get("cost", ability.get("cost", "0"))
    cost_first = _num(str(cost_str)) or 0
    resource_cost = cost_first // 5 if cost_first > 0 else 0  # normalize mana

    # Effects
    effects, ai_hint = extract_effects(ability, slot)

    # Area
    area = detect_area(ability)
    if area and effects:
        # Attach area to first damage/CC effect
        for e in effects:
            if e.type in ("damage", "stun", "root", "slow", "fear", "knockback",
                          "silence", "taunt", "charm", "suppress"):
                e.area = area
                break

    # Delivery
    delivery = detect_delivery(ability)

    # Special mechanics
    mechanics = detect_special_mechanics(ability)

    # If no effects extracted, add a default damage
    if not effects:
        effects = [EffectDef("damage", {"amount": 15, "damage_type": detect_damage_type(ability)})]

    out = AbilityOut(
        name=name,
        targeting=targeting,
        range=round(range_norm, 1),
        cooldown_ms=cd_ms,
        cast_time_ms=cast_time_ms,
        ai_hint=ai_hint,
        effects=effects,
        resource_cost=resource_cost,
        max_charges=mechanics.get("max_charges", 0),
        charge_recharge_ms=mechanics.get("charge_recharge_ms", 0),
        recast_count=mechanics.get("recast_count", 0),
        recast_window_ms=mechanics.get("recast_window_ms", 0),
        is_toggle=mechanics.get("is_toggle", False),
        unstoppable=mechanics.get("unstoppable", False),
    )
    return out

# ---------------------------------------------------------------------------
# TOML generation
# ---------------------------------------------------------------------------

def effect_to_toml_lines(eff: EffectDef, prefix: str = "abilities") -> list[str]:
    """Render one effect as TOML lines.

    For prefix="abilities", generates [[abilities.effects]] with sub-tables.
    For prefix="abilities.delivery.on_hit", generates [[abilities.delivery.on_hit]]
    (on_hit is already the array, effects are flattened into it).
    """
    lines = []
    # on_hit/on_arrival are direct ConditionalEffect arrays — no .effects nesting
    is_delivery_array = prefix.endswith(".on_hit") or prefix.endswith(".on_arrival")
    if is_delivery_array:
        array_key = prefix
        sub_prefix = prefix
    else:
        array_key = f"{prefix}.effects"
        sub_prefix = f"{prefix}.effects"

    lines.append(f"[[{array_key}]]")
    lines.append(f'type = "{eff.type}"')
    for k, v in eff.fields.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, float):
            lines.append(f"{k} = {v}")
        elif isinstance(v, int):
            lines.append(f"{k} = {v}")
    if eff.condition:
        lines.append("")
        lines.append(f"[{sub_prefix}.condition]")
        for k, v in eff.condition.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            else:
                lines.append(f"{k} = {v}")
    if eff.area:
        lines.append("")
        lines.append(f"[{sub_prefix}.area]")
        for k, v in eff.area.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, float):
                lines.append(f"{k} = {v}")
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")
    if eff.tags:
        lines.append("")
        lines.append(f"[{sub_prefix}.tags]")
        for k, v in eff.tags.items():
            lines.append(f'{k} = {v}')
    return lines

def ability_to_toml(ab: AbilityOut, delivery: dict | None = None) -> list[str]:
    lines = []
    lines.append("[[abilities]]")
    lines.append(f'name = "{ab.name}"')
    lines.append(f'targeting = "{ab.targeting}"')
    lines.append(f"range = {ab.range}")
    lines.append(f"cooldown_ms = {ab.cooldown_ms}")
    if ab.cast_time_ms > 0:
        lines.append(f"cast_time_ms = {ab.cast_time_ms}")
    lines.append(f'ai_hint = "{ab.ai_hint}"')
    if ab.resource_cost > 0:
        lines.append(f"resource_cost = {ab.resource_cost}")
    if ab.max_charges > 0:
        lines.append(f"max_charges = {ab.max_charges}")
        lines.append(f"charge_recharge_ms = {ab.charge_recharge_ms}")
    if ab.recast_count > 0:
        lines.append(f"recast_count = {ab.recast_count}")
        lines.append(f"recast_window_ms = {ab.recast_window_ms}")
    if ab.is_toggle:
        lines.append("is_toggle = true")
    if ab.unstoppable:
        lines.append("unstoppable = true")

    # Split effects: for projectile delivery, damage/CC go into on_hit,
    # self-targeting effects (dash, buff, stealth, heal, shield) stay on ability
    is_projectile = delivery and delivery.get("method") == "projectile"
    PROJECTILE_EFFECT_TYPES = {
        "damage", "stun", "root", "slow", "fear", "knockback", "silence",
        "taunt", "charm", "suppress", "grounded", "pull", "blind",
        "debuff", "damage_modify", "apply_stacks",
    }
    SELF_EFFECT_TYPES = {
        "dash", "buff", "stealth", "heal", "shield", "lifesteal",
        "reflect", "summon", "command_summons",
    }

    if is_projectile:
        on_hit_effects = []
        self_effects = []
        for eff in ab.effects:
            if eff.type in PROJECTILE_EFFECT_TYPES:
                on_hit_effects.append(eff)
            else:
                self_effects.append(eff)
    else:
        on_hit_effects = []
        self_effects = ab.effects

    # Delivery
    if delivery:
        lines.append("")
        lines.append("[abilities.delivery]")
        for k, v in delivery.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, float):
                lines.append(f"{k} = {v}")
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")

        # on_hit effects for projectile
        for eff in on_hit_effects:
            lines.append("")
            lines.extend(effect_to_toml_lines(eff, "abilities.delivery.on_hit"))

    # Self-targeting effects on the ability itself
    lines.append("")
    for eff in self_effects:
        lines.extend(effect_to_toml_lines(eff, "abilities"))
        lines.append("")

    return lines

def passive_to_toml(ps: PassiveOut) -> list[str]:
    lines = []
    lines.append("[[passives]]")
    lines.append(f'name = "{ps.name}"')
    lines.append(f"cooldown_ms = {ps.cooldown_ms}")
    lines.append(f"range = {ps.range}")
    lines.append("")
    lines.append("[passives.trigger]")
    lines.append(f'type = "{ps.trigger_type}"')
    for k, v in ps.trigger_extra.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    for eff in ps.effects:
        lines.extend(effect_to_toml_lines(eff, "passives"))
        lines.append("")
    return lines

def generate_toml(champ: dict) -> tuple[str, ConversionResult]:
    name = champ["name"]
    stats = get_stats(champ)
    warnings = []

    lines = []
    lines.append("[hero]")
    lines.append(f'name = "{name}"')
    lines.append("")
    lines.append("[stats]")
    lines.append(f"hp = {stats['hp']}")
    lines.append(f"move_speed = {stats['move_speed']}")
    if stats["armor"] > 0:
        lines.append(f"armor = {stats['armor']:.1f}")
    if stats["magic_resist"] > 0:
        lines.append(f"magic_resist = {stats['magic_resist']:.1f}")
    if stats["max_resource"] > 0:
        lines.append(f"resource = {stats['max_resource']}")
        lines.append(f"max_resource = {stats['max_resource']}")
        lines.append(f"resource_regen_per_sec = {stats.get('resource_regen_per_sec', 2.0)}")
    lines.append("")

    lines.append("[attack]")
    lines.append(f"damage = {stats['attack_damage']}")
    lines.append(f"range = {stats['attack_range']}")
    lines.append(f"cooldown = {stats['attack_cooldown']}")
    lines.append("")

    # Abilities (Q, W, E, R)
    abilities_data = champ.get("abilities", {})
    slot_count = 0
    for slot in ["Q", "W", "E", "R"]:
        ab_data = abilities_data.get(slot)
        if not ab_data:
            continue
        ab = map_ability(ab_data, slot)
        delivery = detect_delivery(ab_data)
        lines.append(f"# --- {slot}: {ab.name} ---")
        lines.extend(ability_to_toml(ab, delivery))
        slot_count += 1
        if not ab.effects:
            warnings.append(f"{slot}: no effects extracted")

    # Passive
    passive = map_passive(champ)
    if passive:
        lines.append(f"# --- Passive: {passive.name} ---")
        lines.extend(passive_to_toml(passive))

    # Confidence
    tags = champ.get("tags", [])
    if slot_count < 4:
        confidence = "low"
        warnings.append(f"only {slot_count}/4 abilities mapped")
    elif any(t in tags for t in ["Assassin"]) and len(tags) > 1:
        confidence = "medium"
    else:
        confidence = "high"

    result = ConversionResult(name=name, confidence=confidence, warnings=warnings)
    return "\n".join(lines) + "\n", result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    specific = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not IN_DIR.exists():
        print(f"ERROR: {IN_DIR} does not exist. Run fetch_lol_champions.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(IN_DIR.glob("*.json"))
    if specific:
        json_files = [f for f in json_files if f.stem in specific]

    if not json_files:
        print("No champion files found.")
        sys.exit(1)

    manifest = {}
    success = 0
    fail = 0

    for jf in json_files:
        try:
            with open(jf) as f:
                champ = json.load(f)
            toml_str, result = generate_toml(champ)

            if dry_run:
                print(f"=== {result.name} ({result.confidence}) ===")
                if result.warnings:
                    for w in result.warnings:
                        print(f"  WARN: {w}")
                print(toml_str[:500])
                print("..." if len(toml_str) > 500 else "")
                print()
            else:
                out_file = OUT_DIR / f"{jf.stem}.toml"
                with open(out_file, "w") as f:
                    f.write(toml_str)

            manifest[jf.stem] = {
                "name": result.name,
                "confidence": result.confidence,
                "warnings": result.warnings,
            }
            success += 1
        except Exception as e:
            print(f"ERROR {jf.stem}: {e}")
            manifest[jf.stem] = {"name": jf.stem, "confidence": "error", "warnings": [str(e)]}
            fail += 1

    # Write manifest
    if not dry_run:
        manifest_path = OUT_DIR / "_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Summary
    high = sum(1 for m in manifest.values() if m["confidence"] == "high")
    med = sum(1 for m in manifest.values() if m["confidence"] == "medium")
    low = sum(1 for m in manifest.values() if m["confidence"] == "low")
    err = sum(1 for m in manifest.values() if m["confidence"] == "error")
    warned = sum(1 for m in manifest.values() if m["warnings"])

    print(f"\nConverted {success}/{success + fail} champions")
    print(f"  High confidence: {high}")
    print(f"  Medium confidence: {med}")
    print(f"  Low confidence: {low}")
    print(f"  Errors: {err}")
    print(f"  With warnings: {warned}")
    if not dry_run:
        print(f"\nOutput: {OUT_DIR}/")
        print(f"Manifest: {OUT_DIR}/_manifest.json")


if __name__ == "__main__":
    main()
