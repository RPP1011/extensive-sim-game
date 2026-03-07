#!/usr/bin/env python3
"""Convert LoL hero TOML ability definitions to .ability DSL format."""

import sys
import tomllib
from pathlib import Path


def fmt_duration(ms: int) -> str:
    if ms % 1000 == 0 and ms >= 1000:
        return f"{ms // 1000}s"
    return f"{ms}ms"


def fmt_area(area: dict) -> str:
    shape = area["shape"]
    if shape == "circle":
        return f"circle({area['radius']})"
    elif shape == "cone":
        return f"cone({area['radius']}, {area['angle_deg']})"
    elif shape == "line":
        return f"line({area['length']}, {area.get('width', 1.0)})"
    return ""


def sanitize_name(name: str) -> str:
    """Remove or replace characters not valid in DSL identifiers."""
    import re
    # Replace / | , with nothing (merge words)
    name = name.replace("/", "").replace("|", "").replace(",", "").replace("(", "").replace(")", "")
    # Remove other special chars
    name = name.replace(" ", "").replace("'", "").replace("-", "").replace("!", "").replace(":", "").replace(".", "")
    # If starts with digit, prefix with underscore
    if name and name[0].isdigit():
        name = "_" + name
    return name


def map_targeting(t: str) -> str:
    return {
        "target_enemy": "enemy",
        "target_ally": "ally",
        "self_cast": "self",
        "ground_target": "ground",
        "direction": "direction",
    }.get(t, t)


def map_damage_type_tag(dt: str) -> str:
    return {
        "physical": "PHYSICAL",
        "magic": "MAGIC",
        "true": "TRUE",
        "fire": "FIRE",
        "ice": "ICE",
    }.get(dt, dt.upper())


def fmt_effect(eff: dict, indent: str = "    ") -> list[str]:
    """Convert a single TOML effect dict to DSL lines."""
    lines = []
    etype = eff.get("type", "")
    area = eff.get("area")
    area_str = f" in {fmt_area(area)}" if area else ""

    # Damage with DoT (tick_interval_ms) — emit as regular damage for now
    # The DoT semantics come from the delivery method (zone/channel)
    if etype == "damage":
        amount = eff.get("amount", 0)
        per_tick = eff.get("amount_per_tick", 0)
        dt = eff.get("damage_type")
        tag = f" [{map_damage_type_tag(dt)}: 50]" if dt else ""

        if amount > 0:
            lines.append(f"{indent}damage {amount}{area_str}{tag}")
        elif per_tick > 0:
            # DoT: use per-tick amount
            lines.append(f"{indent}damage {per_tick}{area_str}{tag}")
        else:
            # Zero damage with area — still emit for zone ticks
            if area_str:
                lines.append(f"{indent}damage 0{area_str}{tag}")

    elif etype == "heal":
        amount = eff.get("amount", 0)
        lines.append(f"{indent}heal {amount}")

    elif etype == "shield":
        amount = eff.get("amount", 0)
        dur = eff.get("duration_ms", 5000)
        lines.append(f"{indent}shield {amount} for {fmt_duration(dur)}")

    elif etype == "stun":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}stun {fmt_duration(dur)}{area_str}")

    elif etype == "slow":
        factor = eff.get("factor", 0.3)
        dur = eff.get("duration_ms", 2000)
        lines.append(f"{indent}slow {factor} for {fmt_duration(dur)}{area_str}")

    elif etype == "root":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}root {fmt_duration(dur)}{area_str}")

    elif etype == "silence":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}silence {fmt_duration(dur)}{area_str}")

    elif etype == "fear":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}fear {fmt_duration(dur)}{area_str}")

    elif etype == "taunt":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}taunt {fmt_duration(dur)}{area_str}")

    elif etype == "charm":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}charm {fmt_duration(dur)}{area_str}")

    elif etype == "suppress":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}suppress {fmt_duration(dur)}{area_str}")

    elif etype == "grounded":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}grounded {fmt_duration(dur)}{area_str}")

    elif etype == "knockback":
        dist = eff.get("distance", 2.0)
        lines.append(f"{indent}knockback {dist}{area_str}")

    elif etype == "pull":
        dist = eff.get("distance", 2.0)
        lines.append(f"{indent}pull {dist}{area_str}")

    elif etype == "dash":
        if eff.get("to_target"):
            lines.append(f"{indent}dash to_target")
        elif eff.get("distance"):
            lines.append(f"{indent}dash {eff['distance']}")
        else:
            lines.append(f"{indent}dash to_target")

    elif etype == "buff":
        stat = eff.get("stat", "move_speed")
        factor = eff.get("factor", 0.3)
        dur = eff.get("duration_ms", 3000)
        lines.append(f"{indent}buff {stat} {factor} for {fmt_duration(dur)}{area_str}")

    elif etype == "debuff":
        stat = eff.get("stat", "move_speed")
        factor = eff.get("factor", 0.3)
        dur = eff.get("duration_ms", 3000)
        lines.append(f"{indent}debuff {stat} {factor} for {fmt_duration(dur)}{area_str}")

    elif etype == "apply_stacks":
        name = eff.get("name", "stacks")
        count = eff.get("count", 1)
        mx = eff.get("max_stacks", 4)
        lines.append(f"{indent}apply_stacks \"{name}\" {count} max {mx}")

    elif etype == "stealth":
        dur = eff.get("duration_ms", 3000)
        bod = " break_on_damage" if eff.get("break_on_damage") else ""
        lines.append(f"{indent}stealth for {fmt_duration(dur)}{bod}")

    elif etype == "summon":
        template = eff.get("template", "minion")
        count = eff.get("count", 1)
        count_str = f" x{count}" if count > 1 else ""
        lines.append(f"{indent}summon \"{template}\"{count_str}")

    elif etype == "lifesteal":
        pct = eff.get("percent", 0.2)
        dur = eff.get("duration_ms", 5000)
        lines.append(f"{indent}lifesteal {pct} for {fmt_duration(dur)}")

    elif etype == "reflect":
        pct = eff.get("percent", 0.3)
        dur = eff.get("duration_ms", 3000)
        lines.append(f"{indent}reflect {pct} for {fmt_duration(dur)}")

    elif etype == "polymorph":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}polymorph {fmt_duration(dur)}{area_str}")

    elif etype == "banish":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}banish {fmt_duration(dur)}{area_str}")

    elif etype == "confuse":
        dur = eff.get("duration_ms", 1000)
        lines.append(f"{indent}confuse {fmt_duration(dur)}{area_str}")

    elif etype == "blind":
        chance = eff.get("miss_chance", 0.5)
        dur = eff.get("duration_ms", 2000)
        lines.append(f"{indent}blind {chance} for {fmt_duration(dur)}{area_str}")

    return lines


def fmt_delivery(delivery: dict, effects_in_delivery: list[dict], indent: str = "    ") -> list[str]:
    """Format a delivery block."""
    method = delivery.get("method", "projectile")
    lines = []

    if method == "projectile":
        speed = delivery.get("speed", 10.0)
        props = [f"speed: {speed}"]
        if delivery.get("pierce"):
            props.append("pierce")
        if delivery.get("width"):
            props.append(f"width: {delivery['width']}")
        lines.append(f"{indent}deliver projectile {{ {', '.join(props)} }} {{")

        on_hit = delivery.get("on_hit", [])
        if on_hit:
            lines.append(f"{indent}    on_hit {{")
            for eff in on_hit:
                lines.extend(fmt_effect(eff, indent + "        "))
            lines.append(f"{indent}    }}")

        lines.append(f"{indent}}}")

    elif method == "chain":
        bounces = delivery.get("bounces", 3)
        brange = delivery.get("bounce_range", 4.0)
        props = [f"bounces: {bounces}", f"range: {brange}"]
        if delivery.get("falloff"):
            props.append(f"falloff: {delivery['falloff']}")
        lines.append(f"{indent}deliver chain {{ {', '.join(props)} }} {{")
        lines.append(f"{indent}    on_hit {{")
        # Chain effects come from the ability's top-level effects
        for eff in effects_in_delivery:
            lines.extend(fmt_effect(eff, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    elif method == "zone":
        dur = delivery.get("duration_ms", 4000)
        tick = delivery.get("tick_interval_ms", 1000)
        lines.append(f"{indent}deliver zone {{ duration: {fmt_duration(dur)}, tick: {fmt_duration(tick)} }} {{")
        lines.append(f"{indent}    on_hit {{")
        for eff in effects_in_delivery:
            lines.extend(fmt_effect(eff, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    elif method == "channel":
        dur = delivery.get("duration_ms", 2000)
        tick = delivery.get("tick_interval_ms", 500)
        lines.append(f"{indent}deliver channel {{ duration: {fmt_duration(dur)}, tick: {fmt_duration(tick)} }} {{")
        lines.append(f"{indent}    on_hit {{")
        for eff in effects_in_delivery:
            lines.extend(fmt_effect(eff, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    elif method == "tether":
        max_range = delivery.get("max_range", 6.0)
        props = [f"max_range: {max_range}"]
        if delivery.get("tick_interval_ms"):
            props.append(f"tick: {fmt_duration(delivery['tick_interval_ms'])}")
        lines.append(f"{indent}deliver tether {{ {', '.join(props)} }} {{")
        lines.append(f"{indent}    on_complete {{")
        for eff in effects_in_delivery:
            lines.extend(fmt_effect(eff, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    elif method == "trap":
        dur = delivery.get("duration_ms", 60000)
        trig = delivery.get("trigger_radius", 1.5)
        props = [f"duration: {fmt_duration(dur)}", f"trigger_radius: {trig}"]
        if delivery.get("arm_time_ms"):
            props.append(f"arm_time: {fmt_duration(delivery['arm_time_ms'])}")
        lines.append(f"{indent}deliver trap {{ {', '.join(props)} }} {{")
        lines.append(f"{indent}    on_hit {{")
        for eff in effects_in_delivery:
            lines.extend(fmt_effect(eff, indent + "        "))
        lines.append(f"{indent}    }}")
        lines.append(f"{indent}}}")

    return lines


def convert_ability(ab: dict) -> list[str]:
    """Convert one TOML ability to DSL lines."""
    name = sanitize_name(ab["name"])
    lines = []
    lines.append(f"ability {name} {{")

    # Properties line 1: target, range
    target = map_targeting(ab.get("targeting", "self_cast"))
    props1 = [f"target: {target}"]
    rng = ab.get("range", 0.0)
    if rng > 0:
        props1.append(f"range: {rng}")
    lines.append(f"    {', '.join(props1)}")

    # Properties line 2: cooldown, cast
    props2 = []
    cd = ab.get("cooldown_ms")
    if cd:
        props2.append(f"cooldown: {fmt_duration(cd)}")
    cast = ab.get("cast_time_ms")
    if cast:
        props2.append(f"cast: {fmt_duration(cast)}")
    if props2:
        lines.append(f"    {', '.join(props2)}")

    # Hint
    hint = ab.get("ai_hint")
    if hint:
        lines.append(f"    hint: {hint}")

    # Resource cost
    cost = ab.get("resource_cost")
    if cost:
        lines.append(f"    cost: {cost}")

    # Charges
    charges = ab.get("max_charges")
    if charges:
        lines.append(f"    charges: {charges}")
        recharge = ab.get("charge_recharge_ms")
        if recharge:
            lines.append(f"    recharge: {fmt_duration(recharge)}")

    # Recast
    recast = ab.get("recast_count")
    if recast:
        lines.append(f"    recast: {recast}")
        rw = ab.get("recast_window_ms")
        if rw:
            lines.append(f"    recast_window: {fmt_duration(rw)}")

    lines.append("")

    delivery = ab.get("delivery")
    effects = ab.get("effects", [])

    if delivery:
        method = delivery.get("method", "projectile")
        on_hit = delivery.get("on_hit", [])

        if method == "projectile" and on_hit:
            # Projectile: on_hit effects go inside delivery, top-level effects go after
            lines.extend(fmt_delivery(delivery, [], "    "))
            # Non-delivery effects after
            for eff in effects:
                lines.extend(fmt_effect(eff, "    "))
        elif method in ("chain", "zone", "channel", "tether", "trap"):
            # These methods: top-level effects go inside the delivery block
            # But some effects (dash, buff, heal) might be "extra" effects on the ability
            delivery_effects = []
            extra_effects = []
            for eff in effects:
                et = eff.get("type", "")
                # Damage, heal (for channel), root, stun, slow, knockback go in delivery
                # Buff, dash, stealth, summon, lifesteal are "extra" top-level effects
                if et in ("dash", "buff", "stealth", "lifesteal", "summon", "shield"):
                    extra_effects.append(eff)
                else:
                    delivery_effects.append(eff)
            lines.extend(fmt_delivery(delivery, delivery_effects, "    "))
            for eff in extra_effects:
                lines.extend(fmt_effect(eff, "    "))
        else:
            lines.extend(fmt_delivery(delivery, effects, "    "))
            for eff in effects:
                lines.extend(fmt_effect(eff, "    "))
    else:
        # No delivery — all effects are direct
        for eff in effects:
            lines.extend(fmt_effect(eff, "    "))

    lines.append("}")
    return lines


def map_passive_trigger(trigger: dict) -> str:
    ttype = trigger.get("type", "on_damage_dealt")
    if ttype == "periodic":
        interval = trigger.get("interval_ms", 5000)
        return f"periodic({fmt_duration(interval)})"
    elif ttype == "on_hp_below":
        pct = trigger.get("threshold", 25)
        return f"on_hp_below({pct}%)"
    elif ttype == "on_hp_above":
        pct = trigger.get("threshold", 75)
        return f"on_hp_above({pct}%)"
    elif ttype == "on_ally_damaged":
        rng = trigger.get("range", 5.0)
        return f"on_ally_damaged(range: {rng})"
    elif ttype == "on_ally_killed":
        rng = trigger.get("range", 5.0)
        return f"on_ally_killed(range: {rng})"
    elif ttype == "on_stack_reached":
        name = trigger.get("name", "stacks")
        count = trigger.get("count", 3)
        return f'on_stack_reached("{name}", {count})'
    return ttype


def convert_passive(p: dict) -> list[str]:
    name = sanitize_name(p["name"])
    lines = []
    lines.append(f"passive {name} {{")

    trigger = p.get("trigger", {})
    lines.append(f"    trigger: {map_passive_trigger(trigger)}")

    cd = p.get("cooldown_ms")
    if cd:
        lines.append(f"    cooldown: {fmt_duration(cd)}")

    rng = p.get("range")
    if rng and rng > 0:
        lines.append(f"    range: {rng}")

    lines.append("")

    effects = p.get("effects", [])
    for eff in effects:
        lines.extend(fmt_effect(eff, "    "))

    lines.append("}")
    return lines


def convert_hero(toml_path: Path) -> str:
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    hero_name = data.get("hero", {}).get("name", toml_path.stem)
    out = [f"// {hero_name} abilities", ""]

    abilities = data.get("abilities", [])
    for ab in abilities:
        out.extend(convert_ability(ab))
        out.append("")

    passives = data.get("passives", [])
    for p in passives:
        out.extend(convert_passive(p))
        out.append("")

    # Remove trailing blank line
    while out and out[-1] == "":
        out.pop()
    out.append("")

    return "\n".join(out)


def main():
    lol_dir = Path("assets/lol_heroes")
    toml_files = sorted(lol_dir.glob("*.toml"))

    if not toml_files:
        print("No TOML files found in assets/lol_heroes/")
        sys.exit(1)

    converted = 0
    errors = []
    for tf in toml_files:
        ability_path = tf.with_suffix(".ability")
        try:
            result = convert_hero(tf)
            ability_path.write_text(result)
            converted += 1
        except Exception as e:
            errors.append((tf.name, str(e)))
            print(f"ERROR: {tf.name}: {e}")

    print(f"Converted {converted}/{len(toml_files)} heroes")
    if errors:
        print(f"Errors: {len(errors)}")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
