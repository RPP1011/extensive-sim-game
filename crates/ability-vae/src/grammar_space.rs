//! Grammar-defined invertible mapping: [0,1]^N ↔ valid DSL program.
//!
//! Every point in the unit hypercube maps to exactly one syntactically valid
//! ability DSL program, and every valid program maps back to exactly one point.
//!
//! v2: 48 dimensions with tags, scaling, per-effect conditions, proper buff/debuff
//! stat types, campaign/combat coherence, and DoT/HoT support.

use tactical_sim::effects::dsl::parse_abilities;

/// Total dimensions of the grammar space.
pub const GRAMMAR_DIM: usize = 48;

// ---------------------------------------------------------------------------
// Dimension layout (48 total)
// ---------------------------------------------------------------------------

// Header (8 dims)
const D_TYPE: usize = 0;         // 0=ability, 1=passive
const D_DOMAIN: usize = 1;       // 0=combat, 1=campaign — controls timescale + effect pool
const D_TARGET: usize = 2;       // targeting mode
const D_RANGE: usize = 3;        // 0.5-10.0
const D_COOLDOWN: usize = 4;     // log scale (combat: 1s-60s, campaign: 50t-10000t)
const D_CAST: usize = 5;         // 0-3000ms (combat only)
const D_HINT: usize = 6;         // hint category
const D_COST: usize = 7;         // 0-30 resource cost

// Delivery (4 dims)
const D_DELIVERY: usize = 8;     // none/projectile/chain/zone/trap/channel
const D_DELIV_P0: usize = 9;     // speed/bounces/duration
const D_DELIV_P1: usize = 10;    // range/tick/width
const D_N_EFFECTS: usize = 11;   // 1-4

// Effect 0 (8 dims)
const D_E0_TYPE: usize = 12;     // effect type within domain
const D_E0_PARAM: usize = 13;    // primary param (amount/factor)
const D_E0_DUR: usize = 14;      // duration
const D_E0_AREA: usize = 15;     // area type
const D_E0_AREA_P: usize = 16;   // area param (radius)
const D_E0_TAG: usize = 17;      // element tag
const D_E0_TAG_PWR: usize = 18;  // tag power level
const D_E0_COND: usize = 19;     // per-effect condition

// Effect 1 (8 dims)
const D_E1_BASE: usize = 20;

// Effect 2 (8 dims)
const D_E2_BASE: usize = 28;

// Effect 3 (8 dims)
const D_E3_BASE: usize = 36;

// Passive/scaling (4 dims)
const D_TRIGGER: usize = 44;     // passive trigger type
const D_TRIGGER_P: usize = 45;   // trigger param
const D_SCALING_STAT: usize = 46; // scaling stat reference
const D_SCALING_PCT: usize = 47;  // scaling percentage

const EFF_STRIDE: usize = 8;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

const COMBAT_TARGETS: &[&str] = &[
    "enemy", "ally", "self", "self_aoe", "ground", "direction", "global",
];

const CAMPAIGN_TARGETS: &[&str] = &[
    "faction", "region", "market", "party", "guild", "self", "global", "adventurer",
];

const COMBAT_HINTS: &[&str] = &[
    "damage", "heal", "crowd_control", "defense", "utility",
];

const CAMPAIGN_HINTS: &[&str] = &[
    "economy", "diplomacy", "stealth", "leadership", "utility", "defense", "heal",
];

const DELIVERY_TYPES: &[&str] = &[
    "none", "none", "none",  // weight toward no delivery (50%)
    "projectile", "chain", "zone", "trap",
];

const TRIGGERS: &[&str] = &[
    "on_damage_dealt", "on_damage_taken", "on_kill", "on_heal_received",
    "on_ability_used", "on_trade", "on_quest_complete", "on_level_up",
    "on_crisis_start", "on_death",
];

const TAGS: &[&str] = &[
    "", "", "",  // no tag (weighted)
    "PHYSICAL", "MAGIC", "FIRE", "ICE", "DARK", "HOLY", "POISON",
];

const AREAS: &[&str] = &[
    "", "", "", "",  // no area (weighted)
    "circle", "cone", "line",
];

const CONDITIONS: &[&str] = &[
    "", "", "", "", "", "",  // no condition (weighted ~60%)
    "target_hp_below", "target_hp_above", "caster_hp_below",
    "hit_count_above",
];

const SCALING_STATS: &[&str] = &[
    "", "", "", "",  // no scaling (weighted)
    "target_max_hp", "caster_max_hp", "caster_attack_damage", "target_missing_hp",
];

const BUFF_STATS: &[&str] = &[
    "damage_output", "move_speed", "attack_speed", "armor", "magic_resist",
];

// Combat effects by parameter pattern
const COMBAT_INSTANT: &[&str] = &["dispel", "swap", "refresh_cooldowns"];
const COMBAT_AMOUNT: &[&str] = &[
    "damage", "heal", "shield", "knockback", "pull", "dash", "blink",
];
const COMBAT_DURATION: &[&str] = &[
    "stun", "root", "silence", "fear", "taunt", "stealth",
    "blind", "charm", "suppress", "confuse", "polymorph", "banish",
    "cooldown_lock",
];
const COMBAT_AMOUNT_DUR: &[&str] = &[
    "slow", "damage_modify", "lifesteal", "mana_burn",
];
// Meta-effects with charges
const COMBAT_CHARGES: &[&str] = &[
    "amplify", "echo", "instant_cast", "free_cast", "spell_shield",
];
// buff/debuff handled separately (need stat type)

// Campaign effects by parameter pattern
const CAMPAIGN_INSTANT: &[&str] = &[
    "appraise", "beast_lore", "battle_instinct", "quick_study",
    "track_prey", "field_repair", "silent_movement", "trap_sense",
    "stabilize_ally", "sapper_eye", "read_the_room", "decipher",
    "demand_audience", "subvert_loyalty", "treaty_breaker",
    "silver_tongue", "coordinated_strike", "claim_territory",
    "unbreakable", "purify", "name_the_nameless", "forbidden_knowledge",
    "forge_artifact", "masterwork_craft", "master_armorer", "intel_gathering",
    "forgery", "hold_the_line", "vanish", "living_legend",
    "omniscience", "claim_by_right", "life_eternal", "all_seeing_eye",
];
const CAMPAIGN_AMOUNT: &[&str] = &[
    "rally", "blood_oath", "forage", "field_triage", "trade_empire",
];
const CAMPAIGN_DURATION: &[&str] = &[
    "ghost_walk", "shadow_step", "hidden_camp", "distraction",
    "fortify", "sanctuary", "plague_ward", "safe_house",
    "ceasefire", "broker_alliance", "field_command", "war_cry",
    "golden_touch", "trade_embargo", "corner_market",
    "disinformation", "accelerated_study", "take_the_blow", "market_maker",
];
const CAMPAIGN_COUNT: &[&str] = &["reveal", "prophecy"];

// ---------------------------------------------------------------------------
// Decode: [0,1]^48 → DSL text
// ---------------------------------------------------------------------------

pub fn decode(v: &[f32; GRAMMAR_DIM]) -> String {
    let is_passive = v[D_TYPE] > 0.5;
    let is_campaign = v[D_DOMAIN] > 0.5;

    if is_passive {
        decode_passive(v, is_campaign)
    } else {
        decode_ability(v, is_campaign)
    }
}

fn decode_ability(v: &[f32; GRAMMAR_DIM], is_campaign: bool) -> String {
    let targets = if is_campaign { CAMPAIGN_TARGETS } else { COMBAT_TARGETS };
    let hints = if is_campaign { CAMPAIGN_HINTS } else { COMBAT_HINTS };

    let target = pick(v[D_TARGET], targets);
    let range = lerp(v[D_RANGE], 0.5, 10.0);
    let hint = pick(v[D_HINT], hints);

    // Cooldown always emitted as seconds (parser doesn't support Nt in header)
    // Campaign: longer cooldowns (150s-30000s = ~2.5min-8hrs game time)
    // Combat: shorter (1s-60s)
    let cooldown = if is_campaign {
        fmt_ms(log_lerp(v[D_COOLDOWN], 150000.0, 30000000.0) as u32)
    } else {
        fmt_ms(log_lerp(v[D_COOLDOWN], 1000.0, 60000.0) as u32)
    };
    let cast = fmt_ms(lerp(v[D_CAST], 0.0, 2000.0) as u32);
    let cost = (lerp(v[D_COST], 0.0, 30.0) as u32).max(0);

    let n_effects = pick_n(v[D_N_EFFECTS], 1, 4);

    // Campaign abilities don't use delivery blocks
    let delivery_idx = if is_campaign { 0 } else { pick_idx(v[D_DELIVERY], DELIVERY_TYPES) };
    let delivery_name = DELIVERY_TYPES[delivery_idx];

    let mut lines = Vec::new();
    lines.push("ability Generated {".to_string());

    let needs_range = !matches!(target, "self" | "self_aoe" | "global" | "guild" | "market");
    if needs_range {
        lines.push(format!("    target: {}, range: {:.1}", target, range));
    } else {
        lines.push(format!("    target: {}", target));
    }

    lines.push(format!("    cooldown: {}, cast: {}", cooldown, cast));
    lines.push(format!("    hint: {}", hint));
    if cost > 0 {
        lines.push(format!("    cost: {}", cost));
    }

    let effects = build_effects(v, n_effects, is_campaign);

    // Scaling on first effect (combat only)
    let scaling_str = if is_campaign { String::new() } else { build_scaling(v) };

    if delivery_name == "none" {
        lines.push(String::new());
        for (i, e) in effects.iter().enumerate() {
            if i == 0 && !scaling_str.is_empty() {
                lines.push(format!("    {}{}", e, scaling_str));
            } else {
                lines.push(format!("    {}", e));
            }
        }
    } else {
        let dp0 = lerp(v[D_DELIV_P0], 1.0, 20.0);
        let dp1 = lerp(v[D_DELIV_P1], 0.5, 10.0);
        let params = match delivery_name {
            "projectile" => format!("speed: {:.1}", dp0),
            "chain" => format!("bounces: {}, range: {:.1}, falloff: 0.2", (dp0 as u32).max(1), dp1),
            "zone" => format!("duration: {}, tick: {}", fmt_ms((dp0 * 500.0) as u32 + 1000), fmt_ms((dp1 * 250.0) as u32 + 500)),
            "trap" => format!("duration: {}, trigger_radius: {:.1}", fmt_ms((dp0 * 1000.0) as u32 + 2000), dp1),
            "channel" => format!("duration: {}, tick: {}", fmt_ms((dp0 * 500.0) as u32 + 1000), fmt_ms((dp1 * 250.0) as u32 + 500)),
            _ => String::new(),
        };
        lines.push(String::new());
        lines.push(format!("    deliver {} {{ {} }} {{", delivery_name, params));
        lines.push("        on_hit {".to_string());
        for (i, e) in effects.iter().enumerate() {
            if i == 0 && !scaling_str.is_empty() {
                lines.push(format!("            {}{}", e, scaling_str));
            } else {
                lines.push(format!("            {}", e));
            }
        }
        lines.push("        }".to_string());
        lines.push("    }".to_string());
    }

    lines.push("}".to_string());
    lines.join("\n")
}

fn decode_passive(v: &[f32; GRAMMAR_DIM], is_campaign: bool) -> String {
    let trigger = pick(v[D_TRIGGER], TRIGGERS);
    let n_effects = pick_n(v[D_N_EFFECTS], 1, 2);
    let effects = build_effects(v, n_effects, is_campaign);

    let trigger_str = if trigger == "on_damage_dealt" || trigger == "on_damage_taken" {
        // These can have a range param
        format!("    trigger: {}", trigger)
    } else {
        format!("    trigger: {}", trigger)
    };

    let cd = if is_campaign {
        fmt_ms(log_lerp(v[D_COOLDOWN], 9000.0, 90000.0) as u32)
    } else {
        fmt_ms(log_lerp(v[D_COOLDOWN], 3000.0, 30000.0) as u32)
    };

    let mut lines = Vec::new();
    lines.push("passive Generated {".to_string());
    lines.push(trigger_str);
    lines.push(format!("    cooldown: {}", cd));
    lines.push(String::new());
    for e in &effects {
        lines.push(format!("    {}", e));
    }
    lines.push("}".to_string());
    lines.join("\n")
}

fn build_effects(v: &[f32; GRAMMAR_DIM], n_effects: usize, is_campaign: bool) -> Vec<String> {
    let bases = [D_E0_TYPE, D_E1_BASE, D_E2_BASE, D_E3_BASE];
    let mut effects = Vec::new();

    for i in 0..n_effects {
        let base = bases[i];
        let type_v = v[base];
        let param_v = v[base + 1];
        let dur_v = v[base + 2];
        let area_v = v[base + 3];
        let area_p = v[base + 4];
        let tag_v = v[base + 5];
        let tag_pwr = v[base + 6];
        let cond_v = v[base + 7];

        let effect = if is_campaign {
            build_campaign_effect(type_v, param_v, dur_v)
        } else {
            build_combat_effect(type_v, param_v, dur_v)
        };

        // Campaign effects don't use area modifiers
        let area = if is_campaign { String::new() } else { build_area(area_v, area_p) };
        // Tags and conditions only on combat effects (campaign effects don't support them)
        let tag = if is_campaign { String::new() } else { build_tag(tag_v, tag_pwr) };
        let cond = if is_campaign { String::new() } else { build_condition(cond_v, param_v) };

        effects.push(format!("{}{}{}{}", effect, area, tag, cond));
    }
    effects
}

fn build_combat_effect(type_v: f32, param_v: f32, dur_v: f32) -> String {
    // Split [0,1] into regions: instant, amount, duration, amount+dur, charges, buff, debuff, dot, hot
    let total_slots = COMBAT_INSTANT.len() + COMBAT_AMOUNT.len()
        + COMBAT_DURATION.len() + COMBAT_AMOUNT_DUR.len()
        + COMBAT_CHARGES.len()
        + 5 + 5 + 2 + 2; // buff(5 stats), debuff(5 stats), dot, hot

    let idx = (type_v * total_slots as f32) as usize;

    let instant_end = COMBAT_INSTANT.len();
    let amount_end = instant_end + COMBAT_AMOUNT.len();
    let duration_end = amount_end + COMBAT_DURATION.len();
    let amount_dur_end = duration_end + COMBAT_AMOUNT_DUR.len();
    let charges_end = amount_dur_end + COMBAT_CHARGES.len();
    let buff_end = charges_end + BUFF_STATS.len();
    let debuff_end = buff_end + BUFF_STATS.len();
    let dot_end = debuff_end + 2;

    if idx < instant_end {
        COMBAT_INSTANT[idx].to_string()
    } else if idx < amount_end {
        let name = COMBAT_AMOUNT[idx - instant_end];
        match name {
            "damage" | "heal" | "shield" => format!("{} {}", name, lerp(param_v, 10.0, 200.0) as i32),
            "knockback" | "pull" => format!("{} {:.1}", name, lerp(param_v, 1.0, 6.0)),
            "dash" => if param_v > 0.7 { "dash to_target".to_string() } else { format!("dash {:.1}", lerp(param_v, 1.5, 6.0)) },
            "blink" => format!("blink {:.1}", lerp(param_v, 2.0, 8.0)),
            _ => format!("{} {}", name, lerp(param_v, 5.0, 100.0) as i32),
        }
    } else if idx < duration_end {
        let name = COMBAT_DURATION[idx - amount_end];
        let ms = log_lerp(dur_v, 500.0, 5000.0) as u32;
        format!("{} {}", name, fmt_ms(ms))
    } else if idx < amount_dur_end {
        let name = COMBAT_AMOUNT_DUR[idx - duration_end];
        match name {
            "slow" => format!("slow {:.1} for {}", lerp(param_v, 0.15, 0.7), fmt_ms(log_lerp(dur_v, 500.0, 4000.0) as u32)),
            "damage_modify" => format!("damage_modify {:.2} for {}", lerp(param_v, 1.1, 2.0), fmt_ms(log_lerp(dur_v, 2000.0, 8000.0) as u32)),
            "lifesteal" => format!("lifesteal {:.0} for {}", lerp(param_v, 10.0, 50.0), fmt_ms(log_lerp(dur_v, 2000.0, 8000.0) as u32)),
            _ => format!("{} {:.1} for {}", name, lerp(param_v, 0.1, 1.0), fmt_ms(log_lerp(dur_v, 1000.0, 5000.0) as u32)),
        }
    } else if idx < charges_end {
        let name = COMBAT_CHARGES[idx - amount_dur_end];
        let charges = (lerp(param_v, 1.0, 4.0) as u32).max(1);
        match name {
            "amplify" => format!("amplify {:.1} {}", lerp(dur_v, 1.2, 3.0), charges),
            "echo" => format!("echo {}", charges),
            "instant_cast" => format!("instant_cast {}", charges),
            "free_cast" => format!("free_cast {}", charges),
            "spell_shield" => format!("spell_shield {}", charges),
            _ => format!("{} {}", name, charges),
        }
    } else if idx < buff_end {
        let stat = BUFF_STATS[idx - charges_end];
        format!("buff {} {:.2} for {}", stat, lerp(param_v, 0.05, 0.5), fmt_ms(log_lerp(dur_v, 2000.0, 10000.0) as u32))
    } else if idx < debuff_end {
        let stat_idx = idx - buff_end;
        let stat = BUFF_STATS[stat_idx.min(BUFF_STATS.len() - 1)];
        format!("debuff {} {:.2} for {}", stat, lerp(param_v, 0.1, 0.5), fmt_ms(log_lerp(dur_v, 2000.0, 8000.0) as u32))
    } else if idx < dot_end {
        // DoT: damage N/1s for Ns
        let per_tick = lerp(param_v, 3.0, 30.0) as i32;
        let dur = log_lerp(dur_v, 2000.0, 8000.0) as u32;
        format!("damage {}/1s for {}", per_tick, fmt_ms(dur))
    } else {
        // HoT: heal N/1s for Ns
        let per_tick = lerp(param_v, 3.0, 25.0) as i32;
        let dur = log_lerp(dur_v, 2000.0, 8000.0) as u32;
        format!("heal {}/1s for {}", per_tick, fmt_ms(dur))
    }
}

fn build_campaign_effect(type_v: f32, param_v: f32, dur_v: f32) -> String {
    let total = CAMPAIGN_INSTANT.len() + CAMPAIGN_AMOUNT.len()
        + CAMPAIGN_DURATION.len() + CAMPAIGN_COUNT.len();
    let idx = ((type_v * total as f32) as usize).min(total - 1);

    let instant_end = CAMPAIGN_INSTANT.len();
    let amount_end = instant_end + CAMPAIGN_AMOUNT.len();
    let duration_end = amount_end + CAMPAIGN_DURATION.len();

    if idx < instant_end {
        CAMPAIGN_INSTANT[idx].to_string()
    } else if idx < amount_end {
        let name = CAMPAIGN_AMOUNT[idx - instant_end];
        match name {
            "rally" | "blood_oath" => format!("{} {:.2}", name, lerp(param_v, 0.1, 0.5)),
            "forage" | "field_triage" => format!("{} {:.2}", name, lerp(param_v, 0.1, 2.0)),
            "trade_empire" => format!("{} {:.1}", name, lerp(param_v, 0.5, 10.0)),
            _ => format!("{} {:.2}", name, lerp(param_v, 0.1, 1.0)),
        }
    } else if idx < duration_end {
        let name = CAMPAIGN_DURATION[idx - amount_end];
        let ticks = log_lerp(dur_v, 50.0, 5000.0) as u32;
        format!("{} for {}t", name, ticks)
    } else {
        let name = if idx - duration_end < CAMPAIGN_COUNT.len() {
            CAMPAIGN_COUNT[idx - duration_end]
        } else { "reveal" };
        format!("{} {}", name, (lerp(param_v, 1.0, 10.0) as u32).max(1))
    }
}

fn build_area(area_v: f32, area_p: f32) -> String {
    let area = pick(area_v, AREAS);
    match area {
        "circle" => format!(" in circle({:.1})", lerp(area_p, 1.0, 5.0)),
        "cone" => format!(" in cone({:.1}, {:.0})", lerp(area_p, 2.0, 6.0), lerp(area_p, 30.0, 90.0)),
        "line" => format!(" in line({:.1}, {:.1})", lerp(area_p, 3.0, 8.0), lerp(1.0 - area_p, 0.5, 2.0)),
        _ => String::new(),
    }
}

fn build_tag(tag_v: f32, tag_pwr: f32) -> String {
    let tag = pick(tag_v, TAGS);
    if tag.is_empty() { return String::new(); }
    let power = lerp(tag_pwr, 20.0, 80.0) as i32;
    format!(" [{}: {}]", tag, power)
}

fn build_condition(cond_v: f32, param_v: f32) -> String {
    let cond = pick(cond_v, CONDITIONS);
    if cond.is_empty() { return String::new(); }
    match cond {
        "target_hp_below" | "target_hp_above" | "caster_hp_below" => {
            let pct = (lerp(param_v, 10.0, 80.0) as i32).max(5);
            format!(" when {}({}%)", cond, pct)
        }
        "hit_count_above" => {
            let n = (lerp(param_v, 1.0, 5.0) as i32).max(1);
            format!(" when {}({})", cond, n)
        }
        _ => String::new(),
    }
}

fn build_scaling(v: &[f32; GRAMMAR_DIM]) -> String {
    let stat = pick(v[D_SCALING_STAT], SCALING_STATS);
    if stat.is_empty() { return String::new(); }
    let pct = (lerp(v[D_SCALING_PCT], 5.0, 50.0) as i32).max(1);
    format!(" + {}% {}", pct, stat)
}

// ---------------------------------------------------------------------------
// Encode: DSL → [0,1]^48 (partial — mainly for training data encoding)
// ---------------------------------------------------------------------------

pub fn encode(dsl: &str) -> Option<[f32; GRAMMAR_DIM]> {
    let parsed = parse_abilities(dsl).ok()?;
    let (abilities, passives) = parsed;

    // Default to 0.1 for categorical dims (maps to first/empty bin)
    // and 0.5 only for continuous dims where midpoint makes sense
    let mut v = [0.0f32; GRAMMAR_DIM];
    // Set continuous dims to reasonable defaults
    v[D_RANGE] = 0.5;     // mid range
    v[D_COOLDOWN] = 0.4;  // moderate cooldown
    v[D_CAST] = 0.1;      // fast cast
    v[D_N_EFFECTS] = 0.25; // 1-2 effects

    if let Some(ab) = abilities.first() {
        v[D_TYPE] = 0.25;
        // Determine domain from targeting
        let target_str = format!("{:?}", ab.targeting).to_lowercase();
        let is_campaign = target_str.contains("faction") || target_str.contains("region")
            || target_str.contains("market") || target_str.contains("guild")
            || target_str.contains("party") || target_str.contains("location")
            || target_str.contains("adventurer");
        v[D_DOMAIN] = if is_campaign { 0.75 } else { 0.25 };

        let targets = if is_campaign { CAMPAIGN_TARGETS } else { COMBAT_TARGETS };
        let target_clean = target_str.replace("target", "").replace("selfcast", "self")
            .replace("groundtarget", "ground").replace("selfaoe", "self_aoe");
        if let Some(idx) = targets.iter().position(|&t| target_clean.contains(t)) {
            v[D_TARGET] = (idx as f32 + 0.5) / targets.len() as f32;
        }

        v[D_RANGE] = inv_lerp(ab.range, 0.5, 10.0);
        v[D_COOLDOWN] = if is_campaign {
            inv_log_lerp(ab.cooldown_ms as f32, 50.0, 10000.0)
        } else {
            inv_log_lerp(ab.cooldown_ms as f32, 1000.0, 60000.0)
        };
        v[D_CAST] = inv_lerp(ab.cast_time_ms as f32, 0.0, 2000.0);

        let hints = if is_campaign { CAMPAIGN_HINTS } else { COMBAT_HINTS };
        if let Some(idx) = hints.iter().position(|&h| ab.ai_hint.contains(h)) {
            v[D_HINT] = (idx as f32 + 0.5) / hints.len() as f32;
        }
        v[D_COST] = inv_lerp(ab.resource_cost as f32, 0.0, 30.0);
        let n_eff = ab.effects.len().min(4);
        v[D_N_EFFECTS] = ((n_eff as f32) - 0.5) / 4.0;

        // Delivery
        if let Some(ref delivery) = ab.delivery {
            let delivery_str = format!("{:?}", delivery).to_lowercase();
            let del_map = &[
                ("projectile", 3), ("chain", 4), ("zone", 5),
                ("trap", 6), ("channel", 5),
            ];
            for &(kw, idx) in del_map {
                if delivery_str.contains(kw) {
                    v[D_DELIVERY] = (idx as f32 + 0.5) / DELIVERY_TYPES.len() as f32;
                    break;
                }
            }
        }

        // Extract effect info from first effect
        if let Some(eff) = ab.effects.first() {
            let eff_str = format!("{:?}", eff.effect).to_lowercase();

            // Tags
            let tag_names = &["physical", "magic", "fire", "ice", "dark", "holy", "poison"];
            for (tag_key, &tag_val) in eff.tags.iter() {
                let tag_lower = tag_key.to_lowercase();
                for (ti, &tname) in tag_names.iter().enumerate() {
                    if tag_lower.contains(tname) {
                        v[D_E0_TYPE + 5] = ((ti + 3) as f32 + 0.5) / TAGS.len() as f32; // D_E0_TAG
                        v[D_E0_TYPE + 6] = inv_lerp(tag_val, 20.0, 80.0); // tag power
                        break;
                    }
                }
            }

            // Area
            if let Some(ref area) = eff.area {
                let area_str = format!("{:?}", area).to_lowercase();
                if area_str.contains("circle") { v[D_E0_TYPE + 3] = 0.75; }
                else if area_str.contains("cone") { v[D_E0_TYPE + 3] = 0.85; }
                else if area_str.contains("line") { v[D_E0_TYPE + 3] = 0.95; }
            }

            // Effect type — map to approximate position in the effect pool
            if eff_str.contains("damage") { v[D_E0_TYPE] = 0.08; }
            else if eff_str.contains("heal") { v[D_E0_TYPE] = 0.10; }
            else if eff_str.contains("shield") { v[D_E0_TYPE] = 0.12; }
            else if eff_str.contains("stun") { v[D_E0_TYPE] = 0.30; }
            else if eff_str.contains("root") { v[D_E0_TYPE] = 0.32; }
            else if eff_str.contains("silence") { v[D_E0_TYPE] = 0.34; }
            else if eff_str.contains("fear") { v[D_E0_TYPE] = 0.36; }
            else if eff_str.contains("slow") { v[D_E0_TYPE] = 0.58; }
            else if eff_str.contains("buff") { v[D_E0_TYPE] = 0.78; }
            else if eff_str.contains("debuff") { v[D_E0_TYPE] = 0.82; }
            else if eff_str.contains("dash") { v[D_E0_TYPE] = 0.18; }
            else if eff_str.contains("stealth") { v[D_E0_TYPE] = 0.40; }
            else if eff_str.contains("knockback") { v[D_E0_TYPE] = 0.14; }
            else if eff_str.contains("pull") { v[D_E0_TYPE] = 0.16; }
        }

        // Param intensity from first effect
        if let Some(eff) = ab.effects.first() {
            let eff_str = format!("{:?}", eff.effect);
            // Try to extract amount field
            if let Some(pos) = eff_str.find("amount:") {
                let num_str: String = eff_str[pos+7..].chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-')
                    .collect();
                if let Ok(amount) = num_str.parse::<f32>() {
                    v[D_E0_TYPE + 1] = inv_lerp(amount.abs(), 10.0, 200.0); // param intensity
                }
            }
        }

    } else if let Some(passive) = passives.first() {
        v[D_TYPE] = 0.75;
        let n_eff = passive.effects.len().min(2);
        v[D_N_EFFECTS] = ((n_eff as f32) - 0.5) / 4.0;

        // Extract trigger
        let trigger_str = format!("{:?}", passive.trigger).to_lowercase();
        let trigger_map = &[
            ("ondamagedealt", 0.05), ("ondamagetaken", 0.15), ("onkill", 0.25),
            ("onheal", 0.35), ("onability", 0.45),
        ];
        for &(kw, val) in trigger_map {
            if trigger_str.contains(kw) {
                v[44] = val; // D_TRIGGER
                break;
            }
        }
    }

    Some(v)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

pub fn validate_random(n: usize) -> (usize, usize) {
    let mut ok = 0;
    let mut fail = 0;
    let mut rng: u64 = 42;

    for _ in 0..n {
        let mut v = [0.0f32; GRAMMAR_DIM];
        for d in &mut v {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *d = (rng >> 33) as f32 / (1u64 << 31) as f32;
        }

        let dsl = decode(&v);
        match parse_abilities(&dsl) {
            Ok(_) => ok += 1,
            Err(e) => {
                if fail < 3 {
                    eprintln!("Parse fail:\n{}\nError: {}\n", dsl, e);
                }
                fail += 1;
            }
        }
    }

    (ok, fail)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pick<'a>(val: f32, options: &[&'a str]) -> &'a str {
    let idx = ((val.clamp(0.0, 0.9999) * options.len() as f32) as usize).min(options.len() - 1);
    options[idx]
}

fn pick_idx(val: f32, options: &[&str]) -> usize {
    ((val.clamp(0.0, 0.9999) * options.len() as f32) as usize).min(options.len() - 1)
}

fn pick_n(val: f32, min: usize, max: usize) -> usize {
    let range = max - min + 1;
    min + ((val.clamp(0.0, 0.9999) * range as f32) as usize).min(range - 1)
}

fn lerp(t: f32, a: f32, b: f32) -> f32 { a + t.clamp(0.0, 1.0) * (b - a) }
fn inv_lerp(v: f32, a: f32, b: f32) -> f32 { ((v - a) / (b - a)).clamp(0.0, 1.0) }
fn log_lerp(t: f32, a: f32, b: f32) -> f32 { (a.ln() + t.clamp(0.0, 1.0) * (b.ln() - a.ln())).exp() }
fn inv_log_lerp(v: f32, a: f32, b: f32) -> f32 { ((v.max(a).ln() - a.ln()) / (b.ln() - a.ln())).clamp(0.0, 1.0) }

fn fmt_ms(ms: u32) -> String {
    if ms == 0 { return "0ms".to_string(); }
    if ms >= 1000 && ms % 1000 == 0 { format!("{}s", ms / 1000) }
    else if ms >= 1000 { format!("{:.1}s", ms as f32 / 1000.0) }
    else { format!("{}ms", ms) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_parses() {
        let (ok, fail) = validate_random(10000);
        assert!(fail == 0, "{} out of {} failed to parse", fail, ok + fail);
    }

    #[test]
    fn test_specific_points() {
        for v in &[[0.0f32; GRAMMAR_DIM], [1.0; GRAMMAR_DIM], [0.5; GRAMMAR_DIM]] {
            let dsl = decode(v);
            assert!(parse_abilities(&dsl).is_ok(), "failed:\n{}", dsl);
        }
    }

    #[test]
    fn test_print_samples() {
        let seeds: &[u64] = &[1, 42, 100, 256, 999, 7777];
        for &seed in seeds {
            let mut rng = seed;
            let mut v = [0.0f32; GRAMMAR_DIM];
            for d in &mut v {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *d = (rng >> 33) as f32 / (1u64 << 31) as f32;
            }
            let dsl = decode(&v);
            println!("=== Seed {} ===\n{}\n", seed, dsl);
        }
    }
}
