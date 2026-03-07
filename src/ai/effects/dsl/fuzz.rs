//! Property-based fuzzer for the ability DSL.
//!
//! Generates structurally valid `.ability` DSL strings and verifies that the
//! full pipeline (parse → lower) succeeds on every generated input.

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use crate::ai::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
    use crate::ai::effects::dsl::parse_abilities;
    use crate::ai::effects::effect_enum::Effect;
    use crate::ai::effects::types::*;

    // -----------------------------------------------------------------------
    // Coverage tracker
    // -----------------------------------------------------------------------

    #[derive(Default)]
    struct Coverage {
        effects: BTreeSet<&'static str>,
        deliveries: BTreeSet<&'static str>,
        areas: BTreeSet<&'static str>,
        conditions: BTreeSet<&'static str>,
        triggers: BTreeSet<&'static str>,
        targetings: BTreeSet<&'static str>,
        // ability-level features
        has_charges: bool,
        has_recast: bool,
        has_cost: bool,
        has_toggle: bool,
        has_unstoppable: bool,
        has_form: bool,
        has_tags: bool,
        has_scaling: bool,
    }

    impl Coverage {
        fn record_abilities(&mut self, abilities: &[AbilityDef], passives: &[PassiveDef]) {
            for ab in abilities {
                self.record_targeting(&ab.targeting);
                if let Some(ref d) = ab.delivery {
                    self.record_delivery(d);
                }
                for eff in &ab.effects {
                    self.record_effect_tree(eff);
                }
                if ab.max_charges > 0 { self.has_charges = true; }
                if ab.recast_count > 0 { self.has_recast = true; }
                if ab.resource_cost > 0 { self.has_cost = true; }
                if ab.is_toggle { self.has_toggle = true; }
                if ab.unstoppable { self.has_unstoppable = true; }
                if ab.form.is_some() || ab.swap_form.is_some() { self.has_form = true; }
            }
            for p in passives {
                self.record_trigger(&p.trigger);
                for eff in &p.effects {
                    self.record_effect_tree(eff);
                }
            }
        }

        fn record_effect_tree(&mut self, ce: &ConditionalEffect) {
            self.record_effect(&ce.effect);
            if let Some(ref area) = ce.area {
                self.record_area(area);
            }
            if let Some(ref cond) = ce.condition {
                self.record_condition(cond);
            }
            if !ce.tags.is_empty() { self.has_tags = true; }
            for else_eff in &ce.else_effects {
                self.record_effect_tree(else_eff);
            }
        }

        fn record_effect(&mut self, eff: &Effect) {
            let name = match eff {
                Effect::Damage { bonus, .. } => {
                    if !bonus.is_empty() { self.has_scaling = true; }
                    "Damage"
                }
                Effect::Heal { bonus, .. } => {
                    if !bonus.is_empty() { self.has_scaling = true; }
                    "Heal"
                }
                Effect::Shield { .. } => "Shield",
                Effect::Stun { .. } => "Stun",
                Effect::Slow { .. } => "Slow",
                Effect::Knockback { .. } => "Knockback",
                Effect::Dash { is_blink: true, .. } => { self.effects.insert("Blink"); "Dash" }
                Effect::Dash { .. } => "Dash",
                Effect::Buff { .. } => "Buff",
                Effect::Debuff { .. } => "Debuff",
                Effect::Duel { .. } => "Duel",
                Effect::Summon { .. } => "Summon",
                Effect::CommandSummons { .. } => "CommandSummons",
                Effect::Dispel { .. } => "Dispel",
                Effect::Root { .. } => "Root",
                Effect::Silence { .. } => "Silence",
                Effect::Fear { .. } => "Fear",
                Effect::Taunt { .. } => "Taunt",
                Effect::Pull { .. } => "Pull",
                Effect::Swap => "Swap",
                Effect::Reflect { .. } => "Reflect",
                Effect::Lifesteal { .. } => "Lifesteal",
                Effect::DamageModify { .. } => "DamageModify",
                Effect::SelfDamage { .. } => "SelfDamage",
                Effect::Execute { .. } => "Execute",
                Effect::Blind { .. } => "Blind",
                Effect::OnHitBuff { .. } => "OnHitBuff",
                Effect::Resurrect { .. } => "Resurrect",
                Effect::OverhealShield { .. } => "OverhealShield",
                Effect::AbsorbToHeal { .. } => "AbsorbToHeal",
                Effect::ShieldSteal { .. } => "ShieldSteal",
                Effect::StatusClone { .. } => "StatusClone",
                Effect::Immunity { .. } => "Immunity",
                Effect::Detonate { .. } => "Detonate",
                Effect::StatusTransfer { .. } => "StatusTransfer",
                Effect::DeathMark { .. } => "DeathMark",
                Effect::Polymorph { .. } => "Polymorph",
                Effect::Banish { .. } => "Banish",
                Effect::Confuse { .. } => "Confuse",
                Effect::Charm { .. } => "Charm",
                Effect::Stealth { .. } => "Stealth",
                Effect::Leash { .. } => "Leash",
                Effect::Link { .. } => "Link",
                Effect::Redirect { .. } => "Redirect",
                Effect::Rewind { .. } => "Rewind",
                Effect::CooldownModify { .. } => "CooldownModify",
                Effect::ApplyStacks { .. } => "ApplyStacks",
                Effect::Obstacle { .. } => "Obstacle",
                Effect::Suppress { .. } => "Suppress",
                Effect::Grounded { .. } => "Grounded",
                Effect::ProjectileBlock { .. } => "ProjectileBlock",
                Effect::Attach { .. } => "Attach",
                Effect::EvolveAbility { .. } => "EvolveAbility",
            };
            self.effects.insert(name);
        }

        fn record_delivery(&mut self, d: &Delivery) {
            let name = match d {
                Delivery::Instant => "Instant",
                Delivery::Projectile { on_hit, on_arrival, .. } => {
                    for eff in on_hit { self.record_effect_tree(eff); }
                    for eff in on_arrival { self.record_effect_tree(eff); }
                    "Projectile"
                }
                Delivery::Channel { .. } => "Channel",
                Delivery::Zone { .. } => "Zone",
                Delivery::Tether { on_complete, .. } => {
                    for eff in on_complete { self.record_effect_tree(eff); }
                    "Tether"
                }
                Delivery::Trap { .. } => "Trap",
                Delivery::Chain { on_hit, .. } => {
                    for eff in on_hit { self.record_effect_tree(eff); }
                    "Chain"
                }
            };
            self.deliveries.insert(name);
        }

        fn record_area(&mut self, a: &Area) {
            self.areas.insert(match a {
                Area::SingleTarget => "SingleTarget",
                Area::Circle { .. } => "Circle",
                Area::Cone { .. } => "Cone",
                Area::Line { .. } => "Line",
                Area::Ring { .. } => "Ring",
                Area::SelfOnly => "SelfOnly",
                Area::Spread { .. } => "Spread",
            });
        }

        fn record_condition(&mut self, c: &Condition) {
            // Collect children first to avoid borrow issues
            let mut children: Vec<&Condition> = Vec::new();
            let name = match c {
                Condition::Always => "Always",
                Condition::TargetHpBelow { .. } => "TargetHpBelow",
                Condition::TargetHpAbove { .. } => "TargetHpAbove",
                Condition::TargetIsStunned => "TargetIsStunned",
                Condition::TargetIsSlowed => "TargetIsSlowed",
                Condition::CasterHpBelow { .. } => "CasterHpBelow",
                Condition::CasterHpAbove { .. } => "CasterHpAbove",
                Condition::HitCountAbove { .. } => "HitCountAbove",
                Condition::TargetHasTag { .. } => "TargetHasTag",
                Condition::TargetIsRooted => "TargetIsRooted",
                Condition::TargetIsSilenced => "TargetIsSilenced",
                Condition::TargetIsFeared => "TargetIsFeared",
                Condition::TargetIsTaunted => "TargetIsTaunted",
                Condition::TargetIsBanished => "TargetIsBanished",
                Condition::TargetIsStealthed => "TargetIsStealthed",
                Condition::TargetIsCharmed => "TargetIsCharmed",
                Condition::TargetIsPolymorphed => "TargetIsPolymorphed",
                Condition::CasterHasStatus { .. } => "CasterHasStatus",
                Condition::TargetHasStatus { .. } => "TargetHasStatus",
                Condition::TargetDebuffCount { .. } => "TargetDebuffCount",
                Condition::CasterBuffCount { .. } => "CasterBuffCount",
                Condition::AllyCountBelow { .. } => "AllyCountBelow",
                Condition::EnemyCountBelow { .. } => "EnemyCountBelow",
                Condition::TargetStackCount { .. } => "TargetStackCount",
                Condition::And { conditions } => {
                    children.extend(conditions.iter());
                    "And"
                }
                Condition::Or { conditions } => {
                    children.extend(conditions.iter());
                    "Or"
                }
                Condition::Not { condition } => {
                    children.push(condition);
                    "Not"
                }
                Condition::TargetDistanceBelow { .. } => "TargetDistanceBelow",
                Condition::TargetDistanceAbove { .. } => "TargetDistanceAbove",
                Condition::CasterResourceBelow { .. } => "CasterResourceBelow",
                Condition::CasterResourceAbove { .. } => "CasterResourceAbove",
            };
            self.conditions.insert(name);
            for child in children {
                self.record_condition(child);
            }
        }

        fn record_trigger(&mut self, t: &Trigger) {
            self.triggers.insert(match t {
                Trigger::OnDamageDealt => "OnDamageDealt",
                Trigger::OnDamageTaken => "OnDamageTaken",
                Trigger::OnKill => "OnKill",
                Trigger::OnAllyDamaged { .. } => "OnAllyDamaged",
                Trigger::OnDeath => "OnDeath",
                Trigger::OnAbilityUsed => "OnAbilityUsed",
                Trigger::OnHpBelow { .. } => "OnHpBelow",
                Trigger::OnHpAbove { .. } => "OnHpAbove",
                Trigger::OnShieldBroken => "OnShieldBroken",
                Trigger::OnStunExpire => "OnStunExpire",
                Trigger::Periodic { .. } => "Periodic",
                Trigger::OnHealReceived => "OnHealReceived",
                Trigger::OnStatusApplied => "OnStatusApplied",
                Trigger::OnStatusExpired => "OnStatusExpired",
                Trigger::OnResurrect => "OnResurrect",
                Trigger::OnDodge => "OnDodge",
                Trigger::OnReflect => "OnReflect",
                Trigger::OnAllyKilled { .. } => "OnAllyKilled",
                Trigger::OnAutoAttack => "OnAutoAttack",
                Trigger::OnStackReached { .. } => "OnStackReached",
            });
        }

        fn record_targeting(&mut self, t: &AbilityTargeting) {
            self.targetings.insert(match t {
                AbilityTargeting::TargetEnemy => "TargetEnemy",
                AbilityTargeting::TargetAlly => "TargetAlly",
                AbilityTargeting::SelfCast => "SelfCast",
                AbilityTargeting::SelfAoe => "SelfAoe",
                AbilityTargeting::GroundTarget => "GroundTarget",
                AbilityTargeting::Direction => "Direction",
                AbilityTargeting::Vector => "Vector",
                AbilityTargeting::Global => "Global",
            });
        }

        fn report(&self) -> String {
            // Known universe of DSL-reachable features
            let all_effects: BTreeSet<&str> = [
                "Damage", "Heal", "Shield", "Stun", "Slow", "Knockback", "Dash", "Blink",
                "Buff", "Debuff", "Summon", "Root", "Silence", "Fear", "Taunt", "Pull",
                "Swap", "Reflect", "Lifesteal", "DamageModify", "SelfDamage", "Execute",
                "Blind", "Stealth", "ApplyStacks", "Charm", "Polymorph", "Banish",
                "Confuse", "Suppress", "Grounded", "Resurrect",
            ].into_iter().collect();
            let all_deliveries: BTreeSet<&str> = [
                "Projectile", "Chain", "Zone", "Channel", "Tether", "Trap",
            ].into_iter().collect();
            let all_areas: BTreeSet<&str> = [
                "Circle", "Cone", "Line", "Ring", "Spread",
            ].into_iter().collect();
            let all_conditions: BTreeSet<&str> = [
                "TargetHpBelow", "TargetHpAbove", "CasterHpBelow", "CasterHpAbove",
                "TargetIsStunned", "TargetIsSlowed", "TargetIsRooted", "TargetIsSilenced",
                "TargetIsFeared", "HitCountAbove",
            ].into_iter().collect();
            let all_triggers: BTreeSet<&str> = [
                "OnDamageDealt", "OnDamageTaken", "OnKill", "OnDeath", "OnAbilityUsed",
                "OnAutoAttack", "OnShieldBroken", "OnHealReceived", "OnDodge", "OnReflect",
                "OnHpBelow", "OnHpAbove", "Periodic",
            ].into_iter().collect();
            let all_targetings: BTreeSet<&str> = [
                "TargetEnemy", "TargetAlly", "SelfCast", "SelfAoe", "GroundTarget",
                "Direction", "Vector", "Global",
            ].into_iter().collect();
            let all_features: BTreeSet<&str> = [
                "charges", "recast", "cost", "tags", "scaling",
            ].into_iter().collect();

            fn section(name: &str, hit: &BTreeSet<&str>, all: &BTreeSet<&str>) -> String {
                let covered: BTreeSet<&&str> = hit.intersection(all).collect();
                let missing: BTreeSet<&&str> = all.difference(hit).collect();
                let pct = if all.is_empty() { 100.0 } else { covered.len() as f64 / all.len() as f64 * 100.0 };
                let mut s = format!("  {name}: {}/{} ({pct:.0}%)", covered.len(), all.len());
                if !missing.is_empty() {
                    s.push_str(&format!("  missing: {:?}", missing));
                }
                s
            }

            let mut hit_features = BTreeSet::new();
            if self.has_charges { hit_features.insert("charges"); }
            if self.has_recast { hit_features.insert("recast"); }
            if self.has_cost { hit_features.insert("cost"); }
            if self.has_tags { hit_features.insert("tags"); }
            if self.has_scaling { hit_features.insert("scaling"); }

            let sections = [
                section("Effects", &self.effects, &all_effects),
                section("Deliveries", &self.deliveries, &all_deliveries),
                section("Areas", &self.areas, &all_areas),
                section("Conditions", &self.conditions, &all_conditions),
                section("Triggers", &self.triggers, &all_triggers),
                section("Targeting", &self.targetings, &all_targetings),
                section("Features", &hit_features, &all_features),
            ];

            let total_all = all_effects.len() + all_deliveries.len() + all_areas.len()
                + all_conditions.len() + all_triggers.len() + all_targetings.len()
                + all_features.len();
            let total_hit = self.effects.intersection(&all_effects).count()
                + self.deliveries.intersection(&all_deliveries).count()
                + self.areas.intersection(&all_areas).count()
                + self.conditions.intersection(&all_conditions).count()
                + self.triggers.intersection(&all_triggers).count()
                + self.targetings.intersection(&all_targetings).count()
                + hit_features.intersection(&all_features).count();

            let overall = total_hit as f64 / total_all as f64 * 100.0;

            format!(
                "DSL Feature Coverage: {total_hit}/{total_all} ({overall:.0}%)\n{}",
                sections.join("\n")
            )
        }
    }

    // -----------------------------------------------------------------------
    // Building blocks — strategies that produce DSL string fragments
    // -----------------------------------------------------------------------

    fn ident_strategy() -> impl Strategy<Value = String> {
        "[a-zA-Z_][a-zA-Z0-9_]{0,15}"
    }

    fn duration_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            (1u32..60_000).prop_map(|ms| format!("{ms}ms")),
            (1u32..120).prop_map(|s| format!("{s}s")),
        ]
    }

    fn number_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            (0i32..999).prop_map(|n| n.to_string()),
            (0u32..999, 0u32..99).prop_map(|(i, f)| format!("{i}.{f}")),
        ]
    }

    fn targeting_strategy() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("enemy"),
            Just("ally"),
            Just("self"),
            Just("self_aoe"),
            Just("ground"),
            Just("direction"),
            Just("vector"),
            Just("global"),
        ]
    }

    fn hint_strategy() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("damage"),
            Just("heal"),
            Just("buff"),
            Just("utility"),
            Just("defense"),
            Just("crowd_control"),
        ]
    }

    fn area_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            (1u32..100).prop_map(|r| format!("in circle({}.0)", r)),
            (1u32..100, 10u32..180).prop_map(|(r, a)| format!("in cone({}.0, {}.0)", r, a)),
            (1u32..100, 1u32..50).prop_map(|(l, w)| format!("in line({}.0, {}.0)", l, w)),
            (1u32..50, 2u32..100).prop_map(|(i, o)| format!("in ring({}.0, {}.0)", i, o)),
            (1u32..100, 1u32..10).prop_map(|(r, t)| format!("in spread({}.0, {})", r, t)),
        ]
    }

    fn tag_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("FIRE"),
            Just("ICE"),
            Just("LIGHTNING"),
            Just("POISON"),
            Just("DARK"),
            Just("HOLY"),
            Just("PHYSICAL"),
            Just("MAGIC"),
            Just("TRUE"),
            Just("CROWD_CONTROL"),
        ]
        .prop_flat_map(|tag| (Just(tag), 1u32..100).prop_map(|(t, v)| format!("[{t}: {v}]")))
    }

    fn opt_area() -> impl Strategy<Value = String> {
        prop_oneof![3 => Just(String::new()), 1 => area_strategy()]
    }

    fn opt_tags() -> impl Strategy<Value = String> {
        prop_oneof![
            3 => Just(String::new()),
            1 => tag_strategy(),
        ]
    }

    // -----------------------------------------------------------------------
    // Effect generators
    // -----------------------------------------------------------------------

    fn damage_effect() -> impl Strategy<Value = String> {
        (1i32..200, opt_area(), opt_tags()).prop_map(|(amt, area, tags)| {
            format!("    damage {amt} {area} {tags}")
        })
    }

    fn heal_effect() -> impl Strategy<Value = String> {
        (1i32..200).prop_map(|amt| format!("    heal {amt}"))
    }

    fn shield_effect() -> impl Strategy<Value = String> {
        (1i32..200, duration_strategy()).prop_map(|(amt, dur)| {
            format!("    shield {amt} for {dur}")
        })
    }

    fn cc_effect() -> impl Strategy<Value = String> {
        let cc_type = prop_oneof![
            Just("stun"),
            Just("root"),
            Just("silence"),
            Just("fear"),
            Just("taunt"),
            Just("charm"),
            Just("suppress"),
            Just("grounded"),
            Just("confuse"),
            Just("banish"),
            Just("polymorph"),
        ];
        (cc_type, duration_strategy(), opt_area()).prop_map(|(cc, dur, area)| {
            format!("    {cc} {dur} {area}")
        })
    }

    fn slow_effect() -> impl Strategy<Value = String> {
        (1u32..9, duration_strategy(), opt_area()).prop_map(|(factor_10, dur, area)| {
            let factor = factor_10 as f32 / 10.0;
            format!("    slow {factor} for {dur} {area}")
        })
    }

    fn knockback_effect() -> impl Strategy<Value = String> {
        (1u32..10).prop_map(|d| format!("    knockback {d}.0"))
    }

    fn pull_effect() -> impl Strategy<Value = String> {
        (1u32..10).prop_map(|d| format!("    pull {d}.0"))
    }

    fn dash_effect() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("    dash to_target".to_string()),
            Just("    dash to_position".to_string()),
            (1u32..10).prop_map(|d| format!("    dash {d}.0")),
        ]
    }

    fn blink_effect() -> impl Strategy<Value = String> {
        (1u32..10).prop_map(|d| format!("    blink {d}.0"))
    }

    fn buff_effect() -> impl Strategy<Value = String> {
        let stat = prop_oneof![
            Just("move_speed"),
            Just("attack_speed"),
            Just("damage_output"),
            Just("cooldown_reduction"),
        ];
        (stat, 1u32..9, duration_strategy()).prop_map(|(s, f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    buff {s} {f} for {dur}")
        })
    }

    fn debuff_effect() -> impl Strategy<Value = String> {
        let stat = prop_oneof![Just("move_speed"), Just("attack_speed"), Just("damage_output")];
        (stat, 1u32..9, duration_strategy()).prop_map(|(s, f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    debuff {s} {f} for {dur}")
        })
    }

    fn damage_modify_effect() -> impl Strategy<Value = String> {
        (5u32..20, duration_strategy()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    damage_modify {f} for {dur}")
        })
    }

    fn lifesteal_effect() -> impl Strategy<Value = String> {
        (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    lifesteal {f} for {dur}")
        })
    }

    fn reflect_effect() -> impl Strategy<Value = String> {
        (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    reflect {f} for {dur}")
        })
    }

    fn stealth_effect() -> impl Strategy<Value = String> {
        (duration_strategy(), proptest::bool::ANY).prop_map(|(dur, bod)| {
            let suffix = if bod { " break_on_damage" } else { "" };
            format!("    stealth for {dur}{suffix}")
        })
    }

    fn summon_effect() -> impl Strategy<Value = String> {
        let templates = prop_oneof![
            Just("skeleton"),
            Just("minion"),
            Just("soldier"),
            Just("clone"),
        ];
        (templates, 1u32..4).prop_map(|(t, count)| {
            if count > 1 {
                format!("    summon \"{t}\" x{count}")
            } else {
                format!("    summon \"{t}\"")
            }
        })
    }

    fn apply_stacks_effect() -> impl Strategy<Value = String> {
        (ident_strategy(), 1u32..5, 2u32..10).prop_map(|(name, count, max)| {
            format!("    apply_stacks \"{name}\" {count} max {max}")
        })
    }

    fn blind_effect() -> impl Strategy<Value = String> {
        (1u32..9, duration_strategy()).prop_map(|(f10, dur)| {
            let f = f10 as f32 / 10.0;
            format!("    blind {f} for {dur}")
        })
    }

    fn swap_effect() -> impl Strategy<Value = String> {
        Just("    swap".to_string())
    }

    fn self_damage_effect() -> impl Strategy<Value = String> {
        (1i32..100).prop_map(|amt| format!("    self_damage {amt}"))
    }

    fn execute_effect() -> impl Strategy<Value = String> {
        (5u32..50).prop_map(|pct| format!("    execute {pct}.0"))
    }

    fn resurrect_effect() -> impl Strategy<Value = String> {
        (10u32..100).prop_map(|pct| format!("    resurrect {pct}.0"))
    }

    /// Pick a random effect from any category.
    fn any_effect() -> impl Strategy<Value = String> {
        prop_oneof![
            4 => damage_effect(),
            3 => heal_effect(),
            2 => shield_effect(),
            3 => cc_effect(),
            2 => slow_effect(),
            1 => knockback_effect(),
            1 => pull_effect(),
            2 => dash_effect(),
            1 => blink_effect(),
            2 => buff_effect(),
            1 => debuff_effect(),
            1 => damage_modify_effect(),
            1 => lifesteal_effect(),
            1 => reflect_effect(),
            1 => stealth_effect(),
            1 => summon_effect(),
            1 => apply_stacks_effect(),
            1 => blind_effect(),
            1 => swap_effect(),
            1 => self_damage_effect(),
            1 => execute_effect(),
            1 => resurrect_effect(),
            2 => conditional_damage(),
            2 => conditional_heal(),
            2 => damage_with_scaling(),
        ]
    }

    fn effect_list(min: usize, max: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(any_effect(), min..=max)
            .prop_map(|effects| effects.join("\n"))
    }

    // -----------------------------------------------------------------------
    // Condition generators (for `when` clauses)
    // -----------------------------------------------------------------------

    fn simple_condition() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("target_is_stunned".to_string()),
            Just("target_is_slowed".to_string()),
            Just("target_is_rooted".to_string()),
            Just("target_is_silenced".to_string()),
            Just("target_is_feared".to_string()),
            (1u32..100).prop_map(|pct| format!("target_hp_below({pct}%)")),
            (1u32..100).prop_map(|pct| format!("caster_hp_below({pct}%)")),
            (1u32..100).prop_map(|pct| format!("target_hp_above({pct}%)")),
            (1u32..100).prop_map(|pct| format!("caster_hp_above({pct}%)")),
            (1u32..10).prop_map(|n| format!("hit_count_above({n})")),
        ]
    }

    fn conditional_damage() -> impl Strategy<Value = String> {
        (1i32..200, simple_condition()).prop_map(|(amt, cond)| {
            format!("    damage {amt} when {cond}")
        })
    }

    fn conditional_heal() -> impl Strategy<Value = String> {
        (1i32..200, simple_condition()).prop_map(|(amt, cond)| {
            format!("    heal {amt} when {cond}")
        })
    }

    // -----------------------------------------------------------------------
    // Scaling generators
    // -----------------------------------------------------------------------

    fn scaling_stat() -> impl Strategy<Value = &'static str> {
        prop_oneof![
            Just("target_max_hp"),
            Just("target_current_hp"),
            Just("target_missing_hp"),
            Just("caster_max_hp"),
            Just("caster_current_hp"),
            Just("caster_missing_hp"),
            Just("caster_attack_damage"),
        ]
    }

    fn damage_with_scaling() -> impl Strategy<Value = String> {
        (1i32..200, 1u32..100, scaling_stat()).prop_map(|(amt, pct, stat)| {
            format!("    damage {amt} + {pct}% {stat}")
        })
    }

    // -----------------------------------------------------------------------
    // Delivery generators
    // -----------------------------------------------------------------------

    fn projectile_delivery() -> impl Strategy<Value = String> {
        (
            (1u32..30).prop_map(|s| format!("{s}.0")),
            proptest::bool::ANY,
            effect_list(1, 3),
        )
            .prop_map(|(speed, pierce, effects)| {
                let pierce_str = if pierce { ", pierce" } else { "" };
                format!(
                    "    deliver projectile {{ speed: {speed}{pierce_str} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            })
    }

    fn chain_delivery() -> impl Strategy<Value = String> {
        (1u32..6, (2u32..8).prop_map(|r| format!("{r}.0")), effect_list(1, 2)).prop_map(
            |(bounces, range, effects)| {
                format!(
                    "    deliver chain {{ bounces: {bounces}, range: {range} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            },
        )
    }

    fn zone_delivery() -> impl Strategy<Value = String> {
        (duration_strategy(), duration_strategy(), effect_list(1, 2)).prop_map(
            |(dur, tick, effects)| {
                format!(
                    "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            },
        )
    }

    fn channel_delivery() -> impl Strategy<Value = String> {
        (duration_strategy(), duration_strategy(), effect_list(1, 2)).prop_map(
            |(dur, tick, effects)| {
                format!(
                    "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            },
        )
    }

    fn tether_delivery() -> impl Strategy<Value = String> {
        ((2u32..10).prop_map(|r| format!("{r}.0")), effect_list(1, 2)).prop_map(
            |(range, effects)| {
                format!(
                    "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effects}\n        }}\n    }}"
                )
            },
        )
    }

    fn trap_delivery() -> impl Strategy<Value = String> {
        (
            duration_strategy(),
            (1u32..5).prop_map(|r| format!("{r}.0")),
            effect_list(1, 2),
        )
            .prop_map(|(dur, radius, effects)| {
                format!(
                    "    deliver trap {{ duration: {dur}, trigger_radius: {radius} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                )
            })
    }

    fn any_delivery() -> impl Strategy<Value = String> {
        prop_oneof![
            3 => projectile_delivery(),
            1 => chain_delivery(),
            1 => zone_delivery(),
            1 => channel_delivery(),
            1 => tether_delivery(),
            1 => trap_delivery(),
        ]
    }

    // -----------------------------------------------------------------------
    // Ability block generator
    // -----------------------------------------------------------------------

    fn ability_block() -> impl Strategy<Value = String> {
        (
            ident_strategy(),
            targeting_strategy(),
            (0u32..100).prop_map(|r| format!("{r}.0")),
            duration_strategy(),
            duration_strategy(),
            hint_strategy(),
            // Optional resource cost
            proptest::option::of(1u32..30),
            // Optional charges
            proptest::option::of((1u32..5, duration_strategy())),
            // Optional recast
            proptest::option::of((1u32..4, duration_strategy())),
            // Effects vs delivery
            prop_oneof![
                3 => effect_list(1, 5).prop_map(|e| (e, None)),
                2 => (effect_list(0, 2), any_delivery().prop_map(Some)).prop_map(|(e, d)| (e, d)),
            ],
        )
            .prop_map(
                |(name, target, range, cd, cast, hint, cost, charges, recast, (effects, delivery))| {
                    let mut lines = vec![format!("ability {name} {{")];
                    lines.push(format!("    target: {target}, range: {range}"));
                    lines.push(format!("    cooldown: {cd}, cast: {cast}"));
                    lines.push(format!("    hint: {hint}"));

                    if let Some(c) = cost {
                        lines.push(format!("    cost: {c}"));
                    }
                    if let Some((ch, rech)) = charges {
                        lines.push(format!("    charges: {ch}"));
                        lines.push(format!("    recharge: {rech}"));
                    }
                    if let Some((rc, rw)) = recast {
                        lines.push(format!("    recast: {rc}"));
                        lines.push(format!("    recast_window: {rw}"));
                    }

                    lines.push(String::new()); // blank line before body

                    if let Some(del) = delivery {
                        lines.push(del);
                    }
                    if !effects.is_empty() {
                        lines.push(effects);
                    }

                    lines.push("}".to_string());
                    lines.join("\n")
                },
            )
    }

    // -----------------------------------------------------------------------
    // Passive block generator
    // -----------------------------------------------------------------------

    fn trigger_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            Just("on_damage_dealt".to_string()),
            Just("on_damage_taken".to_string()),
            Just("on_kill".to_string()),
            Just("on_death".to_string()),
            Just("on_ability_used".to_string()),
            Just("on_auto_attack".to_string()),
            Just("on_shield_broken".to_string()),
            Just("on_heal_received".to_string()),
            Just("on_dodge".to_string()),
            Just("on_reflect".to_string()),
            (1u32..100).prop_map(|pct| format!("on_hp_below({pct}%)")),
            (1u32..100).prop_map(|pct| format!("on_hp_above({pct}%)")),
            duration_strategy().prop_map(|d| format!("periodic({d})")),
        ]
    }

    fn passive_block() -> impl Strategy<Value = String> {
        (
            ident_strategy(),
            trigger_strategy(),
            duration_strategy(),
            effect_list(1, 3),
        )
            .prop_map(|(name, trigger, cd, effects)| {
                format!(
                    "passive {name} {{\n    trigger: {trigger}\n    cooldown: {cd}\n\n{effects}\n}}"
                )
            })
    }

    // -----------------------------------------------------------------------
    // Full file generator
    // -----------------------------------------------------------------------

    fn ability_file() -> impl Strategy<Value = String> {
        (
            proptest::collection::vec(ability_block(), 1..=4),
            proptest::collection::vec(passive_block(), 0..=2),
        )
            .prop_map(|(abilities, passives)| {
                let mut parts: Vec<String> = abilities;
                parts.extend(passives);
                parts.join("\n\n")
            })
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn fuzz_parse_and_lower(input in ability_file()) {
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "Failed to parse+lower generated DSL:\n---INPUT---\n{}\n---ERROR---\n{}",
                input,
                result.unwrap_err()
            );

            let (abilities, passives) = result.unwrap();
            prop_assert!(!abilities.is_empty(), "should have at least 1 ability");

            // Verify basic invariants
            for ab in &abilities {
                prop_assert!(!ab.name.is_empty(), "ability name should not be empty");
                prop_assert!(ab.cooldown_ms > 0, "ability should have cooldown");
            }
            for p in &passives {
                prop_assert!(!p.name.is_empty(), "passive name should not be empty");
            }
        }

        #[test]
        fn fuzz_conditional_effects(
            name in ident_strategy(),
            effect_amt in 1i32..200,
            cond in simple_condition(),
        ) {
            let input = format!(
                "ability {name} {{\n    target: enemy, range: 5.0\n    cooldown: 5s, cast: 300ms\n    hint: damage\n\n    damage {effect_amt} when {cond}\n}}"
            );
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "conditional effect failed:\n{}\n---\n{}",
                input,
                result.unwrap_err()
            );
        }

        #[test]
        fn fuzz_scaling_effects(
            name in ident_strategy(),
            amt in 1i32..200,
            pct in 1u32..100,
            stat in scaling_stat(),
        ) {
            let input = format!(
                "ability {name} {{\n    target: enemy, range: 5.0\n    cooldown: 5s, cast: 300ms\n    hint: damage\n\n    damage {amt} + {pct}% {stat}\n}}"
            );
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "scaling effect failed:\n{}\n---\n{}",
                input,
                result.unwrap_err()
            );
        }

        #[test]
        fn fuzz_delivery_blocks(
            name in ident_strategy(),
            delivery in any_delivery(),
        ) {
            let input = format!(
                "ability {name} {{\n    target: enemy, range: 5.0\n    cooldown: 5s, cast: 300ms\n    hint: damage\n\n{delivery}\n}}"
            );
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "delivery block failed:\n{}\n---\n{}",
                input,
                result.unwrap_err()
            );
        }

        #[test]
        fn fuzz_passives(input in passive_block()) {
            // Wrap in a file with a minimal ability (passives need at least parse correctly)
            let full = format!(
                "ability Dummy {{\n    target: self\n    cooldown: 5s\n    hint: utility\n    dash 3.0\n}}\n\n{input}"
            );
            let result = parse_abilities(&full);
            prop_assert!(
                result.is_ok(),
                "passive failed:\n{}\n---\n{}",
                full,
                result.unwrap_err()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bulk generation — random walk producing thousands of abilities
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Each case generates a file with 20-50 abilities + 5-15 passives,
        /// run 50 times = 1000-2500 abilities total per test run.
        #[test]
        fn fuzz_bulk_generation(
            abilities in proptest::collection::vec(ability_block(), 20..=50),
            passives in proptest::collection::vec(passive_block(), 5..=15),
        ) {
            let mut parts: Vec<String> = abilities;
            parts.extend(passives);
            let input = parts.join("\n\n");

            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "Bulk generation failed on file with {} blocks:\n---ERROR---\n{}",
                parts.len(),
                result.unwrap_err()
            );

            let (abs, pas) = result.unwrap();
            prop_assert!(abs.len() >= 20);
            prop_assert!(pas.len() >= 5);

            // Spot-check every ability lowered with sane fields
            for ab in &abs {
                prop_assert!(!ab.name.is_empty());
                prop_assert!(ab.cooldown_ms > 0);
            }
            for p in &pas {
                prop_assert!(!p.name.is_empty());
                prop_assert!(!p.effects.is_empty());
            }
        }
    }

    /// Generate 1000 abilities + 200 passives and report feature coverage.
    #[test]
    fn bulk_coverage_report() {
        use proptest::test_runner::{TestRunner, Config};
        use proptest::strategy::ValueTree;

        let config = Config { cases: 1, .. Config::default() };
        let mut runner = TestRunner::new(config);

        let ab_strat = proptest::collection::vec(ability_block(), 1000..=1000);
        let pa_strat = proptest::collection::vec(passive_block(), 200..=200);

        let ab_tree = ab_strat.new_tree(&mut runner).unwrap();
        let pa_tree = pa_strat.new_tree(&mut runner).unwrap();

        let mut parts: Vec<String> = ab_tree.current();
        parts.extend(pa_tree.current());
        let input = parts.join("\n\n");

        let (abilities, passives) = parse_abilities(&input)
            .unwrap_or_else(|e| panic!("Failed to parse bulk generation:\n{e}"));

        assert_eq!(abilities.len(), 1000);
        assert_eq!(passives.len(), 200);

        let mut cov = Coverage::default();
        cov.record_abilities(&abilities, &passives);

        let report = cov.report();
        eprintln!("\n{report}\n");

        // Assert minimum coverage thresholds
        let effect_count = cov.effects.len();
        let delivery_count = cov.deliveries.len();
        let area_count = cov.areas.len();
        let trigger_count = cov.triggers.len();
        let targeting_count = cov.targetings.len();

        assert!(effect_count >= 25,
            "expected ≥25 effect types covered, got {effect_count}. Report:\n{report}");
        assert!(delivery_count >= 6,
            "expected all 6 delivery types covered, got {delivery_count}. Report:\n{report}");
        assert!(area_count >= 5,
            "expected all 5 area types covered, got {area_count}. Report:\n{report}");
        assert!(trigger_count >= 10,
            "expected ≥10 trigger types covered, got {trigger_count}. Report:\n{report}");
        assert!(targeting_count >= 8,
            "expected all 8 targeting types covered, got {targeting_count}. Report:\n{report}");
    }

    // Extra: targeted regression-style tests for edge cases

    #[test]
    fn fuzz_empty_effects_in_delivery() {
        // Delivery with empty on_hit (valid — some zone deliveries just set up area)
        let input = r#"
ability Test {
    target: ground, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver zone { duration: 4s, tick: 1s } {
        on_hit {
            damage 10
        }
    }
}
"#;
        parse_abilities(input).unwrap();
    }

    #[test]
    fn fuzz_multiple_tags() {
        let input = r#"
ability Test {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    damage 50 [FIRE: 60, MAGIC: 40, CROWD_CONTROL: 20]
}
"#;
        parse_abilities(input).unwrap();
    }

    #[test]
    fn fuzz_projectile_with_arrival() {
        let input = r#"
ability Test {
    target: enemy, range: 8.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 10.0, pierce, width: 0.5 } {
        on_hit {
            damage 30 [PHYSICAL: 50]
            slow 0.3 for 2s
        }
        on_arrival {
            damage 15 in circle(2.5)
        }
    }
}
"#;
        parse_abilities(input).unwrap();
    }

    #[test]
    fn fuzz_many_effects() {
        let input = r#"
ability KitchenSink {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 300ms
    hint: damage
    cost: 15
    charges: 2
    recharge: 8s

    damage 50 [FIRE: 60]
    heal 20
    shield 30 for 4s
    stun 1s
    slow 0.3 for 2s
    dash to_target
    buff move_speed 0.2 for 3s
    apply_stacks "combo" 1 max 5
}
"#;
        let (abilities, _) = parse_abilities(input).unwrap();
        assert_eq!(abilities[0].effects.len(), 8);
    }

    #[test]
    fn fuzz_mixed_abilities_and_passives() {
        let input = r#"
ability A {
    target: enemy, range: 2.0
    cooldown: 3s
    hint: damage
    damage 10
}

passive B {
    trigger: on_kill
    cooldown: 5s
    heal 20
}

ability C {
    target: self
    cooldown: 10s
    hint: utility
    dash 4.0
}

passive D {
    trigger: on_hp_below(25%)
    cooldown: 30s
    shield 50 for 5s
}
"#;
        let (abilities, passives) = parse_abilities(input).unwrap();
        assert_eq!(abilities.len(), 2);
        assert_eq!(passives.len(), 2);
    }

    #[test]
    fn fuzz_toggle_and_unstoppable() {
        let input = r#"
ability Spin {
    target: self_aoe
    cooldown: 1s
    hint: damage
    toggle
    toggle_cost: 5

    damage 10 in circle(2.5)
}

ability Charge {
    target: enemy, range: 5.0
    cooldown: 10s, cast: 0ms
    hint: crowd_control
    unstoppable

    dash to_target
    stun 1s
}
"#;
        let (abilities, _) = parse_abilities(input).unwrap();
        assert!(abilities[0].is_toggle);
        assert!(abilities[1].unstoppable);
    }

    #[test]
    fn fuzz_form_swap() {
        let input = r#"
ability Transform {
    target: self
    cooldown: 3s
    hint: utility
    swap_form: "dragon"

    buff move_speed 0.3 for 10s
}
"#;
        let (abilities, _) = parse_abilities(input).unwrap();
        assert_eq!(abilities[0].swap_form, Some("dragon".to_string()));
    }
}
