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
                Effect::OnHitBuff { on_hit_effects, .. } => {
                    for child in on_hit_effects {
                        self.record_effect_tree(child);
                    }
                    "OnHitBuff"
                }
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
                "toggle", "unstoppable", "form",
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
            if self.has_toggle { hit_features.insert("toggle"); }
            if self.has_unstoppable { hit_features.insert("unstoppable"); }
            if self.has_form { hit_features.insert("form"); }

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

    fn multi_tag_strategy() -> impl Strategy<Value = String> {
        proptest::collection::vec(
            prop_oneof![
                Just("FIRE"), Just("ICE"), Just("LIGHTNING"), Just("POISON"),
                Just("DARK"), Just("HOLY"), Just("PHYSICAL"), Just("MAGIC"),
                Just("CROWD_CONTROL"),
            ].prop_flat_map(|tag| (Just(tag), 1u32..100).prop_map(|(t, v)| format!("{t}: {v}"))),
            1..=4,
        ).prop_map(|pairs| format!("[{}]", pairs.join(", ")))
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
    // Rich effects — every effect decorated with area + tags + conditions + scaling
    // -----------------------------------------------------------------------

    /// Area without the `in` prefix, for contexts where we provide it ourselves.
    fn bare_area_strategy() -> impl Strategy<Value = String> {
        prop_oneof![
            (1u32..100).prop_map(|r| format!("circle({}.0)", r)),
            (1u32..100, 10u32..180).prop_map(|(r, a)| format!("cone({}.0, {}.0)", r, a)),
            (1u32..100, 1u32..50).prop_map(|(l, w)| format!("line({}.0, {}.0)", l, w)),
            (1u32..50, 2u32..100).prop_map(|(i, o)| format!("ring({}.0, {}.0)", i, o)),
            (1u32..100, 1u32..10).prop_map(|(r, t)| format!("spread({}.0, {})", r, t)),
        ]
    }

    fn rich_damage() -> impl Strategy<Value = String> {
        (
            1i32..200,
            bare_area_strategy(),
            multi_tag_strategy(),
            simple_condition(),
            1u32..100,
            scaling_stat(),
        ).prop_map(|(amt, area, tags, cond, pct, stat)| {
            format!("    damage {amt} in {area} {tags} + {pct}% {stat} when {cond}")
        })
    }

    fn rich_heal() -> impl Strategy<Value = String> {
        (1i32..200, simple_condition(), 1u32..100, scaling_stat()).prop_map(|(amt, cond, pct, stat)| {
            format!("    heal {amt} + {pct}% {stat} when {cond}")
        })
    }

    fn rich_cc() -> impl Strategy<Value = String> {
        let cc = prop_oneof![
            Just("stun"), Just("root"), Just("silence"), Just("fear"),
            Just("suppress"), Just("grounded"), Just("charm"), Just("banish"),
        ];
        (cc, duration_strategy(), bare_area_strategy(), multi_tag_strategy(), simple_condition())
            .prop_map(|(cc, dur, area, tags, cond)| {
                format!("    {cc} {dur} in {area} {tags} when {cond}")
            })
    }

    fn rich_slow() -> impl Strategy<Value = String> {
        (1u32..9, duration_strategy(), bare_area_strategy(), multi_tag_strategy())
            .prop_map(|(f10, dur, area, tags)| {
                let f = f10 as f32 / 10.0;
                format!("    slow {f} for {dur} in {area} {tags}")
            })
    }

    fn rich_buff() -> impl Strategy<Value = String> {
        let stat = prop_oneof![
            Just("move_speed"), Just("attack_speed"), Just("damage_output"),
        ];
        (stat, 1u32..9, duration_strategy(), bare_area_strategy())
            .prop_map(|(s, f10, dur, area)| {
                let f = f10 as f32 / 10.0;
                format!("    buff {s} {f} for {dur} in {area}")
            })
    }

    fn any_rich_effect() -> impl Strategy<Value = String> {
        prop_oneof![
            3 => rich_damage(),
            2 => rich_heal(),
            2 => rich_cc(),
            2 => rich_slow(),
            2 => rich_buff(),
            // Sprinkle in plain effects so the abomination has variety
            1 => dash_effect(),
            1 => shield_effect(),
            1 => stealth_effect(),
            1 => summon_effect(),
            1 => lifesteal_effect(),
            1 => apply_stacks_effect(),
            1 => blind_effect(),
            1 => reflect_effect(),
            1 => execute_effect(),
        ]
    }

    fn rich_effect_list(min: usize, max: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(any_rich_effect(), min..=max)
            .prop_map(|effects| effects.join("\n"))
    }

    fn rich_delivery() -> impl Strategy<Value = String> {
        prop_oneof![
            // Projectile with on_hit + on_arrival both stuffed
            (
                (1u32..30).prop_map(|s| format!("{s}.0")),
                rich_effect_list(2, 4),
                rich_effect_list(1, 3),
            ).prop_map(|(speed, hit_effs, arrival_effs)| {
                format!(
                    "    deliver projectile {{ speed: {speed}, pierce, width: 0.5 }} {{\n        on_hit {{\n{hit_effs}\n        }}\n        on_arrival {{\n{arrival_effs}\n        }}\n    }}"
                )
            }),
            // Zone with rich tick effects
            (duration_strategy(), duration_strategy(), rich_effect_list(2, 4))
                .prop_map(|(dur, tick, effs)| {
                    format!(
                        "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effs}\n        }}\n    }}"
                    )
                }),
            // Channel with rich effects
            (duration_strategy(), duration_strategy(), rich_effect_list(2, 3))
                .prop_map(|(dur, tick, effs)| {
                    format!(
                        "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effs}\n        }}\n    }}"
                    )
                }),
            // Tether with rich on_complete
            ((2u32..10).prop_map(|r| format!("{r}.0")), rich_effect_list(2, 4))
                .prop_map(|(range, effs)| {
                    format!(
                        "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effs}\n        }}\n    }}"
                    )
                }),
        ]
    }

    /// A god ability: packs ALL 32 effect types, all 5 areas, all 10 conditions,
    /// and all ability-level features into a single ability via on_hit_buff nesting
    /// and delivery sub-hooks. Theoretical max: 54/79 features (triggers are passive-only).
    fn god_ability_block() -> impl Strategy<Value = String> {
        (
            ident_strategy(),
            // Random values for numeric fields
            (1u32..100).prop_map(|r| format!("{r}.0")),
            duration_strategy(),
            duration_strategy(),
        ).prop_map(|(name, range, cd, cast)| {
            // NOTE: `when` clauses inside delivery on_hit/on_arrival consume newlines
            // after parsing the condition (parser bug), swallowing subsequent effects.
            // So we keep delivery hooks plain and put `when` clauses on top-level effects.
            let lines = vec![
                format!("ability {name} {{"),
                format!("    target: enemy, range: {range}"),
                format!("    cooldown: {cd}, cast: {cast}"),
                "    hint: damage".to_string(),
                "    cost: 15".to_string(),
                "    charges: 3".to_string(),
                "    recharge: 8s".to_string(),
                "    recast: 2".to_string(),
                "    recast_window: 3s".to_string(),
                "    unstoppable".to_string(),
                "".to_string(),
                // Projectile delivery — no `when` or `+ scaling` in hooks
                // (parser eats newlines after those, swallowing next effects)
                "    deliver projectile { speed: 10.0, pierce, width: 0.5 } {".to_string(),
                "        on_hit {".to_string(),
                "            damage 50 [FIRE: 60, ICE: 20]".to_string(),
                "            heal 30".to_string(),
                "            stun 1s in circle(3.0)".to_string(),
                "            slow 0.4 for 2s in cone(4.0, 90.0)".to_string(),
                "            root 1500ms in line(5.0, 1.5)".to_string(),
                "            silence 2s".to_string(),
                "            fear 1s".to_string(),
                "            shield 40 for 4s in ring(1.0, 5.0)".to_string(),
                "            knockback 3.0".to_string(),
                "            pull 2.0".to_string(),
                "        }".to_string(),
                "        on_arrival {".to_string(),
                "            damage 20 in spread(3.0, 3) [LIGHTNING: 40]".to_string(),
                "            blind 0.5 for 2s".to_string(),
                "            taunt 1s".to_string(),
                "            charm 1500ms".to_string(),
                "            suppress 1s".to_string(),
                "            grounded 2s".to_string(),
                "            confuse 1s".to_string(),
                "            banish 1s".to_string(),
                "            polymorph 2s".to_string(),
                "        }".to_string(),
                "    }".to_string(),
                "".to_string(),
                // Each `when` clause must be the last effect in its block (parser
                // bug: `when` consumes newlines), so wrap each in on_hit_buff.
                "    on_hit_buff for 1s { damage 10 when target_hp_below(50%) }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_hp_above(80%) }".to_string(),
                "    on_hit_buff for 1s { heal 10 when caster_hp_below(30%) }".to_string(),
                "    on_hit_buff for 1s { heal 10 when caster_hp_above(60%) }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_is_stunned }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_is_slowed }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_is_rooted }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_is_silenced }".to_string(),
                "    on_hit_buff for 1s { damage 10 when target_is_feared }".to_string(),
                "    on_hit_buff for 1s { damage 10 when hit_count_above(3) }".to_string(),
                // Scaling also eats newlines, so wrap it too
                "    on_hit_buff for 1s { damage 10 + 15% target_max_hp }".to_string(),
                "".to_string(),
                // Movement + buffs + utility
                "    dash to_target".to_string(),
                "    blink 4.0".to_string(),
                "    buff move_speed 0.3 for 5s".to_string(),
                "    debuff attack_speed 0.4 for 3s".to_string(),
                "    damage_modify 1.5 for 4s".to_string(),
                "    lifesteal 0.3 for 5s".to_string(),
                "    reflect 0.5 for 3s".to_string(),
                "    stealth for 5s break_on_damage".to_string(),
                "    summon \"golem\" x2".to_string(),
                "    apply_stacks \"doom\" 2 max 5".to_string(),
                "    swap".to_string(),
                "    self_damage 25".to_string(),
                "    execute 20.0".to_string(),
                "    resurrect 50.0".to_string(),
                // on_hit_buff with nested child effects
                "    on_hit_buff for 6s {".to_string(),
                "        damage 10 [DARK: 30]".to_string(),
                "        slow 0.2 for 1s".to_string(),
                "    }".to_string(),
                "}".to_string(),
            ];
            lines.join("\n")
        })
    }

    /// An abomination: an ability that tries to use as many features as possible.
    fn abomination_block() -> impl Strategy<Value = String> {
        (
            ident_strategy(),
            targeting_strategy(),
            (1u32..100).prop_map(|r| format!("{r}.0")),
            duration_strategy(),
            duration_strategy(),
            hint_strategy(),
            1u32..30,                  // cost (always present)
            (1u32..5, duration_strategy()), // charges (always present)
            (1u32..4, duration_strategy()), // recast (always present)
            rich_delivery(),
            rich_effect_list(3, 8),    // extra effects on top of delivery
        )
            .prop_map(
                |(name, target, range, cd, cast, hint, cost, (ch, rech), (rc, rw), delivery, effects)| {
                    let mut lines = vec![format!("ability {name} {{")];
                    lines.push(format!("    target: {target}, range: {range}"));
                    lines.push(format!("    cooldown: {cd}, cast: {cast}"));
                    lines.push(format!("    hint: {hint}"));
                    lines.push(format!("    cost: {cost}"));
                    lines.push(format!("    charges: {ch}"));
                    lines.push(format!("    recharge: {rech}"));
                    lines.push(format!("    recast: {rc}"));
                    lines.push(format!("    recast_window: {rw}"));
                    lines.push(String::new());
                    lines.push(delivery);
                    lines.push(effects);
                    lines.push("}".to_string());
                    lines.join("\n")
                },
            )
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// Abomination bulk: 10-20 maximally complex abilities per case × 50 = 500-1000.
        #[test]
        fn fuzz_abomination_bulk(
            abilities in proptest::collection::vec(abomination_block(), 10..=20),
        ) {
            let input = abilities.join("\n\n");
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "Abomination failed:\n---ERROR---\n{}",
                result.unwrap_err()
            );
            let (abs, _) = result.unwrap();
            prop_assert!(abs.len() >= 10);
            for ab in &abs {
                prop_assert!(!ab.name.is_empty());
            }
        }
    }

    /// Generate 500 normal + 500 abomination abilities + 200 passives and report coverage.
    #[test]
    fn bulk_coverage_report() {
        use proptest::test_runner::{TestRunner, Config};
        use proptest::strategy::ValueTree;

        let config = Config { cases: 1, .. Config::default() };
        let mut runner = TestRunner::new(config);

        let normal_strat = proptest::collection::vec(ability_block(), 500..=500);
        let abom_strat = proptest::collection::vec(abomination_block(), 500..=500);
        let pa_strat = proptest::collection::vec(passive_block(), 200..=200);

        let normal_tree = normal_strat.new_tree(&mut runner).unwrap();
        let abom_tree = abom_strat.new_tree(&mut runner).unwrap();
        let pa_tree = pa_strat.new_tree(&mut runner).unwrap();

        let mut parts: Vec<String> = normal_tree.current();
        parts.extend(abom_tree.current());
        parts.extend(pa_tree.current());
        let input = parts.join("\n\n");

        let (abilities, passives) = parse_abilities(&input)
            .unwrap_or_else(|e| panic!("Failed to parse bulk generation:\n{e}"));

        assert_eq!(abilities.len(), 1000);
        assert_eq!(passives.len(), 200);

        // --- Global coverage ---
        let mut cov = Coverage::default();
        cov.record_abilities(&abilities, &passives);

        let report = cov.report();
        eprintln!("\n{report}");

        // --- Per-ability feature depth ---
        let mut per_ab_features: Vec<usize> = Vec::new();
        let mut per_ab_unique_effects: Vec<usize> = Vec::new();
        let mut per_ab_effect_counts: Vec<usize> = Vec::new();
        let mut abilities_with_delivery = 0usize;
        let mut abilities_with_condition = 0usize;
        let mut abilities_with_area = 0usize;
        let mut abilities_with_tags = 0usize;
        let mut abilities_with_scaling = 0usize;

        for ab in &abilities {
            let mut ab_cov = Coverage::default();
            ab_cov.record_abilities(std::slice::from_ref(ab), &[]);

            // Count distinct features this ability touches
            let feat_count = ab_cov.effects.len()
                + ab_cov.deliveries.len()
                + ab_cov.areas.len()
                + ab_cov.conditions.len()
                + (if ab_cov.has_charges { 1 } else { 0 })
                + (if ab_cov.has_recast { 1 } else { 0 })
                + (if ab_cov.has_cost { 1 } else { 0 })
                + (if ab_cov.has_tags { 1 } else { 0 })
                + (if ab_cov.has_scaling { 1 } else { 0 })
                + (if ab_cov.has_toggle { 1 } else { 0 })
                + (if ab_cov.has_unstoppable { 1 } else { 0 })
                + (if ab_cov.has_form { 1 } else { 0 });

            per_ab_features.push(feat_count);
            per_ab_unique_effects.push(ab_cov.effects.len());

            // Total effect count (including delivery sub-effects)
            let mut total_effects = ab.effects.len();
            if let Some(ref d) = ab.delivery {
                match d {
                    Delivery::Projectile { on_hit, on_arrival, .. } => {
                        total_effects += on_hit.len() + on_arrival.len();
                    }
                    Delivery::Chain { on_hit, .. } => total_effects += on_hit.len(),
                    Delivery::Tether { on_complete, .. } => total_effects += on_complete.len(),
                    _ => {}
                }
            }
            per_ab_effect_counts.push(total_effects);

            if ab.delivery.is_some() { abilities_with_delivery += 1; }
            if ab_cov.has_tags { abilities_with_tags += 1; }
            if ab_cov.has_scaling { abilities_with_scaling += 1; }
            if !ab_cov.conditions.is_empty() { abilities_with_condition += 1; }
            if !ab_cov.areas.is_empty() { abilities_with_area += 1; }
        }

        per_ab_features.sort();
        per_ab_unique_effects.sort();
        per_ab_effect_counts.sort();

        let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
        let percentile = |v: &[usize], p: usize| v[v.len() * p / 100];

        eprintln!("\nPer-ability stats (n=1000):");
        eprintln!("  Distinct features per ability:");
        eprintln!("    mean={:.1}  p5={}  p25={}  p50={}  p75={}  p95={}  max={}",
            mean(&per_ab_features),
            percentile(&per_ab_features, 5),
            percentile(&per_ab_features, 25),
            percentile(&per_ab_features, 50),
            percentile(&per_ab_features, 75),
            percentile(&per_ab_features, 95),
            per_ab_features.last().unwrap(),
        );
        eprintln!("  Unique effect types per ability:");
        eprintln!("    mean={:.1}  p5={}  p25={}  p50={}  p75={}  p95={}  max={}",
            mean(&per_ab_unique_effects),
            percentile(&per_ab_unique_effects, 5),
            percentile(&per_ab_unique_effects, 25),
            percentile(&per_ab_unique_effects, 50),
            percentile(&per_ab_unique_effects, 75),
            percentile(&per_ab_unique_effects, 95),
            per_ab_unique_effects.last().unwrap(),
        );
        eprintln!("  Total effects per ability:");
        eprintln!("    mean={:.1}  p5={}  p25={}  p50={}  p75={}  p95={}  max={}",
            mean(&per_ab_effect_counts),
            percentile(&per_ab_effect_counts, 5),
            percentile(&per_ab_effect_counts, 25),
            percentile(&per_ab_effect_counts, 50),
            percentile(&per_ab_effect_counts, 75),
            percentile(&per_ab_effect_counts, 95),
            per_ab_effect_counts.last().unwrap(),
        );
        eprintln!("  Abilities with delivery:  {:>4} ({:.0}%)", abilities_with_delivery, abilities_with_delivery as f64 / 10.0);
        eprintln!("  Abilities with area:      {:>4} ({:.0}%)", abilities_with_area, abilities_with_area as f64 / 10.0);
        eprintln!("  Abilities with tags:      {:>4} ({:.0}%)", abilities_with_tags, abilities_with_tags as f64 / 10.0);
        eprintln!("  Abilities with condition: {:>4} ({:.0}%)", abilities_with_condition, abilities_with_condition as f64 / 10.0);
        eprintln!("  Abilities with scaling:   {:>4} ({:.0}%)", abilities_with_scaling, abilities_with_scaling as f64 / 10.0);
        eprintln!();

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

        // Assert per-ability diversity minimums
        assert!(mean(&per_ab_unique_effects) >= 1.5,
            "mean unique effects per ability too low: {:.1}", mean(&per_ab_unique_effects));
        assert!(mean(&per_ab_features) >= 2.0,
            "mean features per ability too low: {:.1}", mean(&per_ab_features));
    }

    // -----------------------------------------------------------------------
    // God ability — single ability coverage test
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn fuzz_god_ability(input in god_ability_block()) {
            let result = parse_abilities(&input);
            prop_assert!(
                result.is_ok(),
                "God ability failed:\n---INPUT---\n{}\n---ERROR---\n{}",
                input,
                result.unwrap_err()
            );
            let (abilities, _) = result.unwrap();
            prop_assert_eq!(abilities.len(), 1);
        }
    }

    #[test]
    fn god_ability_coverage() {
        use proptest::test_runner::{TestRunner, Config};
        use proptest::strategy::ValueTree;

        let config = Config { cases: 1, .. Config::default() };
        let mut runner = TestRunner::new(config);

        let tree = god_ability_block().new_tree(&mut runner).unwrap();
        let input = tree.current();

        let (abilities, _) = parse_abilities(&input)
            .unwrap_or_else(|e| panic!("God ability failed to parse:\n{input}\n---\n{e}"));

        assert_eq!(abilities.len(), 1);

        let mut cov = Coverage::default();
        cov.record_abilities(&abilities, &[]);

        let report = cov.report();
        eprintln!("\n=== God Ability Coverage ===\n{report}");

        // Count features this single ability covers
        let mut hit_features = BTreeSet::new();
        if cov.has_charges { hit_features.insert("charges"); }
        if cov.has_recast { hit_features.insert("recast"); }
        if cov.has_cost { hit_features.insert("cost"); }
        if cov.has_tags { hit_features.insert("tags"); }
        if cov.has_scaling { hit_features.insert("scaling"); }
        if cov.has_toggle { hit_features.insert("toggle"); }
        if cov.has_unstoppable { hit_features.insert("unstoppable"); }
        if cov.has_form { hit_features.insert("form"); }

        let total = cov.effects.len()
            + cov.deliveries.len()
            + cov.areas.len()
            + cov.conditions.len()
            + cov.targetings.len()
            + hit_features.len();

        eprintln!("Single ability feature count: {total}");
        eprintln!("  effects: {} {:?}", cov.effects.len(), cov.effects);
        eprintln!("  deliveries: {} {:?}", cov.deliveries.len(), cov.deliveries);
        eprintln!("  areas: {} {:?}", cov.areas.len(), cov.areas);
        eprintln!("  conditions: {} {:?}", cov.conditions.len(), cov.conditions);
        eprintln!("  targeting: {} {:?}", cov.targetings.len(), cov.targetings);
        eprintln!("  features: {} {:?}", hit_features.len(), hit_features);

        // A god ability should cover at least 45 of 57 ability-applicable features
        // (32 effects + 1 delivery + 5 areas + 10 conditions + 1 targeting + 8 features)
        // We get ~50+ since we pack all 32 effects, all 5 areas, all 10 conditions
        assert!(total >= 45,
            "God ability should cover ≥45 features, got {total}. Report:\n{report}");
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

    // -----------------------------------------------------------------------
    // Realistic parameter strategies — calibrated from real hero data
    // -----------------------------------------------------------------------

    /// Cooldowns: 3–30s, weighted toward 5–12s (matches hero template median)
    fn realistic_cooldown() -> impl Strategy<Value = String> {
        prop_oneof![
            2 => (3u32..8).prop_map(|s| format!("{s}s")),      // short: 3–7s
            4 => (8u32..15).prop_map(|s| format!("{s}s")),     // medium: 8–14s
            2 => (15u32..30).prop_map(|s| format!("{s}s")),    // long: 15–29s
            1 => (30u32..60).prop_map(|s| format!("{s}s")),    // ultimate: 30–59s
        ]
    }

    /// Cast times: most are instant or short
    fn realistic_cast() -> impl Strategy<Value = String> {
        prop_oneof![
            3 => Just("0ms".to_string()),                       // instant
            3 => (100u32..500).prop_map(|ms| format!("{ms}ms")), // short: 100–499ms
            2 => (500u32..1500).prop_map(|ms| format!("{ms}ms")),// medium: 500ms–1.5s
            1 => (2u32..4).prop_map(|s| format!("{s}s")),       // long: 2–3s
        ]
    }

    /// Effect durations (cc, buff, shield): 0.5–8s, weighted toward 1–4s
    fn realistic_effect_duration() -> impl Strategy<Value = String> {
        prop_oneof![
            2 => (300u32..1000).prop_map(|ms| format!("{ms}ms")),// brief: 0.3–1s
            4 => (1u32..4).prop_map(|s| format!("{s}s")),       // standard: 1–3s
            2 => (4u32..8).prop_map(|s| format!("{s}s")),       // long: 4–7s
            1 => (8u32..15).prop_map(|s| format!("{s}s")),      // very long: 8–14s
        ]
    }

    /// Ranges: 1–8, matching real hero data (most at 3–6)
    fn realistic_range() -> impl Strategy<Value = String> {
        prop_oneof![
            2 => (1u32..3).prop_map(|r| format!("{r}.0")),     // melee: 1–2
            4 => (3u32..7).prop_map(|r| format!("{r}.0")),     // standard: 3–6
            2 => (7u32..10).prop_map(|r| format!("{r}.0")),    // long: 7–9
        ]
    }

    /// Damage amounts: 15–80, matching real hero data
    fn realistic_damage() -> impl Strategy<Value = String> {
        (15i32..80).prop_map(|d| d.to_string())
    }

    /// Heal amounts: 15–60
    fn realistic_heal() -> impl Strategy<Value = String> {
        (15i32..60).prop_map(|d| d.to_string())
    }

    /// Shield amounts: 20–60
    fn realistic_shield() -> impl Strategy<Value = String> {
        (20i32..60).prop_map(|d| d.to_string())
    }

    /// Area shapes with realistic radii
    fn realistic_area() -> impl Strategy<Value = String> {
        prop_oneof![
            (2u32..6).prop_map(|r| format!("in circle({r}.0)")),
            (3u32..6, 30u32..120).prop_map(|(r, a)| format!("in cone({r}.0, {a}.0)")),
            (3u32..8, 1u32..3).prop_map(|(l, w)| format!("in line({l}.0, {w}.0)")),
            (1u32..3, 3u32..7).prop_map(|(i, o)| format!("in ring({i}.0, {o}.0)")),
            (2u32..5, 2u32..5).prop_map(|(r, t)| format!("in spread({r}.0, {t})")),
        ]
    }

    fn realistic_opt_area() -> impl Strategy<Value = String> {
        prop_oneof![3 => Just(String::new()), 1 => realistic_area()]
    }

    // -----------------------------------------------------------------------
    // Archetype-based ability generators — hint-coherent effect selection
    // -----------------------------------------------------------------------

    /// Damage archetype: primary damage + optional secondary (slow, scaling, area)
    fn damage_archetype_effects() -> impl Strategy<Value = String> {
        prop_oneof![
            // Pure damage
            (realistic_damage(), opt_tags()).prop_map(|(d, tags)|
                format!("    damage {d} {tags}")),
            // Damage + area
            (realistic_damage(), realistic_area(), opt_tags()).prop_map(|(d, area, tags)|
                format!("    damage {d} {area} {tags}")),
            // Damage + slow
            (realistic_damage(), 2u32..6, realistic_effect_duration(), opt_tags())
                .prop_map(|(d, slow10, dur, tags)| {
                    let slow = slow10 as f32 / 10.0;
                    format!("    damage {d} {tags}\n    slow {slow} for {dur}")
                }),
            // Damage + scaling
            (realistic_damage(), 5u32..30, scaling_stat(), opt_tags())
                .prop_map(|(d, pct, stat, tags)|
                    format!("    damage {d} + {pct}% {stat} {tags}")),
            // Damage + knockback
            (realistic_damage(), 2u32..5, opt_tags()).prop_map(|(d, kb, tags)|
                format!("    damage {d} {tags}\n    knockback {kb}.0")),
        ]
    }

    /// Heal archetype: heal + optional shield/buff
    fn heal_archetype_effects() -> impl Strategy<Value = String> {
        prop_oneof![
            // Pure heal
            realistic_heal().prop_map(|h| format!("    heal {h}")),
            // Heal + area
            (realistic_heal(), realistic_area()).prop_map(|(h, area)|
                format!("    heal {h} {area}")),
            // Heal + shield
            (realistic_heal(), realistic_shield(), realistic_effect_duration())
                .prop_map(|(h, s, dur)|
                    format!("    heal {h}\n    shield {s} for {dur}")),
            // Heal + buff
            (realistic_heal(), 1u32..4, realistic_effect_duration()).prop_map(|(h, f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    heal {h}\n    buff move_speed {f} for {dur}")
            }),
            // Conditional heal
            (realistic_heal(), simple_condition()).prop_map(|(h, cond)|
                format!("    heal {h} when {cond}")),
        ]
    }

    /// CC archetype: primary cc + optional secondary
    fn cc_archetype_effects() -> impl Strategy<Value = String> {
        let cc_type = prop_oneof![
            3 => Just("stun"), 3 => Just("root"), 2 => Just("silence"),
            1 => Just("fear"), 1 => Just("taunt"),
        ];
        prop_oneof![
            // Pure CC
            (cc_type.clone(), realistic_effect_duration()).prop_map(|(cc, dur)|
                format!("    {cc} {dur}")),
            // CC + area
            (cc_type.clone(), realistic_effect_duration(), realistic_area())
                .prop_map(|(cc, dur, area)|
                    format!("    {cc} {dur} {area}")),
            // CC + damage
            (cc_type.clone(), realistic_effect_duration(), realistic_damage())
                .prop_map(|(cc, dur, d)|
                    format!("    {cc} {dur}\n    damage {d}")),
            // Slow + damage
            (2u32..6, realistic_effect_duration(), realistic_damage(), realistic_opt_area())
                .prop_map(|(slow10, dur, d, area)| {
                    let slow = slow10 as f32 / 10.0;
                    format!("    slow {slow} for {dur} {area}\n    damage {d}")
                }),
        ]
    }

    /// Buff/utility archetype: buff/debuff + optional secondary
    fn buff_archetype_effects() -> impl Strategy<Value = String> {
        let buff_stat = prop_oneof![
            Just("move_speed"), Just("attack_speed"),
            Just("damage_output"), Just("cooldown_reduction"),
        ];
        prop_oneof![
            // Pure buff
            (buff_stat.clone(), 1u32..5, realistic_effect_duration())
                .prop_map(|(s, f10, dur)| {
                    let f = f10 as f32 / 10.0;
                    format!("    buff {s} {f} for {dur}")
                }),
            // Buff + shield
            (buff_stat.clone(), 1u32..5, realistic_effect_duration(), realistic_shield(), realistic_effect_duration())
                .prop_map(|(s, f10, dur, sh, sh_dur)| {
                    let f = f10 as f32 / 10.0;
                    format!("    buff {s} {f} for {dur}\n    shield {sh} for {sh_dur}")
                }),
            // Buff area
            (buff_stat.clone(), 1u32..4, realistic_effect_duration(), realistic_area())
                .prop_map(|(s, f10, dur, area)| {
                    let f = f10 as f32 / 10.0;
                    format!("    buff {s} {f} for {dur} {area}")
                }),
            // Damage modify
            (10u32..20, realistic_effect_duration()).prop_map(|(f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    damage_modify {f} for {dur}")
            }),
            // Lifesteal
            (2u32..5, realistic_effect_duration()).prop_map(|(f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    lifesteal {f} for {dur}")
            }),
        ]
    }

    /// Defense archetype: shield/stealth/reflect
    fn defense_archetype_effects() -> impl Strategy<Value = String> {
        prop_oneof![
            // Shield
            (realistic_shield(), realistic_effect_duration()).prop_map(|(s, dur)|
                format!("    shield {s} for {dur}")),
            // Shield + heal
            (realistic_shield(), realistic_effect_duration(), realistic_heal())
                .prop_map(|(s, dur, h)|
                    format!("    shield {s} for {dur}\n    heal {h}")),
            // Reflect
            (2u32..6, realistic_effect_duration()).prop_map(|(f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    reflect {f} for {dur}")
            }),
            // Stealth
            realistic_effect_duration().prop_map(|dur|
                format!("    stealth for {dur} break_on_damage")),
            // Shield + buff
            (realistic_shield(), realistic_effect_duration(), 1u32..4, realistic_effect_duration())
                .prop_map(|(s, s_dur, f10, b_dur)| {
                    let f = f10 as f32 / 10.0;
                    format!("    shield {s} for {s_dur}\n    buff armor {f} for {b_dur}")
                }),
        ]
    }

    /// Utility archetype: dash/blink/swap/misc
    fn utility_archetype_effects() -> impl Strategy<Value = String> {
        prop_oneof![
            // Dash
            Just("    dash to_target".to_string()),
            Just("    dash to_position".to_string()),
            (2u32..6).prop_map(|d| format!("    blink {d}.0")),
            // Dash + buff
            (2u32..4, realistic_effect_duration()).prop_map(|(f10, dur)| {
                let f = f10 as f32 / 10.0;
                format!("    dash to_target\n    buff move_speed {f} for {dur}")
            }),
            // Swap
            Just("    swap".to_string()),
            // Dispel
            Just("    dispel".to_string()),
        ]
    }

    /// Pick archetype effects matching the given hint.
    fn archetype_effects(hint: &'static str) -> proptest::strategy::BoxedStrategy<String> {
        match hint {
            "damage" => damage_archetype_effects().boxed(),
            "heal" => heal_archetype_effects().boxed(),
            "crowd_control" => cc_archetype_effects().boxed(),
            "buff" => buff_archetype_effects().boxed(),
            "defense" => defense_archetype_effects().boxed(),
            "utility" => utility_archetype_effects().boxed(),
            _ => damage_archetype_effects().boxed(),
        }
    }

    /// Realistic delivery with sane parameters
    fn realistic_delivery() -> impl Strategy<Value = String> {
        prop_oneof![
            // Projectile
            3 => (5u32..15, proptest::bool::ANY, effect_list(1, 2))
                .prop_map(|(speed, pierce, effects)| {
                    let pierce_str = if pierce { ", pierce" } else { "" };
                    format!(
                        "    deliver projectile {{ speed: {speed}.0{pierce_str} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                    )
                }),
            // Chain
            1 => (2u32..5, (3u32..6).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
                .prop_map(|(bounces, range, effects)|
                    format!(
                        "    deliver chain {{ bounces: {bounces}, range: {range} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                    )
                ),
            // Zone
            2 => ((3u32..10).prop_map(|s| format!("{s}s")), (500u32..2000).prop_map(|ms| format!("{ms}ms")), effect_list(1, 2))
                .prop_map(|(dur, tick, effects)|
                    format!(
                        "    deliver zone {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                    )
                ),
            // Channel
            1 => ((2u32..5).prop_map(|s| format!("{s}s")), (300u32..1000).prop_map(|ms| format!("{ms}ms")), effect_list(1, 2))
                .prop_map(|(dur, tick, effects)|
                    format!(
                        "    deliver channel {{ duration: {dur}, tick: {tick} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                    )
                ),
            // Tether
            1 => ((3u32..7).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
                .prop_map(|(range, effects)|
                    format!(
                        "    deliver tether {{ max_range: {range} }} {{\n        on_complete {{\n{effects}\n        }}\n    }}"
                    )
                ),
            // Trap
            1 => ((5u32..15).prop_map(|s| format!("{s}s")), (1u32..3).prop_map(|r| format!("{r}.0")), effect_list(1, 2))
                .prop_map(|(dur, radius, effects)|
                    format!(
                        "    deliver trap {{ duration: {dur}, trigger_radius: {radius} }} {{\n        on_hit {{\n{effects}\n        }}\n    }}"
                    )
                ),
        ]
    }

    /// Coherent ability: archetype-driven, realistic parameters, hint-matched effects.
    fn coherent_ability() -> impl Strategy<Value = String> {
        // Pick hint first, then build coherent ability around it
        prop_oneof![
            4 => Just("damage"),
            2 => Just("heal"),
            2 => Just("crowd_control"),
            2 => Just("buff"),
            2 => Just("defense"),
            1 => Just("utility"),
        ]
        .prop_flat_map(|hint| {
            (
                ident_strategy(),
                // Targeting coherent with hint
                match hint {
                    "heal" => prop_oneof![3 => Just("ally"), 1 => Just("self"), 1 => Just("self_aoe")].boxed(),
                    "buff" | "defense" => prop_oneof![2 => Just("self"), 2 => Just("ally"), 1 => Just("self_aoe")].boxed(),
                    "utility" => prop_oneof![Just("self"), Just("enemy"), Just("ground")].boxed(),
                    _ => prop_oneof![3 => Just("enemy"), 1 => Just("ground"), 1 => Just("direction")].boxed(),
                },
                realistic_range(),
                realistic_cooldown(),
                realistic_cast(),
                Just(hint),
                // Optional cost (30% chance)
                proptest::option::weighted(0.3, 5u32..25),
                // Optional charges (20% chance)
                proptest::option::weighted(0.2, (1u32..4, (5u32..15).prop_map(|s| format!("{s}s")))),
                // Optional recast (15% chance)
                proptest::option::weighted(0.15, (1u32..3, (2u32..5).prop_map(|s| format!("{s}s")))),
                // Effects: archetype-driven, optionally with delivery
                prop_oneof![
                    3 => archetype_effects(hint).prop_map(|e| (e, None)),
                    2 => (archetype_effects(hint), realistic_delivery().prop_map(Some))
                        .prop_map(|(e, d)| (e, d)),
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

                        lines.push(String::new());

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
        })
    }

    /// Realistic passive with sane parameters
    fn realistic_passive() -> impl Strategy<Value = String> {
        (
            ident_strategy(),
            trigger_strategy(),
            realistic_cooldown(),
            effect_list(1, 2),
        )
            .prop_map(|(name, trigger, cd, effects)| {
                format!(
                    "passive {name} {{\n    trigger: {trigger}\n    cooldown: {cd}\n\n{effects}\n}}"
                )
            })
    }

    // -----------------------------------------------------------------------
    // Dataset generator — 75k diverse abilities written to disk
    // -----------------------------------------------------------------------

    /// Generate 75k abilities across multiple strategies/seeds and write to
    /// `generated/ability_dataset/`. Each ability is a separate .ability file.
    /// Run with: cargo test generate_ability_dataset -- --ignored --nocapture
    #[test]
    #[ignore] // expensive — run manually
    fn generate_ability_dataset() {
        use proptest::test_runner::{TestRunner, Config, TestRng, RngAlgorithm};
        use proptest::strategy::ValueTree;
        use std::collections::HashSet;
        use std::fs;
        use std::path::Path;

        let out_dir = Path::new("generated/ability_dataset");
        if out_dir.exists() {
            fs::remove_dir_all(out_dir).unwrap();
        }
        fs::create_dir_all(out_dir).unwrap();

        let mut all_dsl: Vec<String> = Vec::with_capacity(80_000);
        let mut seen_names: HashSet<String> = HashSet::new();

        fn make_seed(n: u64) -> [u8; 32] {
            let mut seed = [0u8; 32];
            for (i, chunk) in seed.chunks_exact_mut(8).enumerate() {
                let val = n.wrapping_add(i as u64);
                chunk.copy_from_slice(&val.to_le_bytes());
            }
            seed
        }

        fn make_runner(seed_val: u64) -> TestRunner {
            let config = Config { cases: 1, .. Config::default() };
            TestRunner::new_with_rng(config, TestRng::from_seed(
                RngAlgorithm::ChaCha, &make_seed(seed_val),
            ))
        }

        // Mix: 95% coherent + 5% grammar coverage (normal/complex/god)
        // Coherent abilities have hint-aligned effects and realistic numerics,
        // giving the transformer consistent semantic patterns to learn from.
        // The 5% unrestricted tier ensures full grammar coverage.

        // --- Coherent abilities: 288 seeds × 250 = 72,000 ---
        let batch = 250;
        for seed in 0..288u64 {
            let mut runner = make_runner(seed);
            let strat = proptest::collection::vec(coherent_ability(), batch..=batch);
            let tree = strat.new_tree(&mut runner).unwrap();
            all_dsl.extend(tree.current());
        }
        eprintln!("Generated {} coherent abilities", all_dsl.len());

        // --- Normal abilities (unrestricted params): 6 seeds × 250 = 1,500 ---
        let normal_start = all_dsl.len();
        for seed in 0..6u64 {
            let mut runner = make_runner(seed + 1000);
            let strat = proptest::collection::vec(ability_block(), batch..=batch);
            let tree = strat.new_tree(&mut runner).unwrap();
            all_dsl.extend(tree.current());
        }
        eprintln!("Generated {} normal abilities", all_dsl.len() - normal_start);

        // --- Complex (abominations): 4 seeds × 250 = 1,000 ---
        let abom_start = all_dsl.len();
        for seed in 0..4u64 {
            let mut runner = make_runner(seed + 2000);
            let strat = proptest::collection::vec(abomination_block(), batch..=batch);
            let tree = strat.new_tree(&mut runner).unwrap();
            all_dsl.extend(tree.current());
        }
        eprintln!("Generated {} abomination abilities", all_dsl.len() - abom_start);

        // --- God abilities: 2 seeds × 250 = 500 ---
        let god_start = all_dsl.len();
        for seed in 0..2u64 {
            let mut runner = make_runner(seed + 3000);
            let strat = proptest::collection::vec(god_ability_block(), batch..=batch);
            let tree = strat.new_tree(&mut runner).unwrap();
            all_dsl.extend(tree.current());
        }
        eprintln!("Generated {} god abilities", all_dsl.len() - god_start);

        eprintln!("Total generated: {}", all_dsl.len());

        // --- Deduplicate by ability name, assign unique names ---
        let mut written = 0usize;
        let mut parse_failures = 0usize;
        let target = 75_000;

        for (i, dsl) in all_dsl.iter().enumerate() {
            let result = parse_abilities(dsl);
            let (abilities, _) = match result {
                Ok(v) => v,
                Err(_) => { parse_failures += 1; continue; }
            };
            if abilities.is_empty() { continue; }

            let base_name = &abilities[0].name;
            let unique_name = if seen_names.contains(base_name) {
                format!("{base_name}_{i}")
            } else {
                seen_names.insert(base_name.clone());
                base_name.clone()
            };

            let fixed_dsl = if unique_name != *base_name {
                let header = format!("ability {base_name} {{");
                let new_header = format!("ability {unique_name} {{");
                dsl.replacen(&header, &new_header, 1)
            } else {
                dsl.clone()
            };

            let path = out_dir.join(format!("{unique_name}.ability"));
            fs::write(&path, &fixed_dsl).unwrap();
            written += 1;

            if written >= target { break; }
        }

        eprintln!("\nDataset: {written} abilities written to {}", out_dir.display());
        if parse_failures > 0 {
            eprintln!("  ({parse_failures} parse failures skipped)");
        }

        // --- Verify a random sample parses ---
        let entries: Vec<_> = fs::read_dir(out_dir).unwrap()
            .filter_map(|e| e.ok())
            .collect();
        let sample_size = 200.min(entries.len());
        let mut sample_ok = 0;
        for entry in entries.iter().take(sample_size) {
            let content = fs::read_to_string(entry.path()).unwrap();
            if parse_abilities(&content).is_ok() {
                sample_ok += 1;
            }
        }
        eprintln!("  Sample validation: {sample_ok}/{sample_size} parsed OK");
        assert_eq!(sample_ok, sample_size, "all sampled abilities should parse");
        assert!(written >= target, "expected ≥{target} abilities, got {written}");
    }
}
