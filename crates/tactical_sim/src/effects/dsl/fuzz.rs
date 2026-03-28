//! Property-based fuzzer for the ability DSL.
//!
//! Generates structurally valid `.ability` DSL strings and verifies that the
//! full pipeline (parse -> lower) succeeds on every generated input.

#[cfg(test)]
pub(super) mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use crate::effects::defs::{AbilityDef, AbilityTargeting, PassiveDef};
    use crate::effects::dsl::parse_abilities;
    use crate::effects::effect_enum::Effect;
    use crate::effects::types::*;

    use super::super::fuzz_generators::*;

    // -----------------------------------------------------------------------
    // Coverage tracker
    // -----------------------------------------------------------------------

    #[derive(Default)]
    pub(in crate::effects::dsl) struct Coverage {
        pub(in crate::effects::dsl) effects: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) deliveries: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) areas: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) conditions: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) triggers: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) targetings: BTreeSet<&'static str>,
        pub(in crate::effects::dsl) has_charges: bool,
        pub(in crate::effects::dsl) has_recast: bool,
        pub(in crate::effects::dsl) has_cost: bool,
        pub(in crate::effects::dsl) has_toggle: bool,
        pub(in crate::effects::dsl) has_unstoppable: bool,
        pub(in crate::effects::dsl) has_form: bool,
        pub(in crate::effects::dsl) has_tags: bool,
        pub(in crate::effects::dsl) has_scaling: bool,
    }

    impl Coverage {
        pub(in crate::effects::dsl) fn record_abilities(&mut self, abilities: &[AbilityDef], passives: &[PassiveDef]) {
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
                // Campaign effects — catch-all
                _ => "CampaignEffect",
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
                // Campaign conditions — catch-all
                _ => "CampaignCondition",
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
                // Campaign triggers — catch-all
                _ => "CampaignTrigger",
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
                // Campaign targeting — catch-all
                _ => "CampaignTargeting",
            });
        }

        pub(in crate::effects::dsl) fn report(&self) -> String {
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
    // Core property tests
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
    // Bulk generation
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

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

        let mut cov = Coverage::default();
        cov.record_abilities(&abilities, &passives);

        let report = cov.report();
        eprintln!("\n{report}");

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
            mean(&per_ab_features), percentile(&per_ab_features, 5),
            percentile(&per_ab_features, 25), percentile(&per_ab_features, 50),
            percentile(&per_ab_features, 75), percentile(&per_ab_features, 95),
            per_ab_features.last().unwrap());
        eprintln!("  Unique effect types per ability:");
        eprintln!("    mean={:.1}  p5={}  p25={}  p50={}  p75={}  p95={}  max={}",
            mean(&per_ab_unique_effects), percentile(&per_ab_unique_effects, 5),
            percentile(&per_ab_unique_effects, 25), percentile(&per_ab_unique_effects, 50),
            percentile(&per_ab_unique_effects, 75), percentile(&per_ab_unique_effects, 95),
            per_ab_unique_effects.last().unwrap());
        eprintln!("  Total effects per ability:");
        eprintln!("    mean={:.1}  p5={}  p25={}  p50={}  p75={}  p95={}  max={}",
            mean(&per_ab_effect_counts), percentile(&per_ab_effect_counts, 5),
            percentile(&per_ab_effect_counts, 25), percentile(&per_ab_effect_counts, 50),
            percentile(&per_ab_effect_counts, 75), percentile(&per_ab_effect_counts, 95),
            per_ab_effect_counts.last().unwrap());
        eprintln!("  Abilities with delivery:  {:>4} ({:.0}%)", abilities_with_delivery, abilities_with_delivery as f64 / 10.0);
        eprintln!("  Abilities with area:      {:>4} ({:.0}%)", abilities_with_area, abilities_with_area as f64 / 10.0);
        eprintln!("  Abilities with tags:      {:>4} ({:.0}%)", abilities_with_tags, abilities_with_tags as f64 / 10.0);
        eprintln!("  Abilities with condition: {:>4} ({:.0}%)", abilities_with_condition, abilities_with_condition as f64 / 10.0);
        eprintln!("  Abilities with scaling:   {:>4} ({:.0}%)", abilities_with_scaling, abilities_with_scaling as f64 / 10.0);
        eprintln!();

        let effect_count = cov.effects.len();
        let delivery_count = cov.deliveries.len();
        let area_count = cov.areas.len();
        let trigger_count = cov.triggers.len();
        let targeting_count = cov.targetings.len();

        assert!(effect_count >= 25,
            "expected >=25 effect types covered, got {effect_count}. Report:\n{report}");
        assert!(delivery_count >= 6,
            "expected all 6 delivery types covered, got {delivery_count}. Report:\n{report}");
        assert!(area_count >= 5,
            "expected all 5 area types covered, got {area_count}. Report:\n{report}");
        assert!(trigger_count >= 10,
            "expected >=10 trigger types covered, got {trigger_count}. Report:\n{report}");
        assert!(targeting_count >= 8,
            "expected all 8 targeting types covered, got {targeting_count}. Report:\n{report}");

        assert!(mean(&per_ab_unique_effects) >= 1.5,
            "mean unique effects per ability too low: {:.1}", mean(&per_ab_unique_effects));
        assert!(mean(&per_ab_features) >= 2.0,
            "mean features per ability too low: {:.1}", mean(&per_ab_features));
    }
}
