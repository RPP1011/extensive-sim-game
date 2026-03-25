# TWI-Inspired Class System — Implementation Plan

Inspired by The Wandering Inn: classes are EARNED from behavior, not chosen.
The system watches what you do and responds.

## Phase 1: Core Data Model [DONE]
- [x] `BehaviorLedger` struct on Adventurer — 12 lifetime + 12 recent-window counters
- [x] `ClassInstance` struct — class_name, level, xp, xp_to_next, stagnation_ticks, skills_granted
- [x] `Vec<ClassInstance>` on Adventurer (multi-class)
- [x] `ClassTemplate` for matching behavior patterns to class grants
- [x] Quadratic XP curve: level N costs N^2 * 100
- [x] Wire into state.rs with #[serde(default)]

## Phase 2: Behavior Tracking [DONE]
- [x] `update_behavior_ledgers()` — increment counters from WorldEvents each tick
- [x] Action categories: melee_combat, ranged_combat, healing, diplomacy, trading, crafting, exploring, leading, sneaking, researching, defending, supporting
- [x] Rolling window with 0.95 exponential decay
- [x] Behavioral fingerprint: normalized 12-dim vector from ledger

## Phase 3: Class Acquisition [DONE]
- [x] `check_class_acquisition()` — compare fingerprints against templates
- [x] 11 base templates: Warrior, Ranger, Healer, Diplomat, Merchant, Scholar, Rogue, Artisan, Commander, Scout, Guardian
- [x] 4 rare intersection classes: Spellblade, Plague Doctor, Shadowmerchant, Warlord (crisis-only)
- [x] Negative-space classes from avoidance (Peacemaker, Hermit)
- [x] Exclusion zones — competing families locked out for cooldown period
- [x] Landmark achievement prerequisites for rare classes
- [x] Cross-system behavioral fingerprinting for unique classes

## Phase 4: Multi-Class Leveling [DONE]
- [x] Independent XP pools filled by tagged actions
- [x] Quadratic cost curve per class
- [x] Stagnation timer: 500 ticks → halved XP, 1000 → frozen
- [x] Class resonance: 15% XP trickle between overlapping classes
- [x] Activity recency weighting for stat bonuses
- [x] Witness XP multiplier (up to 2x near higher-level same-class)
- [x] Soft level caps with capstone events (level 20+)
- [x] Hybrid class slots (two classes 15+ with 300 co-active ticks)

## Phase 5: Skill Granting [DONE]
- [x] Level thresholds: 3, 5, 7, 10, 15, 20, 25 (to be expanded to 100)
- [x] Behavior vector scoring for skill selection
- [x] Rarity tiers: Common, Uncommon, Rare, Capstone, Unique
- [x] Campaign system hooks: bankruptcy → [Debt Remembrance], civil war → [Oathbreaker's Resolve], etc.
- [x] Cross-class skill contamination: old class behavior bleeds at 25%
- [x] Skill suppression: Precise Strike suppresses Wild Swing, etc.
- [x] Capstone skill synthesis from top 2 behavior axes
- [x] Skill interaction bonuses (empowered flag from shared affinity tags)
- [x] Narrative announcement system (Common→text, Rare→dramatic, Capstone→eerie voice)

## Phase 6: Class Consolidation & Evolution [DONE]
- [x] Consolidation offers when two classes both at level 10+
- [x] Template-based merged class names (18 curated combos + fallback)
- [x] Refusal banking: refused offers upgrade rarity
- [x] Evolution from achievements (level 20+ with 50 battles, 20 quests)
- [x] World-event-gated unique classes (Arbiter, Plague Sovereign, Market Maker, etc.)
- [x] Unique holder tracking via unique_class_holders HashMap
- [x] Skill-intersection scoring (Jaccard similarity) for consolidation quality
- [x] Skill inheritance rules: core/vestigial/lost
- [x] Consolidation rarity tiers with unique holder enforcement
- [x] Crisis escape valve consolidation (emergency merge during high tension)

## Phase 7: Reactive Narrative [DONE]
- [x] Shame classes: [Coward], [Oathbreaker], [Deserter] — permanent, suppress stats 20%
- [x] Crisis grants: [The Last Wall], [Mercy in Iron], [Risen Commander], [The Unkillable]
- [x] Identity erosion: coherence decays when actions contradict class
- [x] Mirror offers: forced choice between contradictory class paths
- [x] Witnessed vs unwitnessed acts → public vs hidden classes
- [x] Rival-reflected classes (reactive to rival's class)
- [x] Oath-locked class ascension (Paladin, Truthspeaker, Sworn Blade)
- [x] Folk hero divergence ([The Hero They Needed])
- [x] Chronicle entries from class system perspective (eerie first-person voice)

## Phase 8: BFS Integration [DONE]
- [x] 15 class system features in aggregate token
- [x] BFS tension includes shame/crisis class counts
- [x] Consolidation as BFS decision point (AcceptConsolidation/RefuseConsolidation)
- [x] Class diversity bonus in value estimation
- [x] State novelty hash includes class counts

## Phase 9: Non-Combat Stat Wiring [DONE]
- [x] `effective_noncombat_stats()` helper computing bonuses from all classes
- [x] diplomacy → faction relation gains (diplomacy.rs)
- [x] commerce → trade income bonus (economy.rs)
- [x] crafting → item quality/speed (crafting.rs)
- [x] medicine → disease recovery rate (disease.rs)
- [x] scholarship → archive point gains (archives.rs)
- [x] stealth → espionage success rates (espionage.rs)
- [x] leadership → recruitment quality (recruitment.rs)
- [x] Party-level quest duration reduction (quest_lifecycle.rs)

## Phase 10: Tiered Ability Framework [DONE]
- [x] Expand SKILL_THRESHOLDS to 25 grants across 100 levels
- [x] SkillEffect enum — non-combat powers that bend existing systems (~45 variants)
- [x] SkillCondition enum — situational constraints, not resource costs (7 conditions)
- [x] Template pools per class per tier in skill_templates.rs (~120 templates across 11 families)
- [x] apply_skill_effect() dispatcher in skill_effects.rs (all variants implemented)
- [x] Wire into UseClassSkill action in step.rs
- [x] BFS action metadata for UseClassSkill
- [ ] Combat ability scaling via generate_tiered_ability()
- [ ] BFS valid_actions filtering by skill conditions

## Design Principles
- No hardcoded primary class ceiling — power emerges naturally from stronger skills at higher levels
- No suppression relationships between classes — warrior-scholar should be unusual but possible
- Rarity emerges from math (quadratic costs, behavior intersections), not artificial gates
- The system watches and responds — it's a reactive entity, not a menu
- Skills constrained by SITUATION and CONSEQUENCE, not resource costs
- Non-combat abilities should feel like powers, not percentage buffs
