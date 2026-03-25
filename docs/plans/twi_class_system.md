# TWI-Inspired Class System — Implementation Plan

Inspired by The Wandering Inn: classes are EARNED from behavior, not chosen.
The system watches what you do and responds.

## Phase 1: Core Data Model [TODO]
- [ ] `BehaviorLedger` struct on Adventurer — per-action-type counters with rolling window
- [ ] `ClassInstance` struct — class_name, level, xp, xp_to_next, stagnation_ticks, skills_granted
- [ ] `Vec<ClassInstance>` on Adventurer (multi-class)
- [ ] `ClassTemplate` for matching behavior patterns to class grants
- [ ] Quadratic XP curve: level N costs N^2 * 100
- [ ] Wire into state.rs with #[serde(default)]

## Phase 2: Behavior Tracking [TODO]
- [ ] `tick_behavior_ledger()` system — increment counters from WorldEvents each tick
- [ ] Action categories: melee_combat, ranged_combat, healing, diplomacy, trading, crafting, exploring, leading, sneaking, researching, defending, supporting
- [ ] Rolling window (200 ticks) weighted 3x over lifetime
- [ ] Behavioral fingerprint: normalized 12-dim vector from ledger

## Phase 3: Class Acquisition [TODO]
- [ ] `tick_class_acquisition()` system — check behavior ledger against templates
- [ ] Base class templates: Warrior, Ranger, Mage, Cleric, Rogue, Merchant, Scholar, Diplomat, Artisan, Scout, Leader, Healer
- [ ] Rare class templates from behavior intersections (Spellblade, Plague Doctor, etc.)
- [ ] Crisis-forged classes (Warlord during war, Survivor from near-death)
- [ ] Negative-space classes from deliberate avoidance (Peacemaker, Hermit)
- [ ] WorldEvent::ClassGranted { adventurer_id, class_name, level: 1 }

## Phase 4: Multi-Class Leveling [TODO]
- [ ] Independent XP pools filled by tagged actions
- [ ] Quadratic cost curve per class
- [ ] Stagnation timer: 500 ticks no activity → halved XP gain, 1000 → frozen
- [ ] Class resonance: thematic overlap trickles 15% XP between related classes
- [ ] Activity recency weighting for stat bonuses (rolling 200-tick average)

## Phase 5: Skill Granting [TODO]
- [ ] At level thresholds (3, 5, 7, 10, 15, 20, 25), evaluate skill candidates
- [ ] Behavior vector scoring: score each candidate against adventurer's axes
- [ ] Rarity tiers: Common (orthodox play), Uncommon (edge cases), Rare (campaign system hooks), Capstone (max level synthesis)
- [ ] Campaign system hooks: surviving bankruptcy → [Debt Remembrance], civil war → [Oathbreaker's Resolve]
- [ ] Cross-class skill contamination: old class behavior bleeds at 25% weight
- [ ] Skill suppression: getting [Precise Strike] suppresses [Wild Swing]
- [ ] WorldEvent::SkillGranted { adventurer_id, skill_name, rarity, class_name }

## Phase 6: Class Consolidation & Evolution [TODO]
- [ ] Consolidation offers when two classes both reach consolidates_at level
- [ ] LLM-generated merge names with template fallback
- [ ] Refusal banking: refused offers upgrade rarity if offered again later
- [ ] Evolution from cumulative achievements (not just level)
- [ ] World-event-gated unique classes
- [ ] Unique holder tracking: only one adventurer can hold a Unique class

## Phase 7: Reactive Narrative [TODO]
- [ ] Shame classes: [Coward], [Oathbreaker], [Kinslayer] from behavioral thresholds
- [ ] Single-moment crisis grants from tension accumulator spikes
- [ ] Class identity erosion: coherence score decays when actions contradict class
- [ ] Mirror offers: forced choice between contradictory class paths
- [ ] Witnessed vs unwitnessed acts → public vs hidden classes
- [ ] Chronicle entries from the class system's perspective

## Phase 8: BFS Integration [TODO]
- [ ] Behavior ledger in tokens (behavioral fingerprint as aggregate features)
- [ ] Class state in tokens (class levels, stagnation, skill counts)
- [ ] Class-related actions in BFS action space
- [ ] Ensure diverse class trajectories across BFS runs
- [ ] Skill diversity: novel abilities generated per run

## Design Principles
- No hardcoded primary class ceiling — power emerges naturally from stronger skills at higher levels
- No suppression relationships between classes — warrior-scholar should be unusual but possible
- Rarity emerges from math (quadratic costs, behavior intersections), not artificial gates
- The system watches and responds — it's a reactive entity, not a menu
