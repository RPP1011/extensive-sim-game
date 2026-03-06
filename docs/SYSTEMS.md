# Game Systems Reference

Complete reference for every system in the codebase. Organized by layer, from high-level campaign down to low-level simulation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Campaign Layer](#campaign-layer)
3. [Mission Layer](#mission-layer)
4. [Combat Simulation](#combat-simulation)
5. [AI Layer](#ai-layer)
6. [UI Systems](#ui-systems)
7. [Audio System](#audio-system)
8. [Camera System](#camera-system)
9. [Map Generation](#map-generation)
10. [Scenario / Testing](#scenario--testing)
11. [Data Pipeline](#data-pipeline)

---

## Architecture Overview

A tactical RPG with real-time combat, built on **Bevy 0.13** ECS. The game has three nested layers:

```
Campaign (turn-based overworld)
  -> Mission (multi-room dungeon run)
    -> Combat (100ms fixed-tick simulation)
```

### Key Modules

| Module | Role |
|--------|------|
| `src/main.rs` | Binary entry point, hub UI, top-level systems |
| `src/game_core/` | Campaign/mission data types, 30+ Bevy systems |
| `src/ai/core/` | Core combat simulation engine |
| `src/ai/effects/` | Data-driven ability/effect engine |
| `src/ai/squad/` | Phase 3 personality-driven squad AI |
| `src/scenario/` | Headless scenario runner + generation engine |
| `src/lib.rs` | Module re-exports |

### Runtime Modes

Controlled by `RuntimeModeState`:
- **Hub** -- Overworld navigation, roster management, diplomacy
- **Mission** -- Active combat in procedurally generated rooms

---

## Campaign Layer

**Files:** `src/game_core/`, `src/events/`

The campaign is a turn-based strategic layer where factions compete for territory, the player manages a roster of heroes, and missions emerge from world tension.

### Overworld Map

**Resource:** `OverworldMap`

A network of regions with faction ownership, unrest levels, and strategic value.

| Struct | Purpose |
|--------|---------|
| `OverworldRegion` | Territory with `owner_faction_id`, `unrest`, `control`, `intel` |
| `FactionState` | Diplomatic relationships, military strength |
| `FlashpointState` | Escalating conflict zones that spawn missions |
| `CommanderState` | AI-controlled faction leaders with strategic goals |
| `DiplomacyState` | Faction relation matrix (100x100) |

**Systems:**

| System | Schedule | Purpose |
|--------|----------|---------|
| `turn_management_system` | Update | Global turn counter increment |
| `overworld_cooldown_system` | Update | Tick down region cooldowns |
| `update_faction_war_goals_system` | Update | Commander strategic planning |
| `generate_commander_intents_system` | Update | AI faction order generation |
| `overworld_ai_border_pressure_system` | Update | Faction expansion pressure |
| `overworld_faction_autonomy_system` | Update | AI faction movement |
| `overworld_intel_update_system` | Update | Intel gathering on regions |
| `flashpoint_progression_system` | Update | Escalation/resolution of conflicts |
| `pressure_spawn_missions_system` | Update | Flashpoint -> mission generation |

### Hero Roster & Companions

**Resource:** `CampaignRoster`, `CampaignParties`

Heroes are recruited, assigned to parties, deployed on missions, and develop personal stories.

| Struct | Purpose |
|--------|---------|
| `CampaignRoster` | All available heroes and equipment |
| `CampaignParties` | Active deployment groups |
| `CompanionStoryState` | Per-hero personal quest tracking |
| `CampaignEventLog` | Turn-by-turn consequence journal |

**Systems:**

| System | Purpose |
|--------|---------|
| `sync_roster_lore_with_overworld_system` | Sync hero traits to world state |
| `sync_campaign_parties_with_roster_system` | Party membership updates |
| `campaign_party_orders_system` | Player party command input |
| `companion_mission_impact_system` | Hero performance -> relationship changes |
| `companion_state_drift_system` | Personality evolution over time |
| `generate_companion_story_quests_system` | Hero-specific quest generation |
| `progress_companion_story_quests_system` | Quest state machine updates |
| `companion_recovery_system` | Hero healing after missions |

### Player Interaction

| System | Purpose |
|--------|---------|
| `overworld_hub_input_system` | Player navigation input |
| `flashpoint_intent_input_system` | Flashpoint command input |
| `focus_input_system` | Region focus control |
| `attention_management_system` | Player focus/unfocus tracking |
| `focused_attention_intervention_system` | Detailed interventions when focused |
| `refresh_interaction_offers_system` | Diplomacy offer generation |
| `interaction_offer_input_system` | Player diplomacy responses |

### Mission Lifecycle

| System | Purpose |
|--------|---------|
| `sync_mission_assignments_system` | Campaign -> mission setup |
| `activate_mission_system` | Mission entry trigger |
| `simulate_unfocused_missions_system` | Fast background mission sims |
| `overworld_sync_from_missions_system` | Mission outcomes -> world changes |
| `resolve_mission_consequences_system` | Outcome -> campaign effects |

### Campaign Events

**File:** `src/events.rs`

**Resource:** `CampaignEventQueue`

Random events generated each turn via seeded LCG.

| Event Kind | Description |
|------------|-------------|
| `MerchantOffer` | Trade supplies for items |
| `DeserterIntel` | Faction intel at a cost |
| `PlagueScare` | Region stress penalty |
| `RivalPartySpotted` | Risk/reward encounter |
| `AllyRequest` | Faction asks for help |
| `FactionRumour` | World flavor text |
| `AbandonedCache` | Free supplies |
| `StormWarning` | Movement slowdown |

**Systems:**
- `campaign_event_generation_system` -- generates events per turn
- `draw_event_notification_system` -- egui UI with accept/decline

---

## Mission Layer

**Files:** `src/mission/`

Missions are multi-room dungeon runs with procedurally generated layouts and objectives.

### Mission Execution

**File:** `src/mission/execution.rs`

| Struct | Purpose |
|--------|---------|
| `ActiveMissionContext` | Parameters: room_type, hero/enemy count, seed, difficulty, global_turn |
| `MissionRoomSequence` | Room ordering and progression |

**Systems:**

| System | Purpose |
|--------|---------|
| `mission_enter_system` | Scene initialization |
| `mission_exit_system` | Scene teardown |
| `spawn_room_door_system` | Victory trigger spawning |
| `advance_room_system` | Room progression (next room gen) |
| `draw_mission_ui_system` | Combat HUD rendering |
| `room_progression_system` | Room transition logic |

### Room Generation

**File:** `src/mission/room_gen.rs`

Deterministic procedural room creation from a seed + room type.

| Struct | Purpose |
|--------|---------|
| `NavGrid` | Walkability/elevation grid (`cell_size`, `cols`, `rows`, `walkable[]`, `elevation[]`) |
| `RoomLayout` | Complete geometry: floors, walls, obstacles, spawn zones |
| `RoomFloor` / `RoomWall` / `RoomObstacle` | Renderable room elements |
| `SpawnZone` | Player/enemy spawn positions |

**Functions:**
- `generate_room(seed, room_type) -> RoomLayout` -- deterministic generation
- `spawn_room(layout, commands, meshes, materials)` -- Bevy entity spawning

### Room Types

```rust
RoomType::Entry     // Tutorial/easy introductory room
RoomType::Pressure  // Time-sensitive, intense combat
RoomType::Pivot     // Tactical decision point
RoomType::Setpiece  // Scripted dramatic encounter
RoomType::Recovery  // Rest and resupply
RoomType::Climax    // Final boss encounter
```

### Room Sequencing

**File:** `src/mission/room_sequence.rs`

Multi-room progression logic. Rooms are sequenced based on mission parameters and difficulty curve.

### Mission Objectives

**File:** `src/mission/objectives.rs`

| Objective Kind | Description |
|----------------|-------------|
| `Eliminate` | Kill all enemies |
| `Hold { zone, ticks_required }` | Hold a zone for N ticks |
| `Extract { npc_unit_id, exit_zone }` | Escort NPC to extraction |
| `Sabotage { nodes[] }` | Activate all sabotage points |

**Systems:**
- `check_objective_system` -- per-tick objective evaluation
- `draw_objective_hud_system` -- objective progress UI

**Functions:**
- `generate_objective(room_type, layout, seed)` -- deterministic from seed
- `reset_objective(state, new_obj)` -- transition between rooms

### Simulation Bridge

**File:** `src/mission/sim_bridge.rs`

Bridges Bevy ECS with the pure-Rust combat simulation.

| Resource | Purpose |
|----------|---------|
| `MissionSimState` | Live combat state (wraps `SimState`) |
| `EnemyAiState` | Wraps `SquadAiState` for AI team |
| `PlayerOrderState` | Player unit selection and move targets |
| `SimEventBuffer` | Accumulated sim events for VFX/audio |
| `PlayerUnitMarker` | Component marking player-controlled units |

**Systems:**
- `mission_sim_tick_system` -- fixed-step sim advancement
- `update_unit_health_data_system` -- HP sync to visuals
- `update_unit_position_data_system` -- position sync
- `mission_input_system` -- player order input handling

**Functions:**
- `build_sim_with_templates(hero_count, enemy_wave, seed)` -- standard init
- `build_sim_with_hero_templates(hero_tomls, enemy_wave, seed)` -- custom heroes
- `scale_enemy_stats(unit, global_turn)` -- difficulty scaling (1.0x-3.0x)
- `threat_level(global_turn) -> 1..5` -- threat tier from turn number

### Unit Visualization

**File:** `src/mission/unit_vis.rs`

| Component/Resource | Purpose |
|--------------------|---------|
| `UnitVisual` | Marks unit 3D representation |
| `SelectionRing` | Selection indicator child entity |
| `HpBarFg` / `HpBarBg` | Health bar child entities |
| `UnitSelection` | Currently selected unit IDs |
| `UnitHealthData` | HP cache for rendering |
| `UnitPositionData` | Position cache for rendering |

### Visual Effects

**File:** `src/mission/vfx.rs`

**Resource:** `VfxEventQueue`

| VFX Event | Description |
|-----------|-------------|
| `Damage` | Floating damage number |
| `Heal` | Floating heal number |
| `Death` | Death animation |
| `Control` | CC applied indicator |
| `ChannelStart/End` | Casting indicator ring |
| `ChainFlash` | Chain lightning visual |
| `Trail` | Projectile trail |
| `ShieldFlash` | Shield absorb flash |
| `Miss` / `Resist` | Floating "MISS"/"RESIST" text |
| `ZonePulse` | AoE zone pulse effect |

**Components:** `FloatingText`, `HitFlash`, `DeathFade`, `ProjectileVisual`, `ZoneVisual`, `TetherVisual`, `ChannelRing`, `StatusIndicator`

### Enemy Templates

**File:** `src/mission/enemy_templates.rs`

| Template | HP | Role |
|----------|-----|------|
| `Grunt` | 80 | Melee DPS |
| `Brute` | 160 | Tanky melee |
| `Mystic` | 60 | Ranged control |
| `Sentinel` | 75 | Stationary ranged |
| `Berserker` | 90 | Aggressive when <50% HP |
| `Summoner` | 70 | Spawns grunt allies |

**Functions:**
- `build_enemy_unit(template, id, position)` -- template to unit
- `default_enemy_wave(count, seed, spawn_positions)` -- wave generation
- `generate_boss()` -- special boss unit

### Hero Templates

**File:** `src/mission/hero_templates.rs`

Built-in archetypes: `Warrior`, `Ranger`, `Mage`, `Cleric`, `Rogue`, `Paladin`

Data-driven templates loaded from TOML in `assets/hero_templates/`:
alchemist, assassin, bard, berserker, blood_mage, cryomancer, druid, elementalist, engineer, knight, monk, necromancer, pyromancer, samurai, shadow_dancer, shaman, templar, warden, warlock, witch_doctor

**Functions:**
- `parse_hero_toml(toml_str) -> HeroToml`
- `hero_toml_to_unit(toml, id, team, position) -> UnitState`
- `load_embedded_templates() -> HashMap<String, HeroToml>`

### Tag Color Mapping

**File:** `src/mission/tag_color.rs`

Maps ability/damage tags (e.g. "fire", "ice") to display colors.

---

## Combat Simulation

**File:** `src/ai/core.rs`

Pure-Rust, deterministic, fixed-tick combat simulation. No Bevy dependency -- runs identically in tests and in-game.

### Core Types

| Type | Purpose |
|------|---------|
| `SimState` | Global state: `units[]`, `projectiles[]`, `zones[]`, `tethers[]`, `tick`, `rng_state` |
| `UnitState` | Full unit: stats, position, cooldowns, abilities, status effects |
| `Team` | `Hero` or `Enemy` |
| `UnitIntent` | Per-unit action order for a tick |
| `SimEvent` | Output event from simulation step |

### Tick Loop

```
step(sim, intents, dt_ms) -> (new_sim, events[])
```

Constant: `FIXED_TICK_MS = 100` (10 ticks/second)

Each tick:
1. Apply movement from intents
2. Execute ability casts
3. Resolve projectile hits
4. Tick zone effects
5. Tick tethers
6. Tick status effects (DoTs, HoTs, CC durations)
7. Check unit deaths
8. Emit events

### Intent Actions

```rust
IntentAction::
  Attack { target_id }
  CastAbility { target_id }
  CastHeal { target_id }
  CastControl { target_id }
  UseAbility { ability_index, target }
  MoveTo { position }
  Hold
```

### Simulation Events (30+ types)

Movement: `Moved`
Combat: `DamageApplied`, `HealApplied`, `ShieldApplied`, `ShieldAbsorbed`, `AttackMissed`
Abilities: `CastStarted`, `CastFailedOutOfRange`, `AbilityUsed`, `PassiveTriggered`
CC: `ControlApplied`
Lifecycle: `UnitDied`, `ChannelStarted`, `ChannelTicked`
Special: `LifestealHeal`, `ReflectDamage`

### Utility Functions

- `distance(a, b) -> f32` -- 2D distance
- `move_towards(from, to, distance) -> SimVec2` -- clamped movement
- `move_away(from, away, distance) -> SimVec2` -- repulsion
- `position_at_range(target, attacker, range) -> SimVec2` -- position at range
- `run_replay(actions, initial_sim) -> ReplayResult` -- deterministic replay

---

## AI Layer

**Files:** `src/ai/`

Multi-phase AI architecture. Each phase adds sophistication.

### Phase Trait

**File:** `src/ai/phase.rs`

```rust
pub trait AiPhase {
    fn generate_intents(&mut self, state: &SimState, dt_ms: u32) -> Vec<UnitIntent>;
    fn update_from_events(&mut self, events: &[SimEvent]) {} // optional
}
```

### Phase 1: Utility AI

**File:** `src/ai/utility.rs`

Score-based action selection.

| Config Field | Purpose |
|--------------|---------|
| `distance_weight` | Prefer nearby targets |
| `damage_weight` | Prefer high-damage actions |
| `overkill_penalty_weight` | Penalize wasted damage on low-HP targets |
| `stickiness_bonus` | Prefer current target |
| `target_lock_ticks` | How long to stick to a target |

### Phase 3: Squad AI

**File:** `src/ai/squad.rs`

Personality-driven decision-making with force-based steering.

**State:** `SquadAiState`

**Personality System (7 traits, 0.0-1.0):**

| Trait | Effect |
|-------|--------|
| `aggression` | Willingness to attack |
| `compassion` | Healing priority |
| `caution` | Risk aversion |
| `discipline` | Coordination adherence |
| `cunning` | Tactical maneuvering |
| `tenacity` | Commitment to targets |
| `patience` | Cooldown acceptance |

**Dominant Forces (9 tactical impulses):**
Attack, Heal, Retreat, Control, Focus, Protect, Pursue, Regroup, Position

**Formation Modes:** Hold (maintain, defensive), Advance (push forward, offensive), Retreat (fall back)

**Key Functions:**
- `infer_personality(unit) -> Personality` -- auto-infer from stat block
- `generate_intents(sim, squad_state, dt_ms) -> Vec<UnitIntent>` -- per-tick

### Phase 4: Control AI

**File:** `src/ai/control.rs`

Hard CC coordination layer built on Phase 3.

- CC reservation system (avoid overlapping stuns)
- Diminishing returns windows (0.5x multiplier after repeat CC)
- CC profile: duration, diminishing window, multiplier

### Roles

**File:** `src/ai/roles.rs`

| Role | Range | Behavior |
|------|-------|----------|
| `Tank` | Close | Absorbs damage, holds position |
| `Dps` | Mid | Priority target elimination |
| `Healer` | Far | Keeps allies alive |

**RoleProfile:** preferred range, leash distance, threat sensitivity, focus bonus

Threat tables per unit with sticky target lock (4-tick lock after switch).

### Personality

**File:** `src/ai/personality.rs`

6-trait personality profile with preset archetypes:

| Preset | Style |
|--------|-------|
| `Vanguard` | Aggressive tank |
| `Guardian` | Protective support |
| `Tactician` | Strategic controller |

Personalities drift over time based on mission outcomes.

### Pathfinding

**File:** `src/ai/pathing.rs`

A* grid navigation with elevation and slope costs.

| Type | Purpose |
|------|---------|
| `GridNav` | Navigation grid with blocked cells, elevation, slope costs |

**Functions:**
- `find_path(from, to, nav) -> Vec<SimVec2>` -- A* pathfinding
- `clamp_step_to_walkable(pos, desired, nav) -> SimVec2` -- collision response
- `has_line_of_sight(nav, from, to) -> bool` -- raycasting

### Phase 9: Advanced Coordination

**File:** `src/ai/advanced.rs`

Encounter-level pressure system with team-wide burst coordination.

- Arc Net skill (3.0 radius, 16 tick duration, 95 tick cooldown)
- Steam Vent skill (3.5 radius, 11 tick duration, 140 tick cooldown)
- Spatial, tactical, and coordination sampling

### Student Model (Distilled AI)

**File:** `src/ai/student.rs`

Tiny 2-layer MLP (~17K parameters) that approximates the full AI pipeline.

```
Input (60 features) -> 128 hidden -> 64 hidden -> 9 outputs
```

**Inputs (60 features):**
- [0..40] Aggregate: HP, cooldowns, roles, personality, formation, game phase
- [40..60] Spatial: distances, engagement, clustering, threats

**Outputs:**
- 6 personality weights (sigmoid -> [0,1])
- 3 formation logits (softmax)

Uses AVX2+FMA SIMD optimization. No ML framework dependency.

### Ability Embedding Autoencoder

**File:** `src/ai/core/ability_encoding.rs`

Encodes ability definitions into 32-dim semantic embeddings for the self-play policy. Trained via supervised contrastive loss (9 ability categories) + reconstruction MSE on 856 abilities from all hero templates.

```
Encoder:  80 (properties) -> 64 (ReLU) -> 32 (L2-normalized)
Decoder:  32 -> 64 (ReLU) -> 80 (reconstructed)
```

The 80-dim input extracts targeting mode, delivery method, damage/heal/CC values, buff/debuff counts, mobility, and area properties from `AbilityDef` TOML. Expands self-play feature dimension from 311 to 919 (8 ability slots × 34 features each).

**Training pipeline:** `xtask scenario oracle ability-encoder-export` → `scripts/train_ability_encoder.py` → `generated/ability_encoder.json`

See `docs/STUDENT_MODEL_APPROACH.md` § "Ability Embedding Autoencoder" for full details.

### Ability Evaluation System

**Files:** `src/ai/core/ability_eval/`

Interrupt-driven ability priority using per-category neural network evaluators. 9 categories (DamageUnit, DamageAoe, CcUnit, HealUnit, HealAoe, Defense, Utility, Summon, Obstacle) each with a tiny MLP (~300-1.2K params). Urgency threshold = 0.4.

See [docs/ABILITY_PRIORITY_SYSTEM.md](ABILITY_PRIORITY_SYSTEM.md) for full details.

**Key Functions:**
- `evaluate_abilities(state, squad_ai, unit_id, weights) -> Option<(IntentAction, f32)>`
- `evaluate_abilities_with_encoder()` — enhanced version with ability embeddings
- Post-prediction modifiers: heal saturation, cleanup boost (tick > 2000), cleanup suppress (tick > 5000)

### AI Tooling

**File:** `src/ai/tooling.rs`

Utility functions for AI development and debugging.

---

## Effects Engine

**Files:** `src/ai/effects/effect_enum.rs`, `src/ai/effects/types.rs`, `src/ai/effects/defs.rs`

Data-driven ability system. Abilities are defined in TOML and resolved at runtime.

See [docs/ABILITY_SYSTEM.md](ABILITY_SYSTEM.md) for the complete reference with all fields and defaults.

### Ability Definition

```rust
AbilityDef {
    name, targeting, range, cooldown_ms, cast_time_ms, ai_hint,
    effects: Vec<ConditionalEffect>,
    delivery: Option<Delivery>,
    resource_cost, morph_into, morph_duration_ms, zone_tag,
    // LoL Coverage:
    max_charges, charge_recharge_ms,        // ammo system
    is_toggle, toggle_cost_per_sec,         // toggle abilities
    recast_count, recast_window_ms, recast_effects,  // multi-cast
    unstoppable,                            // CC immunity during cast
    swap_form, form,                        // stance/form swap
    evolve_into,                            // permanent ability upgrade
}
```

### Effect Types (45)

**Core Combat:** `Damage` (with `DamageType`: physical/magic/true), `Heal`, `Shield`, `SelfDamage`, `Execute`, `Lifesteal`, `Reflect`, `DamageModify`

**Crowd Control (13):** `Stun`, `Root`, `Silence`, `Slow`, `Fear`, `Taunt`, `Blind`, `Polymorph`, `Banish`, `Confuse`, `Charm`, `Suppress`, `Grounded`

**Positioning:** `Dash` (with `is_blink`), `Knockback`, `Pull`, `Swap`

**Buffs/Debuffs:** `Buff`, `Debuff`, `OnHitBuff`

**Summoning:** `Summon` (with `clone`, `directed` flags), `CommandSummons`

**Healing/Shield:** `Resurrect`, `OverhealShield`, `AbsorbToHeal`, `ShieldSteal`, `StatusClone`

**Status Interaction:** `Immunity`, `DeathMark`, `Detonate`, `StatusTransfer`, `Dispel`

**Complex:** `Duel`, `Stealth`, `Leash`, `Link`, `Redirect`, `Rewind`, `CooldownModify`, `ApplyStacks`, `Obstacle`, `ProjectileBlock`, `Attach`, `EvolveAbility`

### Delivery Methods (7)

`Instant`, `Projectile`, `Channel`, `Zone`, `Tether`, `Trap`, `Chain`

### Targeting Modes (8)

`TargetEnemy`, `TargetAlly`, `SelfCast`, `SelfAoe`, `GroundTarget`, `Direction`, `Vector`, `Global`

### Damage Types (3)

`Physical` (reduced by armor, default), `Magic` (reduced by magic_resist), `True` (ignores all reduction)

### Stacking Modes

`Refresh` (reset duration, default), `Extend` (add duration), `Strongest` (keep highest), `Stack` (allow multiple)

### TOML Format

Hero templates in `assets/hero_templates/*.toml` define:
- Base stats (HP, move speed, armor, magic_resist, resource)
- Active abilities with full effect/delivery/targeting definitions
- Passive abilities with triggers
- Tags for damage/resistance interactions

---

## UI Systems

**Files:** `src/ui/`

All UI uses `bevy_egui` for immediate-mode rendering.

### Save/Load System

**File:** `src/ui/save_browser.rs`

| Feature | Detail |
|---------|--------|
| Save slots | Slot 1 (F5), Slot 2 (Shift+F5), Slot 3 (Ctrl+F5) |
| Autosave | Every 10 turns |
| Save versions | V1, V2, V3 (current) |
| Index file | Tracks slot metadata |

**Systems:**
- `campaign_save_load_input_system` -- keyboard shortcuts
- `campaign_autosave_system` -- periodic autosave
- `campaign_save_panel_input_system` -- load UI interaction

### Quest Log

**File:** `src/ui/quest_log.rs`

**Resource:** `QuestLogState` (open flag, selected_hero_id)

- `quest_log_toggle_system` -- J key toggle
- `draw_quest_log_system` -- egui rendering of hero quest list

### Settings Menu

**File:** `src/ui/settings.rs`

Camera sensitivity, zoom, invert Y, screenshot capture.

**Systems:**
- `settings_menu_toggle_system` -- Esc key
- `settings_menu_slider_input_system` -- orbit/zoom sensitivity
- `settings_menu_toggle_input_system` -- invert Y toggle
- `manual_screenshot_capture_system` -- screenshot on spacebar
- `update_settings_menu_visual_system` -- visual sync

### Tutorial

**File:** `src/ui/tutorial.rs`

- `tutorial_toggle_system` -- T key toggle
- `draw_tutorial_system` -- egui rendering

---

## Audio System

**File:** `src/audio.rs`

Event-driven audio with dynamic music switching.

**Resource:** `AudioSettings` (master, music, sfx volumes), `AudioHandles`, `AudioEventQueue`

### Audio Events

| Event | Variants |
|-------|----------|
| `PlaySfx(SfxKind)` | Hit, Death, Ability, UiClick |
| `StartMusic(MusicKind)` | Hub, Combat |
| `StopMusic` | -- |

**Systems:**
- `load_audio_assets_system` -- asset loading
- `process_audio_events_system` -- audio playback
- `combat_music_intensity_system` -- dynamic music switching based on combat state

---

## Camera System

**File:** `src/camera.rs`

Orbit camera with smooth focus transitions.

| Resource | Purpose |
|----------|---------|
| `CameraSettings` | Sensitivity, zoom, invert_y (persisted to JSON) |
| `CameraFocusTransitionState` | Smooth panning interpolation |
| `SceneViewBounds` | Clamp bounds |

**Component:** `OrbitCameraController` (focus, radius, yaw, pitch)

**Systems:**
- `setup_camera` -- initial camera/light creation
- `orbit_camera_controller_system` -- mouse/keyboard input
- `persist_camera_settings_system` -- auto-save to JSON

**Triggers:** `CameraFocusTrigger::TakeCommand`, `CameraFocusTrigger::FocusSelectedParty`

---

## Map Generation

### Voronoi Map Generation

**File:** `src/mapgen_voronoi.rs`

Procedural overworld map generation using Voronoi diagrams.

**Output:** `VoronoiSpec`
- Grid dimensions
- Regions with position, territory percentage, boundary complexity
- Faction assignments
- Roads, settlements, terrain notes

### Gemini API Map Generation

**File:** `src/mapgen_gemini.rs`

AI-assisted content generation via Google Gemini API.

- `call_gemini(model, prompt, api_key) -> Value`
- `call_gemini_text(model, prompt, api_key) -> Value`
- `call_gemini_with_reference_image(...)` -- image-guided generation
- Dotenv support for API key loading

---

## Scenario / Testing

**Files:** `src/scenario/`

Headless scenario runner for balance testing, CI, and training data generation. Includes a Rust-native scenario generation engine.

### Configuration

```rust
ScenarioCfg {
    name, seed, hero_count, enemy_count, difficulty,
    max_ticks, room_type, hero_templates[], hp_multiplier,
}
```

### Result Capture

- Per-unit combat statistics
- Per-ability usage and damage breakdowns
- Kill attribution, overkill tracking

### Assertions

```rust
assertions: {
    outcome: Victory,   // or "Defeat", "Either" (Victory|Defeat), "Any" (including Timeout)
    max_ticks_to_win: 500,
    min_heroes_alive: 2,
    max_heroes_dead: 1,
}
```

### Functions

- `run_scenario(cfg) -> ScenarioResult` -- full sim with stats
- `run_scenario_to_state(cfg) -> (SimState, SquadAiState)` -- initialized state
- `run_scenario_to_state_with_room(cfg) -> (SimState, SquadAiState, GridNav)` -- with room
- `check_assertions(result, asserts) -> Vec<AssertionResult>`

### Scenario Files

Hand-crafted scenarios in `scenarios/*.toml`:
basic_4v4, outnumbered_2v6, climax_boss, pressure_room, elite_squad, skirmish_3v3, swarm_3v8, duel_1v1, full_roster_8v8, glass_cannon_2v4, steamroll_6v3, attrition_4v4_hard, boss_rush_6v2, escort_5v7, zerg_rush_4v10, climax_solo_1v2, easy_warmup_4v3, healer_heavy_4v6, even_6v6, knife_edge_4v5_hard

Generated scenarios in `scenarios/generated/gen_*.toml` (~3,300+ files).

### Scenario Generation Engine

**File:** `src/scenario/gen.rs`
**CLI:** `cargo xtask scenario generate [OPTIONS]`

Coverage-driven, constraint-based generation that exhaustively explores the space of hero compositions, room types, and difficulty levels. Simulation throughput (~50-250K scenarios/min parallel) means we generate everything meaningful rather than artificially capping count.

**6 generation strategies:**

| Strategy | Base count | What it tests |
|----------|------------|---------------|
| Synergy pairs | ~702 | Every unique hero pair in 2v3 and 3v4 scenarios |
| Stress archetypes | ~78 | Extreme compositions (all-tank, no-healer, glass cannon, etc.) x 6 room types |
| Difficulty ladders | ~80 | Same comp at difficulty 1-5 x hp multipliers {1x, 3x} |
| Room-aware | ~24 | Fit & mismatch compositions per room geometry |
| Size spectrum | ~38 | Team sizes from 1v2 through 8v10 |
| Random fill | ~200 | Coverage-driven random, anchored on least-seen heroes/rooms |

With 3 seed variants per base scenario, default output is ~3,300 scenarios.

**Key properties:**

- **96% hero balance** — every hero appears within 4% of the mean
- **100% pair coverage** — all 351 unique hero pairs tested
- **Flat room distribution** — ~equal scenarios per room type
- **Hash-based dedup** — no redundant compositions (sorted heroes + enemy count + difficulty + room + hp)
- **~4s to run all 3,300** in release mode (sequential)

**CLI options:**

```
--seed N              RNG seed (default 2026, deterministic)
--seed-variants N     Seed variants per base scenario (default 3)
--extra-random N      Extra coverage-driven random scenarios (default 200)
--no-synergy          Skip synergy pair generation
--no-stress           Skip stress archetype generation
--no-ladders          Skip difficulty ladder generation
--no-room-aware       Skip room-aware composition generation
--no-sizes            Skip team size spectrum generation
--output DIR          Output directory (default scenarios/generated)
-v, --verbose         Print coverage report with hero/room/size breakdowns
```

**Room-type affinity** — the engine knows which roles suit which room geometries:

| Room | Preferred roles |
|------|-----------------|
| Entry | Tank, Healer, MeleeDps, RangedDps (balanced) |
| Pressure | RangedDps, Tank, Healer (chokepoints) |
| Pivot | MeleeDps, RangedDps, Hybrid (burst/mobility) |
| Setpiece | RangedDps, Hybrid, Tank (large spaces) |
| Recovery | Healer, Hybrid, Tank (sustain) |
| Climax | Tank, Healer, MeleeDps (boss fights) |

**Hero role catalog** (27 heroes across 5 roles):

| Role | Heroes |
|------|--------|
| Tank | warrior, knight, paladin, warden, templar |
| Healer | cleric, druid, bard, shaman, alchemist |
| MeleeDps | rogue, assassin, berserker, samurai, shadow_dancer, monk |
| RangedDps | mage, ranger, pyromancer, cryomancer, elementalist, engineer, arcanist |
| Hybrid | blood_mage, necromancer, warlock, witch_doctor |

---

## Data Pipeline

### Flow: Campaign -> Mission -> Combat

```
Campaign Turn
  |
  v
FlashpointState escalates -> pressure_spawn_missions_system
  |
  v
ActiveMissionContext created -> mission_enter_system
  |
  v
RoomLayout generated (room_gen) -> NavGrid built
  |
  v
SimState initialized (sim_bridge) -> units spawned
  |
  v
Per-tick loop:
  1. Player input -> PlayerOrderState -> hero_intents
  2. Enemy AI (SquadAiState) -> enemy_intents
  3. step(sim, all_intents, dt_ms) -> (new_sim, events)
  4. Events -> VFX, audio, UI updates
  5. check_objective_system -> completion?
  |
  v
MissionResult -> resolve_mission_consequences_system
  |
  v
OverworldMap updated, heroes healed, faction state changed
```

### Flow: TOML -> Unit

```
assets/hero_templates/knight.toml
  |
  v
parse_hero_toml() -> HeroToml { stats, abilities[], passives[] }
  |
  v
hero_toml_to_unit() -> UnitState { hp, position, ability_slots[], ... }
  |
  v
SimState.units.push(unit)
```

### Difficulty Scaling

Enemy stats scale with `global_turn`:
- `scale_enemy_stats(unit, global_turn)` applies 1.0x to 3.0x multiplier
- `threat_level(global_turn)` returns tier I-V (displayed as roman numerals)

---

## File Map

```
src/
+-- lib.rs                          Module re-exports
+-- main.rs                         Binary: hub UI, top-level systems
+-- game_core.rs                    Campaign data, mission data, 30+ systems
+-- events.rs                       Campaign random events
+-- scenario/                       Headless scenario runner & generation engine
|   +-- types.rs                   ScenarioCfg, ScenarioResult, UnitStats
|   +-- runner.rs                  State builders, hero template resolution
|   +-- simulation.rs              Core runner, assertion checking, file loading
|   +-- gen.rs                     Coverage-driven scenario generation engine
+-- camera.rs                       Orbit camera
+-- audio.rs                        Audio event queue
+-- mapgen_voronoi.rs               Voronoi overworld gen
+-- mapgen_gemini.rs                Gemini API integration
|
+-- mission/
|   +-- mod.rs                      Module exports
|   +-- execution.rs                Mission scene lifecycle
|   +-- objectives.rs               Room objectives (eliminate, hold, etc.)
|   +-- sim_bridge.rs               Bevy <-> SimState bridge
|   +-- room_gen.rs                 Procedural room generation
|   +-- room_sequence.rs            Multi-room progression
|   +-- unit_vis.rs                 Unit 3D visualization
|   +-- vfx.rs                      Visual effects
|   +-- hero_templates.rs           Hero TOML loading
|   +-- enemy_templates.rs          Enemy archetypes
|   +-- tag_color.rs                Tag -> color mapping
|
+-- ai/
|   +-- mod.rs                      Module exports
|   +-- core.rs                     Combat simulation engine
|   +-- effects.rs                  Data-driven ability engine
|   +-- phase.rs                    AiPhase trait
|   +-- utility.rs                  Phase 1: utility scoring
|   +-- squad.rs                    Phase 3: personality-driven AI
|   +-- control.rs                  Phase 4: CC coordination
|   +-- advanced.rs                 Phase 9: team coordination
|   +-- roles.rs                    Tank/DPS/Healer roles
|   +-- personality.rs              Personality traits & presets
|   +-- pathing.rs                  A* pathfinding
|   +-- student.rs                  Distilled MLP model
|   +-- tooling.rs                  AI dev utilities
|
+-- ui/
|   +-- mod.rs                      Module exports
|   +-- save_browser.rs             Save/load system
|   +-- quest_log.rs                Quest log UI
|   +-- settings.rs                 Settings menu
|   +-- tutorial.rs                 Tutorial overlay
|
+-- bin/
    +-- room_preview.rs             Room visualization tool
    +-- gen_scenarios.rs            (deprecated) Legacy scenario gen
    +-- sim_bridge.rs               Standalone sim runner
    +-- xtask/                      Build/dev tasks (scenario run/bench/generate/oracle)
```
