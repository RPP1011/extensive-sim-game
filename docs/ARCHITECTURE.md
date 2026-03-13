# Architecture

A tactical RPG with real-time combat built on **Bevy 0.13.2** ECS. Three nested layers: turn-based campaign overworld, multi-room mission runs, and deterministic fixed-tick combat simulation.

| Layer | Technology |
|-------|-----------|
| Language | Rust (edition 2021) |
| Game engine | Bevy 0.13.2 |
| UI | bevy_egui 0.27 |
| Serialization | serde / serde_json |
| HTTP | reqwest 0.12 (async + blocking) |
| AI API | Gemini (optional `gemini` feature flag) |
| Vector search | hnsw_rs 0.3.3 |
| CLI tooling | clap 4.5 |

---

## Source Layout

Every `.rs` file is kept under 500 lines. Larger modules are split into directories with `mod.rs` + sub-files, re-exporting via `pub use`.

```
src/
├── main.rs                  Entry point & Bevy app setup
├── lib.rs                   Crate root re-exports
├── app_systems.rs           Bevy system registration helpers
├── cli_args.rs              CLI argument parsing
│
├── game_core/               Campaign & mission state (21 files)
├── ai/                      Combat AI pipeline (40+ files)
├── mission/                 Mission execution & visuals (25+ files)
├── hub_ui_draw/             Hub egui UI rendering (11 files)
├── ui/                      Shared UI (save, settings, quest log, tutorial)
│
├── backstory_cinematic/     AI-generated backstory sequences
├── campaign_ops/            Save/load, campaign init
├── events/                  Random campaign events
├── runtime_assets/          AI-generated art pipeline
├── scenario_3d/             3D scenario replay viewer
├── screenshot_capture/      Automated screenshot system
├── simulation_cli/          Headless sim CLI runners
├── mapgen_voronoi/          Procedural overworld generation
├── scenario/                Headless balance test runner
│
├── audio.rs                 Event-driven audio
├── camera.rs                Orbit camera with focus transitions
├── character_select.rs      Faction & backstory selection logic
├── fade.rs                  Screen fade transitions
├── game_loop.rs             Runtime mode, turn pacing, run conditions
├── hub_outcome.rs           Campaign outcome & region transitions
├── hub_systems.rs           Hub action processing & scene sync
├── hub_types.rs             Hub UI state types
├── local_intro.rs           Region eagle-eye intro sequences
├── mapgen_gemini.rs         Gemini API map generation
├── region_nav.rs            Region target picker & party navigation
├── terrain.rs               Overworld terrain scene setup
├── ui_helpers.rs            Shared egui helper widgets
│
├── tests/                   Integration tests (8 files)
└── bin/                     Binary crates (xtask, sim_bridge, etc.)
```

---

## Top-Level Files

### `src/main.rs`
Binary entry point. Parses CLI args, selects runtime mode (hub/mission/scenario/headless), inserts all Bevy resources, and delegates system registration to `app_systems.rs`.

### `src/lib.rs`
Crate root that re-exports `ai`, `game_core`, `mission`, `scenario`, and `mapgen_voronoi` for use by binary crates and integration tests.

### `src/app_systems.rs`
Extracted Bevy system registration. Contains `register_hub_systems`, `register_default_mission_systems`, `register_scenario_3d_systems`, `register_common_systems`, `register_rendered_input_systems`, `register_startup_systems`, and `register_screenshot_systems`.

### `src/cli_args.rs`
Defines `CliArgs` struct and `parse_cli_args()` function. Handles 30+ CLI flags for headless mode, simulation phases, scenario paths, screenshot modes, and seed overrides.

### `src/audio.rs`
Event-driven audio with `AudioEventQueue` resource. Supports SFX (hit, death, ability, UI click), music tracks (hub, combat), and dynamic combat music intensity scaling.

### `src/camera.rs`
Orbit camera controller with mouse/keyboard input, smooth focus transitions via `CameraFocusTransitionState`, and persisted `CameraSettings` (sensitivity, zoom, invert Y) saved to JSON.

### `src/character_select.rs`
Faction and backstory selection logic for new campaigns. Builds selection choices from overworld data, applies stat modifiers, and bootstraps the initial roster and party.

### `src/fade.rs`
Full-screen fade overlay system with `FadeState` resource. Supports fade-in and fade-out transitions with configurable duration and color.

### `src/game_loop.rs`
Runtime control: `RuntimeModeState` (hub vs mission), `SimulationSteps` (headless frame limit), `TurnPacingState`, `StartSceneState`. Provides Bevy run conditions like `run_if_gameplay_active` and `run_if_hub_runtime_active`.

### `src/hub_outcome.rs`
Campaign outcome detection and UI overlay (victory/defeat). Also hosts `region_layer_transition_system`, `local_intro_sequence_system`, `draw_runtime_asset_gen_egui_system`, and `hub_quit_requested_system`.

### `src/hub_systems.rs`
Hub action processing. `hub_menu_input_system` handles keyboard navigation. `hub_apply_action_system` executes queued actions. `apply_hub_action` resolves guild actions (assemble expedition, review recruits, intel sweep, dispatch relief). `sync_hub_scene_visibility_system` toggles 3D scene elements by screen.

### `src/hub_types.rs`
Hub UI state types: `HubMenuState`, `StartMenuState`, `HubActionQueue`, `HubAction` enum (AssembleExpedition, ReviewRecruits, IntelSweep, DispatchRelief, LeaveGuild), `CharacterCreationUiState`, `CampaignOutcomeState`, `HeroDetailUiState`.

### `src/local_intro.rs`
Region eagle-eye intro sequence. Plays a cinematic camera sweep when entering a new region, with phase tracking (`LocalIntroPhase`), anchor points per region, and frame-based progression.

### `src/mapgen_gemini.rs`
Google Gemini API integration for AI-assisted content generation. Provides `call_gemini`, `call_gemini_text`, and `call_gemini_with_reference_image` with dotenv API key loading.

### `src/region_nav.rs`
Region target picker for overworld party navigation. Manages `RegionTargetPickerState`, region click detection, party camera focus transitions, region layer transitions, and direct command transfer between parties.

### `src/terrain.rs`
Overworld terrain scene setup. Spawns the 3D ground plane, directional light, and ambient lighting for the hub overworld view.

### `src/ui_helpers.rs`
Shared egui helper widgets including `gemini_illustration_tile` (styled image card), `split_faction_impact_sections` (faction effect text parsing), and other layout utilities.

---

## `src/game_core/` — Campaign & Mission State

Central module for all campaign data types, mission mechanics, and 30+ Bevy systems. Re-exports everything from `mod.rs`.

### `types.rs`
Core ECS types: `RunState` (global turn counter), `MissionData`, `MissionProgress`, `MissionSnapshot`, `MissionMap` (room metadata), `MissionBoard` (entity tracking), `HubScreen` enum, `HubUiState`, and `CharacterCreationState`.

### `overworld_types.rs`
Overworld state structures: `AttentionState` (focus/energy system), `OverworldRegion` (territory with faction ownership, unrest, control, intel), `FactionState`, `OverworldMap` (seeded world generation), `FlashpointChain`, `FlashpointState`, `DiplomacyState`, `CommanderState`, `InteractionBoard`.

### `roster_types.rs`
Hero and party types: `HeroCompanion` (stats, equipment, personality), `EquipmentItem`, `RecruitCandidate`, `CampaignRoster`, `CampaignParties`, `CampaignParty`, `CampaignLedger`, `CampaignEventLog`.

### `companion.rs`
Companion quest system with `CompanionQuestKind` (Reckoning, Homefront, RivalOath), quest generation, and `CompanionStoryState` for per-hero personal quest tracking.

### `generation.rs`
Procedural generation utilities: splitmix64 hashing, recruit codename/archetype generation, overworld region construction, and `overworld_region_plot_positions` for map rendering.

### `roster_gen.rs`
Recruit generation and roster synchronization. Creates `RecruitCandidate` instances with seeded stats and keeps hero lore aligned with overworld faction state.

### `setup.rs`
Scene initialization for test/headless environments. Spawns hero and enemy 3D models. Also contains `print_game_state` system and `attention_management_system`.

### `campaign_systems.rs`
Party management Bevy systems: `sync_campaign_parties_with_roster_system` and `campaign_party_orders_system` (AI patrol/reinforce/recruit movement).

### `campaign_outcome.rs`
Mission consequence application: XP/stat rewards, level-ups, loot generation (`generate_loot_drop`, `check_level_up`), hero injuries, and desertion checks.

### `consequence_systems.rs`
`resolve_mission_consequences_system` — processes mission results to update hero stress, fatigue, injury, and loyalty based on victory/defeat with alert-level scaling.

### `mission_systems.rs`
Turn-based mission mechanics: `turn_management_system`, `auto_increase_stress`, `activate_mission_system`, `mission_map_progression_system`, `player_command_input_system`, `hero_ability_system`, `enemy_ai_system`, `combat_system`, `complete_objective_system`, `end_mission_system`.

### `attention_systems.rs`
Focus and companion systems: `focused_attention_intervention_system`, `simulate_unfocused_missions_system`, `companion_mission_impact_system`, `companion_state_drift_system`, `companion_recovery_system`, `generate_companion_story_quests_system`, `progress_companion_story_quests_system`.

### `overworld_nav.rs`
Overworld navigation: `overworld_hub_input_system` for region cycling/travel, travel validation with energy/cooldown costs, and `focus_input_system`.

### `overworld_systems.rs`
Faction AI systems: `update_faction_war_goals_system`, `overworld_ai_border_pressure_system`, `overworld_faction_autonomy_system`, `overworld_intel_update_system`, `overworld_sync_from_missions_system`, `overworld_cooldown_system`, `sync_roster_lore_with_overworld_system`.

### `flashpoint_helpers.rs`
Internal flashpoint utility functions: companion hook detection (homefront/aggressor/defender), intent application, stage labeling, narrative text generation, and difficulty scaling.

### `flashpoint_spawn.rs`
`pressure_spawn_missions_system` — initiates flashpoint chains when region pressure exceeds thresholds, creating new mission slots with attached flashpoint state.

### `flashpoint_progression.rs`
`flashpoint_progression_system` — advances multi-stage flashpoint chains through victory conditions, applies companion hooks and intent modifications, and archives completed chains.

### `diplomacy_systems.rs`
`flashpoint_intent_input_system`, `generate_commander_intents_system`, `refresh_interaction_offers_system`, `interaction_offer_input_system`. Manages faction diplomacy, opinion matrices, and commander strategic planning.

### `save.rs`
Save/load data types: `CampaignSaveData` envelope (versioned), `LoadedCampaignData`, `load_campaign_data`, `load_and_prepare_campaign_data`, `normalize_loaded_campaign`, `validate_and_repair_loaded_campaign`.

### `migrate.rs`
Save version migration: `Migrate` trait with version-aware forward-compatible deserialization from v1 through v3 (current).

### `tests/`
Test suite: `helpers.rs` (shared world builders), `mission_tests.rs`, `companion_tests.rs`, `overworld_tests.rs`, `flashpoint_tests.rs`, `input_tests.rs`.

---

## `src/ai/` — Combat AI Pipeline

Multi-phase AI architecture for deterministic combat simulation. No Bevy dependency — runs identically in tests and in-game.

### `mod.rs`
Module root declaring all AI sub-modules.

### `core/` — Combat Simulation Engine
Pure-Rust deterministic fixed-tick (100ms) combat simulation. `SimState` holds units, projectiles, zones, tethers, and RNG state. `step()` advances one tick given unit intents, returning new state + events.

- **`types.rs`** — `SimState`, `UnitState`, `Team`, `CastState`, `SimVec2`, `FIXED_TICK_MS`
- **`events.rs`** — `SimEvent` enum with 30+ variants (damage, heal, CC, death, abilities, etc.)
- **`math.rs`** — `distance`, `move_towards`, `move_away`, `position_at_range`
- **`helpers.rs`** — `is_alive`, `find_unit_idx`, `check_tags_resisted`, utility functions
- **`conditions.rs`** — `evaluate_condition` for conditional effect triggers
- **`targeting.rs`** — `resolve_targets`, `units_in_radius` for area targeting
- **`apply_effect.rs`** / **`apply_effect_ext.rs`** — Main effect dispatcher handling all 32 effect types
- **`damage.rs`** — `apply_damage_to_unit`, `apply_heal_to_unit`, `resolve_chain_delivery`, `scale_effect`
- **`triggers.rs`** — Passive trigger system (`check_passive_triggers`, `fire_damage_triggers`)
- **`hero/`** — Hero ability resolution, split into `resolution.rs` (resolve_hero_ability, cooldown/charge/recast/toggle mechanics) and `reactions.rs` (apply_morph, apply_form_swap, check_zone_reactions, evolve_ability)
- **`intent.rs`** — Converts `IntentAction` into cast attempts (attack, ability, heal, control)
- **`resolve.rs`** — `try_start_cast`, `resolve_cast` for cast execution
- **`tick_systems.rs`** — Per-tick updates: cooldowns, status effects, projectile advancement, periodic passives
- **`tick_world.rs`** — Zone, channel, and tether tick updates
- **`simulation.rs`** — Main `step()` function, `run_replay()`, `ReplayResult`, hash/sample utilities
- **`metrics.rs`** — Post-simulation combat metrics computation
- **`tests/`** — Unit tests split into `determinism.rs`, `mechanics.rs`, `abilities.rs`
- **`tests_stress.rs`** — Stress tests

### `effects/` — Data-Driven Ability Engine
Defines 52 effect types and ability definitions loaded from TOML. Supports 7 delivery methods (Instant, Projectile, Channel, Zone, Tether, Trap, Chain), 8 targeting modes, and 3 damage types (Physical, Magic, True).

- **`effect_enum.rs`** — `Effect` enum with all 52 variants and serde defaults
- **`types.rs`** — `Area` (7 shapes), `Delivery` (7 methods), `Condition` (24), `Trigger` (19), `ConditionalEffect`, `DamageType`, `Stacking`, `Tags`
- **`defs.rs`** — `AbilityDef` (with recast, charges, toggle, unstoppable, form swap, evolve), `PassiveDef`, `AbilitySlot`, `PassiveSlot`, `AbilityTargeting` (8 modes), `HeroToml`, `HeroStats` (with armor/magic_resist), `ActiveStatusEffect`, `StatusKind`, `Projectile`, `AbilityTarget`
- **`tests.rs`** / **`tests_extended.rs`** — Effect parsing and behavior tests

### `squad/` — Phase 3 Squad AI
Personality-driven decision-making with force-based steering. 7 personality traits (aggression, compassion, caution, discipline, cunning, tenacity, patience) drive 9 tactical impulses (Attack, Heal, Retreat, Control, Focus, Protect, Pursue, Regroup, Position).

- **`personality.rs`** — `Personality` struct, `infer_personality` from unit stats
- **`forces.rs`** — `RawForces`, weight matrix, `compute_raw_forces` from personality + context
- **`state.rs`** — `SquadAiState`, `SquadBlackboard`, `FormationMode`, `TickContext`
- **`combat/`** — Split into `healer.rs` (healer intent, backline positioning, heal evaluation), `targeting.rs` (choose_target, target scoring), `abilities.rs` (evaluate_hero_ability, choose_action)
- **`intents.rs`** — Core force-based steering loop `generate_intents`
- **`replay.rs`** — Benchmark and replay utilities
- **`tests.rs`** — Squad AI behavior tests

### `advanced/` — Phase 9 Coordination
Encounter-level pressure system with team-wide burst coordination, spatial analysis, and anti-stack movement.

- **`spatial.rs`** — Spatial types, pressure state, visibility, engagement metrics
- **`tactics.rs`** — Tactical AI: environment-reactive intents, anti-stack stepping
- **`horde.rs`** — Horde chokepoint scenario builders (3 heroes vs 12 enemies)
- **`tests.rs`** — Advanced AI tests

### `tooling/` — AI Development Tools
Scenario matrix running, visualization HTML generation, custom scenario import/export, and debug utilities.

- **`types.rs`** — `CustomScenario`, `ScenarioUnit`, `ScenarioObstacle`, `ScenarioSummary`
- **`debug.rs`** — `build_phase5_debug`, `score_candidates` for AI decision debugging
- **`scenarios.rs`** — `run_scenario_matrix`, `analyze_phase4_cc_metrics`, `run_personality_grid_tuning`
- **`events.rs`** — `build_event_rows`, `build_frame_rows` for visualization data
- **`viz_template.rs`** — HTML visualization template builder
- **`custom.rs`** — Custom scenario handling and all `export_*` functions
- **`tests.rs`** — Tooling tests

### `personality/` — Personality Traits
6-trait personality profile with preset archetypes (Vanguard, Guardian, Tactician). Personalities drift over time based on mission outcomes.

- **`types.rs`** — `PersonalityProfile`, `PersonalityArchetype`, trait definitions, presets
- **`tests.rs`** — Personality inference and drift tests

### `roles/` — Combat Roles
Tank/DPS/Healer role system with preferred ranges, threat tables, and sticky target lock (4-tick lock after switch).

- **`types.rs`** — `Role`, `RoleProfile`, threat table, target selection logic
- **`sample.rs`** — Sample role configurations and replay builders
- **`tests.rs`** — Role behavior tests

### `student/` — Distilled AI Model
Tiny 2-layer MLP (~17K parameters) approximating the full AI pipeline. 60 input features → 128 hidden → 64 hidden → 9 outputs (6 personality weights + 3 formation logits). Uses AVX2+FMA SIMD.

- **`model.rs`** — `StudentMLP` struct, forward pass, SIMD kernels
- **`features.rs`** — Feature extraction from `SimState` (HP ratios, cooldowns, spatial metrics)
- **`tests.rs`** — Model inference tests

### `core/ability_encoding/` — Ability Embedding Autoencoder
Extracts 80-dim property vectors from `AbilityDef` TOML and encodes them into 32-dim L2-normalized embeddings via a trained autoencoder (80→64→32→64→80). Used by self-play to give the policy semantic understanding of abilities.

- **`properties.rs`** — `extract_ability_properties()`, `ability_category_label()`, constants
- **`autoencoder.rs`** — `FlatMLP`, `AbilityEncoder`, `AbilityDecoder`, `load_autoencoder()`
- **`effects.rs`** — `EffectSummary`, `collect_all_effects()`, `summarize_effects()`
- **`training.rs`** — `AbilityTrainingRow`, `extract_training_rows()`
- **`tests.rs`** — Encoder round-trip and property extraction tests

### `core/ability_eval/` — Ability Evaluation System
Per-ability micro-models across 9 categories with terrain-aware feature extraction.

- **`categories.rs`** — `AbilityCategory` enum, classification logic
- **`features.rs`** / **`features_aoe.rs`** — 115-dim feature extraction per category
- **`weights.rs`** — `EvalWeights`, `AbilityEvalWeights`
- **`eval.rs`** — `evaluate_abilities`, `evaluate_abilities_with_encoder`
- **`oracle_scoring.rs`** — `oracle_score_ability`
- **`dataset.rs`** — `AbilityEvalSample`, dataset generation and I/O

### `core/dataset/` — Training Dataset Generation
Action classification and feature extraction for ML training pipelines.

- **`actions.rs`** — `ActionClass` (10-class), `CombatActionClass` (5-class), classification functions
- **`features.rs`** — 115-dim feature extraction (`extract_unit_features`)
- **`training.rs`** — `TrainingSample`, `CombatTrainingSample`, `RawTrainingSample`, dataset generation and I/O

### `core/oracle/` — Decision Oracle & Rollouts
Monte Carlo rollout-based action scoring for optimal play.

- **`rollout.rs`** — `enumerate_candidates`, `run_rollout`, `score_actions`, `score_actions_with_depth`
- **`squad.rs`** — `squad_oracle`, `run_squad_rollout`
- **`focus.rs`** — `FocusCandidate`, `search_focus_target`

### `core/self_play/` — Self-Play Policy Learning
Policy gradient training pipeline with episode generation.

- **`features.rs`** — `encode_unit`, `extract_features`, `extract_features_encoded`
- **`actions.rs`** — `action_mask`, `action_to_intent`
- **`policy.rs`** — `PolicyWeights`, `masked_softmax`, `Episode`, `Step`
- **`episode.rs`** — `run_episode`, `run_episode_greedy`, `write_episodes`, `load_policy`

### `control.rs`
Phase 4 CC coordination layer. CC reservation system to avoid overlapping stuns, diminishing returns windows, and CC profile tracking.

### `utility.rs`
Phase 1 utility scoring AI. Score-based action selection with distance weight, damage weight, overkill penalty, stickiness bonus, and target lock.

### `pathing/`
A* grid navigation with elevation and slope costs.

- **`mod.rs`** — `GridNav` struct, grid utilities (block/carve rects, elevation)
- **`navigation.rs`** — A* pathfinding (`next_waypoint`), LoS, raycast, terrain-biased stepping

### `phase.rs`
`AiPhase` trait defining the common interface: `generate_intents(&mut self, state, dt_ms) -> Vec<UnitIntent>` and optional `update_from_events`.

---

## `src/mission/` — Mission Execution & Visuals

### `mod.rs`
Module root exporting all mission sub-modules.

### `objectives.rs`
Mission objectives: Eliminate, Hold (zone for N ticks), Extract (escort NPC), Sabotage (activate nodes). `check_objective_system` evaluates per-tick, `draw_objective_hud_system` renders progress, `generate_objective` creates from seed.

### `hero_templates.rs`
Hero TOML template loading. Built-in archetypes (Warrior, Ranger, Mage, Cleric, Rogue, Paladin) plus 20 data-driven templates from `assets/hero_templates/`. `parse_hero_toml`, `hero_toml_to_unit`, `load_embedded_templates`.

### `tag_color.rs`
Maps ability/damage tags (fire, ice, lightning, etc.) to display colors for VFX and UI.

### `unit_vis.rs`
Unit 3D visualization components: `UnitVisual`, `SelectionRing`, `HpBarFg`/`HpBarBg`, `UnitSelection`. Systems: `update_unit_positions`, `update_hp_bars`, `update_unit_selection_rings`.

### `enemy_templates/`
Enemy unit archetypes (Grunt, Brute, Mystic, Sentinel, Berserker, Summoner) and wave generation. `build_enemy_unit`, `default_enemy_wave`, `generate_boss`.

- **`templates.rs`** — Enemy archetype definitions and stat blocks
- **`waves.rs`** — Wave composition logic and difficulty scaling

### `execution/`
Mission scene lifecycle: enter/exit transitions, sim-to-visual syncing, ability HUD, and outcome UI.

- **`setup.rs`** — `mission_scene_transition_system`, scene initialization and teardown
- **`ui.rs`** — `ability_hud_system`, `mission_outcome_ui_system`, `sync_sim_to_visuals_system`

### `room_gen/`
Deterministic procedural room generation from seed + room type. `NavGrid` (walkability/elevation), `RoomLayout` (floors, walls, obstacles, spawn zones).

- **`mod.rs`** — `generate_room`, `spawn_room`, room type definitions
- **`lcg.rs`** — Linear congruential generator for deterministic randomness
- **`nav.rs`** — Navigation grid construction and queries
- **`primitives.rs`** — Floor, wall, obstacle geometry primitives
- **`templates.rs`** — Per-room-type layout templates (Entry, Pressure, Pivot, Setpiece, Recovery, Climax)
- **`visuals.rs`** — 3D mesh/material spawning for room geometry

### `room_sequence/`
Multi-room progression through a mission. Sequences rooms by difficulty curve and handles room-to-room transitions.

- **`types.rs`** — `MissionRoomSequence`, room ordering types
- **`systems.rs`** — `spawn_room_door_system`, `advance_room_system`

### `sim_bridge/`
Bridges Bevy ECS with the pure-Rust combat simulation. Manages `MissionSimState`, `EnemyAiState`, `PlayerOrderState`, and `SimEventBuffer`.

- **`types.rs`** — Bridge resource types and marker components
- **`builders.rs`** — `build_sim_with_templates`, `build_sim_with_hero_templates`, `scale_enemy_stats`, `threat_level`
- **`systems.rs`** — `advance_sim_system`, `apply_vfx_from_sim_events_system`, `apply_audio_sfx_from_sim_events_system`, `player_ground_click_system`, `apply_player_orders_system`

### `vfx/`
Visual effects system driven by `VfxEventQueue`. Floating text, hit flash, death fade, projectiles, zones, tethers, shields, status indicators.

- **`types.rs`** — `VfxEvent` enum, `VfxEventQueue`, VFX component types
- **`spawn.rs`** — `spawn_vfx_system`, `update_floating_text_system`, `update_hit_flash_system`, `update_death_fade_system`
- **`sync.rs`** — `sync_projectile_visuals_system`, `sync_zone_visuals_system`, `sync_tether_visuals_system`
- **`indicators.rs`** — `sync_shield_indicators_system`, `sync_status_indicators_system`, `sync_buff_debuff_rings_system`, `emit_dot_hot_particles_system`, `update_channel_ring_system`, `update_zone_pulse_system`

---

## `src/hub_ui_draw/` — Hub UI Rendering

All hub screen egui rendering, delegated from the main `draw_hub_egui_system`.

### `mod.rs`
Main `draw_hub_egui_system` Bevy system. Pre-computes shared values (board rows, save slot metadata), then dispatches to sub-module draw functions based on `HubScreen`.

### `common.rs`
Shared hub UI helpers: `apply_hub_style` (dark theme), `draw_credits_window`, `faction_color` palette.

### `start_menu.rs`
Start menu rendering: `draw_gui_only_screens` (full-screen overlay for StartMenu + CharCreation), `draw_start_menu_side_panel` (save slots, continue/new campaign buttons, settings access).

### `character_create.rs`
Side panel faction and backstory selection UI: `draw_faction_side_panel`, `draw_backstory_side_panel`.

### `character_creation_center.rs`
Center panel rendering for character creation: `draw_faction_center` (faction cards with impact descriptions), `draw_backstory_center` (backstory choices with stat modifiers).

### `guild.rs`
Guild management screen: action grid (Assemble Expedition, Review Recruits, Intel Sweep, Dispatch Relief), mission board table, roster list, hero detail panel with stats/equipment/loot.

### `overworld.rs`
Overworld 3D view side panel: region details, faction intel, diplomatic interaction options.

### `overworld_map.rs`
Overworld map side panel: party controls, region target picker, map navigation, transition controls.

### `overworld_map_strategic.rs`
Hex-tile strategic map rendering: region hexes colored by faction, click-to-select regions, party position markers.

### `overworld_map_parties.rs`
Party management panels: player party controls, delegated party lists, faction control summary, active crises right panel.

### `region.rs`
Region view and local eagle-eye intro screens: region art display, party deployment UI, local intro cinematic overlay.

---

## `src/ui/` — Shared UI Systems

### `mod.rs`
Module root declaring save_browser, quest_log, settings, tutorial sub-modules.

### `save_browser.rs`
Save/load system with 3 manual slots (F5/Shift+F5/Ctrl+F5), autosave (every 10 turns), save index tracking, version migration (v1-v3), and F6 save panel UI.

### `quest_log.rs`
Quest log overlay toggled with J key. Displays per-hero companion quests with progress tracking.

### `settings.rs`
Settings menu toggled with Esc. Camera sensitivity sliders, zoom sensitivity, invert Y toggle, screenshot capture on spacebar. Persisted to camera settings JSON.

### `tutorial.rs`
Tutorial tooltip overlay toggled with T key. Step-by-step gameplay guidance.

---

## `src/backstory_cinematic/` — Backstory Sequences

AI-generated backstory cinematic system using Gemini for narrative text and character portraits.

- **`types.rs`** — `BackstoryCinematicPhase`, `BackstoryCinematicBeat`, `BackstoryNarrativeResult`, `BackstoryNarrativeGenState`, `BackstoryCinematicState`
- **`builders.rs`** — Beat construction, narrative prompt generation, portrait prompt generation, job queuing
- **`systems.rs`** — Bevy systems: bootstrap, dispatch, collect, texture load, playback, draw egui, state reset

---

## `src/campaign_ops/` — Campaign Operations

Save/load operations, campaign initialization, and world snapshot utilities.

- **`mod.rs`** — Re-exports + utility functions: `spawn_mission_entities_from_snapshots`, `despawn_all_mission_entities`, campaign progress snapshot/apply, format helpers (`format_slot_meta`, `format_slot_badge`, `truncate_for_hud`)
- **`save_load.rs`** — `save_campaign_data`, `snapshot_campaign_from_world`, `apply_loaded_campaign_to_world`, `save_campaign_to_slot`, `load_campaign_from_path_into_world`, `hub_continue_campaign_requested_system`
- **`initialization.rs`** — `new_campaign_seed`, `initialize_new_campaign_world`, `hub_new_campaign_requested_system`, `enter_start_menu`

---

## `src/events/` — Campaign Events

Random events generated each turn via seeded LCG (MerchantOffer, DeserterIntel, PlagueScare, RivalPartySpotted, AllyRequest, AbandonedCache, StormWarning, etc.).

- **`types.rs`** — `CampaignEvent`, `CampaignEventKind`, `CampaignEventQueue`
- **`generation.rs`** — `campaign_event_generation_system` (per-turn event generation)
- **`effects.rs`** — `draw_event_notification_system` (egui UI with accept/decline)

---

## `src/runtime_assets/` — AI Art Pipeline

Gemini-powered runtime asset generation for region art and menu backgrounds.

- **`types.rs`** — `RuntimeAssetGenState`, `RuntimeAssetPreviewState`, `RegionArtState`, generation job types
- **`systems.rs`** — Bootstrap, dispatch, collect, preview update, region art loading, menu background drawing

---

## `src/scenario_3d/` — 3D Scenario Viewer

3D replay viewer for combat scenarios with playback controls and speed adjustment.

- **`types.rs`** — `Scenario3dData`, `ScenarioReplay`, `ScenarioPlaybackSpeed`, UI marker components
- **`setup.rs`** — `setup_custom_scenario_scene`, `setup_scenario_playback_ui`, `setup_mission_hud`
- **`systems.rs`** — `advance_scenario_3d_replay_system`, `scenario_replay_keyboard_controls_system`, `update_scenario_hud_system`, playback slider systems, `update_mission_hud_system`
- **`mod.rs`** — `build_horde_3d_bundle`, frame builder, chokepoint obstacle definitions

---

## `src/screenshot_capture/` — Screenshot System

Automated screenshot capture in single, sequence, and hub-stages modes.

- **`types.rs`** — `ScreenshotCaptureConfig`, `ScreenshotCaptureState`, `ScreenshotMode` (Single, Sequence, HubStages)
- **`capture.rs`** — `screenshot_capture_system` with warmup frames, max captures, and attempt limits

---

## `src/simulation_cli/` — Headless Simulation Runners

CLI-invoked simulation phases for balance testing without rendering.

- **`phases.rs`** — `run_phase0_simulation` through `run_phase9_simulation`, `run_phase6_report`
- **`pathing.rs`** — `run_pathing_simulation`, `run_pathing_hero_win_simulation`, `run_pathing_hero_hp_ablation_simulation`
- **`visualization.rs`** — `run_phase6_visualization`, `run_pathing_visualization`, `run_visualization_index`, `run_write_scenario_template`, `run_custom_scenario_visualization`

---

## `src/mapgen_voronoi/` — Procedural Overworld Generation

Voronoi-based overworld map generation producing region boundaries, faction assignments, roads, and settlements.

- **`types.rs`** — `VoronoiSpec`, `VoronoiRegion`, grid dimensions
- **`partition.rs`** — Voronoi partitioning algorithm and region boundary computation
- **`spec.rs`** — Spec construction from partition results, faction assignment, road/settlement placement

---

## `src/scenario/` — Headless Balance Testing

Scenario runner for automated balance testing and CI. Loads scenarios from TOML, runs combat simulations, captures per-unit statistics, and checks assertions. Includes a Rust-native scenario generation engine.

- **`types.rs`** — `ScenarioCfg`, `ScenarioResult`, `ScenarioAssertions`, `AssertionResult`
- **`runner.rs`** — `run_scenario`, `check_assertions`, `run_scenario_to_state`
- **`simulation.rs`** — Scenario initialization, enemy scaling, room generation integration
- **`gen/`** — Coverage-driven scenario generation engine: `metadata.rs` (hero/role definitions, RNG), `coverage.rs` (coverage tracking), `strategies.rs` (6 generation strategies)

---

## `src/tests/` — Integration Tests

Integration tests for the main binary crate, organized by subsystem.

- **`mod.rs`** — Shared imports and test helpers: `build_campaign_test_world`, `build_campaign_sim_schedule`, `canonical_roundtrip_world`, `campaign_signature_from_world`
- **`hub_tests.rs`** — Hub UI, character creation, faction/backstory selection, party command tests
- **`region_tests.rs`** — Region transitions, local intro, camera focus, start menu tests
- **`campaign_save_tests.rs`** — Save JSON roundtrip, file I/O, party field validation
- **`campaign_load_tests.rs`** — Load/restore layer context, region resume, version compatibility
- **`campaign_index_tests.rs`** — Save slot paths, index upsert, save panel, continue candidates, migration
- **`simulation_tests.rs`** — Long-run save/load drift, settings smoke tests, hub action tests
- **`regression_tests.rs`** — Regression fixture capture, determinism, baseline comparison

---

## `src/bin/` — Binary Crates

### `gen_scenarios.rs` (deprecated — replaced by `xtask scenario generate`)
Legacy Python-based scenario generator. Superseded by the Rust-native engine in `src/scenario/gen.rs`.

### `room_preview.rs`
Room visualization tool that generates and renders procedural room layouts for inspection.

### `sim_bridge/`
Standalone simulation bridge binary for external AI integration via JSON wire protocol.

- **`main.rs`** — Entry point, JSON stdin/stdout loop, simulation advancement
- **`types.rs`** — Wire protocol structs: `InitMessage`, `DecisionMessage`, `CondensedUnit`
- **`helpers.rs`** — State condensation, event condensation, role assignment, string formatting

### `xtask/`
Build and development task runner (clap CLI).

- **`main.rs`** — Entry point dispatching to subcommand modules
- **`cli/`** — Clap argument and subcommand definitions: `mod.rs` (Args, TaskCommand), `scenario.rs` (scenario/oracle/self-play args), `map.rs` (map/env-art/capture args)
- **`map.rs`** — `run_map_gemini`, `run_map_voronoi` overworld generation commands
- **`env_art.rs`** — Environment art index build/query/generate with HNSW embedding search
- **`capture.rs`** — `run_capture_windows`, `run_capture_dedupe` screenshot utilities
- **`ralph.rs`** — `run_ralph_status` PRD quality gate checker
- **`scenario_cmd.rs`** — `run_scenario_run`, `run_scenario_bench`, `run_scenario_generate`, balance testing commands
- **`oracle_cmd/`** — Oracle/training subcommands: `eval.rs` (eval/play), `dataset.rs` (dataset gen), `training.rs` (student model, encoder export), `selfplay.rs` (self-play episodes, raw dataset), `transformer_rl.rs` (V3/V4 RL episode generation and eval), `ability_profile.rs` (behavioral profiling), `operator_dataset.rs` (operator dataset gen)

---

## ML Training Pipeline

End-to-end pipeline for training combat AI models. Three pretrained components feed into a V4 actor-critic policy trained via IMPALA RL.

### Architecture Overview

```
                        ┌─────────────────────────┐
                        │  Ability DSL Text        │
                        │  "ability fireball {     │
                        │    damage 50 fire ..."   │
                        └────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  Ability Transformer (d=128) │  ← Pretrained (MLM + behavioral)
                    │  4-layer, 4-head encoder     │
                    │  252-token vocab             │
                    └────────────┬────────────────┘
                                 │
                         [CLS] embedding (128d)
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              ▼                  ▼                   │
   ┌──────────────────┐  ┌──────────────┐           │
   │ Embedding        │  │ Behavioral   │           │
   │ Registry         │  │ Head         │           │
   │ (943 abilities   │  │ (119-dim     │           │
   │  × 128d, cached) │  │  sim outcome │           │
   └────────┬─────────┘  │  prediction) │           │
            │             └──────────────┘           │
            │                                        │
            ▼                                        │
   ┌────────────────┐                                │
   │ external_cls_  │                                │
   │ proj (128→32)  │  ← Learned during RL           │
   └────────┬───────┘                                │
            │                                        │
            ▼          ┌─────────────────────┐       │
   ┌────────────────┐  │ Entity Encoder (d=32)│      │
   │ Cross-Attention│◄─┤ 4-layer, 4-head     │      │
   │ Block          │  │ 7 entity slots ×    │      │
   │ (ability ×     │  │ 30 features each    │      │
   │  game state)   │  └─────────────────────┘      │
   └───┬────────┬───┘                                │
       │        │                                    │
       ▼        ▼                                    │
  ┌─────────┐ ┌──────────────┐                      │
  │ Move    │ │ Combat       │                      │
  │ Head    │ │ Pointer Head │                      │
  │ (9-way) │ │ (type+target)│                      │
  └─────────┘ └──────────────┘                      │
                                                     │
  ┌─────────┐                                        │
  │ Value   │  ← Used during training only           │
  │ Head    │                                        │
  └─────────┘
```

### Component 1: Ability Transformer (Pretrained, d=128)

Encodes ability DSL text into semantic embeddings that capture what each ability *does* in simulation.

**Training (two-phase curriculum):**
1. **Phase 1 — MLM**: Masked language modeling on 75K ability DSL texts. Learns syntax and token relationships. Checkpoint: `generated/ability_transformer_pretrained_v6_base.pt`
2. **Phase 2 — Behavioral finetuning**: Encoder frozen. A behavioral head learns to predict 119-dim sim outcome vectors (damage dealt, healing done, CC applied, etc.) from [CLS] tokens. Z-normalized targets + Huber loss. Checkpoint: `generated/ability_transformer_pretrained_v6.pt`

**Key insight:** Joint MLM+behavioral training destroys MLM performance. Unfreezing the encoder causes catastrophic forgetting (97.5% → 46% MLM accuracy). The curriculum with frozen encoder is essential.

| File | Role |
|------|------|
| `training/pretrain.py` | MLM pretraining + behavioral finetuning |
| `training/tokenizer.py` | 252-token DSL tokenizer |
| `training/model.py` → `AbilityTransformerMLM` | Model definition |
| `src/ai/core/ability_transformer/tokenizer.rs` | Rust port of tokenizer |
| `src/ai/core/ability_transformer/mod.rs` | Rust transformer inference |

### Component 2: Behavioral Embedding Registry

Pre-computed 128-dim [CLS] embeddings for all 943 known abilities, extracted from the pretrained ability transformer.

**Why pre-compute?** The d=128 ability transformer is too large to run per-tick at inference time. Instead, all known abilities are encoded once and stored in a JSON registry. At runtime, Rust looks up each ability by name — no transformer forward pass needed. Only the lightweight `external_cls_proj` (128→32 linear) runs per tick.

**For unknown abilities:** The full transformer can be run as fallback, but in practice all abilities in the game are in the registry.

**Behavioral profiling data** (used to train the behavioral head):
- `src/bin/xtask/oracle_cmd/ability_profile.rs` — Controlled sim experiments: 1 caster + N targets across 144 condition combinations (HP × distance × targets × armor)
- Output: `dataset/ability_profiles.npz` (135,792 samples, 943 abilities, 119-dim outcome vectors)

| File | Role |
|------|------|
| `training/export_embedding_registry.py` | Extract CLS embeddings from pretrained transformer |
| `generated/ability_embedding_registry.json` | Registry (943 × 128d, 2.5 MB, includes model hash) |
| `src/ai/core/ability_transformer/weights.rs` → `EmbeddingRegistry` | Rust loading and lookup |

### Component 3: Entity Encoder (d=32, trained with RL)

Encodes game state (7 entity slots × 30 features each) into d=32 tokens for cross-attention with ability embeddings.

**Entity features (30 per slot):** vitals (5), position/terrain (7), combat stats (3), ability readiness (3), healing state (3), CC state (3), unit state flags (4), cumulative stats (2).

**Entity slots:** Actor (self), up to 3 allies, up to 3 enemies. Missing slots are zero-masked.

| File | Role |
|------|------|
| `training/model.py` → `EntityEncoderV3` | Python model (self-attention over entity tokens) |
| `src/ai/core/ability_eval/game_state.rs` | Rust feature extraction from `SimState` |
| `src/ai/core/ability_transformer/weights.rs` → `FlatEntityEncoderV3` | Rust inference |

### Component 4: V4 Actor-Critic (113K params)

Dual-head policy for IMPALA RL training. Combines all pretrained components.

**Action space:**
- **Move head:** 9-way directional (8 cardinal + stay) — every tick
- **Combat pointer head:** action type (attack/hold + up to 8 abilities) + target pointer via scaled dot-product attention over entity tokens — every tick

**Training:** IMPALA (Importance Weighted Actor-Learner Architecture) with V-trace off-policy correction. Rust generates episodes in parallel using shared-memory GPU inference, Python trains on batches.

| File | Role |
|------|------|
| `training/model.py` → `AbilityActorCriticV4` | Full model definition |
| `training/impala_learner.py` | IMPALA training loop |
| `training/gpu_inference_server.py` | GPU shared-memory inference server |
| `training/export_actor_critic_v4.py` | Export to JSON for Rust |
| `src/bin/xtask/oracle_cmd/transformer_rl.rs` | Rust episode generation + eval |
| `src/ai/core/ability_transformer/weights.rs` → `ActorCriticWeightsV4` | Rust inference |
| `scripts/impala_curriculum.sh` | Curriculum training script |

### Training Data

| Dataset | Source | File |
|---------|--------|------|
| Ability DSL texts | All `.ability` files + hero templates | `generated/ability_dataset_curated.npz` |
| Behavioral profiles | Controlled sim experiments | `dataset/ability_profiles.npz` |
| Hero scenarios (HvH) | Generated compositions | `dataset/scenarios/` (~474 TOML files) |
| Tier 1 heroes | Autoattack-only, 10× HP | `dataset/heroes/tier1_autoattack/` |
| Tier 2 heroes | One ability each, 10× HP | `dataset/heroes/tier2_one_ability/` |

### Episode Generation (Rust → Python)

```
cargo run -p xtask -- scenario oracle transformer-rl generate \
    dataset/scenarios/ \
    --weights generated/actor_critic_v4.pt \
    --embedding-registry generated/ability_embedding_registry.json \
    --gpu-shm /dev/shm/impala_inf \
    --threads 64 --sims-per-thread 64 \
    --episodes 5 --temperature 1.0 \
    -o generated/rl_episodes.jsonl
```

64 threads × 64 concurrent sims = 4,096 parallel episodes. GPU inference server handles batched forward passes via shared memory (`/dev/shm/impala_inf`). Episodes are written as JSONL with per-step observations, actions, rewards, and log probabilities.
