# Changelog

## Unreleased

### Added
- Initial Bevy project setup.
- Project re-structured to focus on Bevy development.
- Documentation consolidated and updated to reflect Bevy migration.
- Deterministic multi-mission triage skeleton with shared attention economy, focus switching, and unattended mission simulation.
- Focused intervention leverage system that spends global attention energy for mission progress and alert control.
- Mission queue HUD extensions for attention state, focus, unattended pressure, assigned hero visibility, and last consequence summary.
- Deterministic hub action pipeline (`Assemble Expedition`, `Review Recruits`, `Intel Sweep`, `Dispatch Relief`) with direct mission-board and attention effects.
- Procedural deterministic campaign roster and recruit generation, including persistent recruit pool and signing flow.
- Companion assignment/impact systems that apply roster state to mission progression each turn.
- Persistent consequence resolution pipeline with one-time outcome ledgering, injury/loyalty/stress drift, desertion rules, and timed recovery.
- Regression and determinism tests for triage cycles, hub action sequences, companion persistence, and consequence propagation.
- Deterministic overworld campaign graph with region unrest/control, mission-slot linkage, travel cooldown/energy rules, and hub travel controls.
- Overworld telemetry integrated into hub and mission HUDs, plus deterministic tests for travel constraints and pressure sync.
- Faction autonomy layer with strength-scaled vassal counts, stationary zone managers vs roaming vassals, and manager bonuses applied per owned region.
- NPC faction commander analogues that generate crisis intents and feed diplomacy interaction offers (joint missions, raids, training loans, recruit borrowing).
- Player interaction board inputs (`U/O` select, `Y/N` resolve) with mission, roster, attention, and relation consequences.
- Seeded hex-style overworld generator with faction boundary assignment, mission-slot placement per faction, and deterministic seeded construction.
- Runtime `--map-seed` support for reproducible overworld variants and hub HUD seed visibility.
- Faction war-goal computation driven by diplomacy and border friction, with war-focus values feeding commander intent urgency.
- AI-vs-AI border pressure simulation that perturbs unrest/control each turn and can transfer region ownership while preserving faction viability.
- Region intel/fog model with decay, mission-signal discovery, travel-based reveals, and hub telemetry that degrades from exact to approximate to unknown.
- Pressure-to-mission bridge: contested regional pressure now spawns/rebinds slot missions to regional flashpoints with deterministic crisis templates.
- Faction-aware roster lore: recruits and companions now carry origin faction/region metadata plus backstories generated from procedural overworld ownership and pressure context.
- Companion story quest framework: pressure-triggered personal quests per hero with active/completed/failed states, ledger-driven progress, and loyalty/resolve rewards.
- Campaign persistence: full-state save/load support (quick save/load via `F5`/`F9` and startup `--load-campaign <path>`), including overworld, missions, roster, diplomacy, ledger, and companion quest state.
- Persistent campaign event feed that records major outcomes (mission resolutions, companion quest updates, border ownership shifts), displays in HUD, and persists through save/load.
- Versioned save schema (`save_version`) with migration scaffolding for legacy saves and compatibility checks for unsupported newer versions.
- Multi-slot save controls (`F5/F9` slot 1, `Shift+F5/F9` slot 2, `Ctrl+F5/F9` slot 3) plus turn-based autosave (`generated/saves/campaign_autosave.json`).
- Save reliability hardening: atomic temp-write + rename flow with `.bak` backup of overwritten slot saves.
- Post-load consistency validation/repair for assignments, region ownership, mission bindings, diplomacy matrix, and quest references, with repair notes emitted to campaign events.
- Save slot metadata index (`generated/saves/campaign_index.json`) with turn/version/seed/timestamp/compatibility, surfaced in HUD for quick slot inspection.
- Save-slot browser panel (`F6`) with selection, compatibility badges, preview details, and confirm-before-load flow (`S` save, `G` preview load, `Enter` confirm).
- Flashpoint vertical-slice loop: pressure can now open staged regional crisis chains (`1/3 Recon`, `2/3 Sabotage`, `3/3 Decisive`) with stage promotion on mission outcomes.
- Flashpoint decisive outcomes now apply campaign consequences (border ownership shifts, faction stat swings, recruit unlock/lock effects) and emit campaign events.
- Companion-specific flashpoint hooks now tune stage profiles/objective flavor (`Homefront Oath`, `Rival Banner`, `Oathbound Defense`) and apply extra morale/renown effects on final resolution.
- Player-facing flashpoint intent controls (`1/2/3`) added for per-stage strategy selection (`Stealth Push`, `Direct Assault`, `Civilian First`) with immediate mission profile retuning.
- Flashpoint missions now include objective/projection telemetry in mission names (hook objective + projected win/loss consequences) for hub and mission queue visibility.
- Flashpoint hook outcomes now feed companion story beats by advancing or regressing active companion quest progress on crisis resolution.
- Flashpoint state now persists in campaign saves and is normalized/repaired on load.
- Added deterministic tests for flashpoint spawn, stage progression, and decisive outcome effects.

---
**Note:** Entries below this line are for the legacy Python/TypeScript project and are no longer relevant to the active Bevy development.
---
