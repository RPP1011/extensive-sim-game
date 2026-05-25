//! Auto-discovers `assets/sim/*.sim` and emits one `pub mod <fixture>`
//! sub-module per file, each with its compiler-emitted `generated.rs` +
//! `runtime_core.rs` included. Plan E-A6 follow-up — eliminates the
//! per-fixture `*_runtime` crate boilerplate.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Workspace root: $CARGO_MANIFEST_DIR is `<root>/crates/sims`,
    // so two parents up is the workspace.
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/sims");
    let sims_dir = workspace_root.join("assets/sim");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", sims_dir.display());

    let mut fixtures: Vec<String> = Vec::new();
    let entries = fs::read_dir(&sims_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", sims_dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("sim") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        // The mega-crate covers fixtures that have been actively
        // migrated off the per-`*_runtime` crate path. Add new
        // fixtures here as they migrate. Other fixtures stay on
        // their per-fixture crate until each is moved over.
        if !matches!(
            stem.as_str(),
            // Migrated fixtures live under `sims::<fixture>::*` and have
            // no per-fixture *_runtime crate. Add to this list as each
            // .sim parses cleanly + its handcoded crate gets deleted.
            "threat_impact"
                | "boids"
                | "cooldown_probe"
                | "for_each_agent_probe"
                | "input_probe"
                | "target_chaser"
                | "stochastic_probe"
                | "quest_probe"
                | "stdlib_math_probe"
                | "bartering"
                | "pair_scoring_probe"
                | "param_rule_smoke"
                | "per_entity_ring_probe"
                | "crafting_diffusion"
                | "flocking_skirmish"
                | "abilities_probe"
                | "apply_ability_chronicle_consumer"
                | "apply_ability_verb_chronicle_consumer"
                | "apply_ability_verb_smoke"
                | "auction_market"
                | "boss_fight"
                | "diplomacy_probe"
                | "duel_1v1"
                | "duel_abilities"
                | "dungeon_crawl"
                | "f32_reduction_probe"
                | "firebolt_interrupt_probe"
                | "firebolt_probe"
                | "foraging_colony"
                | "foraging_real"
                | "mass_battle_100v100"
                | "megaswarm_1000"
                | "megaswarm_10000"
                | "multi_zone_world"
                | "objective_capture_10v10"
                | "predator_prey_real"
                | "quest_arc_real"
                | "scripted_battle"
                | "spy_network"
                | "tactical_horde_500"
                | "tactical_squad_5v5"
                | "threat_stresstest"
                | "threats_struct_probe"
                | "threats_view_probe"
                | "threats_with_decay_probe"
                | "tower_defense"
                | "trade_market_probe"
                | "trade_market_real"
                | "trophic_3tier"
                | "verb_fire_probe"
                | "village_day_cycle"
                | "wave_defense"
                | "duel_25v25"
                | "village_economy"
                | "swarm_event_storm"
                | "ecosystem_cascade"
                | "apply_ability_smoke"
                | "voxel_probe"
                | "dodger_probe"
                | "tom_probe"
                | "debug_probe"
                | "stress_agent_count"
                | "stress_cast_density"
                | "predator_prey"
                | "particle_collision"
                | "crowd_navigation"
                | "dsl_stress_coverage"
                | "hill_raid"
                | "trade_caravans"
                | "palace_coup"
                | "pirate_fleet"
                | "forest_fire"
                | "squad_skirmish"
                | "plague_city"
                | "detective_investigation"
                | "among_us"
                | "dungeon_layout"
                | "dungeon_stealth"
                | "dungeon_horde"
                | "threat_horizon_stresstest"
                | "belief_smoke_probe"
                | "belief_merge_propagation_probe"
                | "room_known_pattern_probe"
                | "belief_merge_ops_probe"
                | "belief_key_typed_probe"
                | "maze_explorer_belief_smart"
                | "maze_explorer_multi"
                | "maze_explorer_smart"
                | "maze_explorer_visited"
                | "maze_explorer"
                | "assassination_threat_test"
                | "threat_scoring_stresstest"
                | "navgrid_probe"
                | "terrain_probe"
                | "terrain_probe_imported"
                | "vampire_survivors"
        ) {
            continue;
        }
        fixtures.push(stem);
    }
    fixtures.sort();

    // Resolve stdlib and sandbox roots once (same logic as build_helper::emit_into).
    let stdlib_root: PathBuf = match env::var_os("WORLDSIM_STDLIB_ROOT") {
        Some(s) => PathBuf::from(s),
        None    => workspace_root.join("stdlib"),
    };
    let sandbox_root: PathBuf = match env::var_os("WORLDSIM_SANDBOX_ROOT") {
        Some(s) => PathBuf::from(s),
        None    => workspace_root.to_path_buf(),
    };

    // Emit each fixture's artifacts into `OUT_DIR/<fixture>/`.
    for f in &fixtures {
        dsl_compiler::build_helper::emit_namespaced(f);

        // Re-parse to discover imported files so edits to stdlib or
        // local-relative imports trigger an incremental rebuild (Option B:
        // one extra parse per fixture; .sim files are <1KB so cost is trivial).
        let sim_path = sims_dir.join(format!("{f}.sim"));
        match dsl_compiler::parse_with_imports(&sim_path, &stdlib_root, &sandbox_root) {
            Ok(program) => {
                for p in &program.imports_resolved {
                    println!("cargo:rerun-if-changed={}", p.display());
                }
            }
            Err(_e) => {
                // emit_namespaced already panicked/surfaced the error above;
                // we just skip the rerun-if-changed lines for this fixture.
            }
        }
    }

    // Synthesise a single include-stub that lib.rs pulls in. One
    // `pub mod <fixture>` per discovered .sim, each wrapping the
    // fixture's compiler-emitted modules. Module-relative paths in
    // the generated runtime_core.rs (`dispatch::KernelCache`,
    // `schedule::SCHEDULE`) resolve correctly because both
    // generated.rs (which declares those modules) and runtime_core.rs
    // are included into the same `pub mod <fixture>` scope.
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let mut stub = String::new();
    stub.push_str(
        "// AUTO-GENERATED by sims/build.rs from assets/sim/*.sim.\n\
         // Do not edit by hand.\n\n",
    );
    for f in &fixtures {
        let has_terrain = out_dir.join(format!("{f}/terrain_gen.rs")).exists();
        stub.push_str(&format!(
            "#[allow(non_snake_case, unused_imports, unused_variables, dead_code, clippy::all)]\n\
             pub mod {f} {{\n\
             \x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/generated.rs\"));\n\
             \x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/runtime_core.rs\"));\n",
        ));
        if has_terrain {
            stub.push_str(&format!(
                "\x20   include!(concat!(env!(\"OUT_DIR\"), \"/{f}/terrain_gen.rs\"));\n",
            ));
        }
        stub.push_str("}\n\n");
    }
    fs::write(out_dir.join("sim_modules.rs"), stub)
        .unwrap_or_else(|e| panic!("write sim_modules.rs: {e}"));
}
