//! Compile-gate tests for the vampire_survivors DSL benchmark fixture.
//! Drives assets/sim/vampire_survivors.sim through
//! parse -> resolve -> lower -> schedule -> emit and asserts emitted
//! kernel shapes. A failing lower IS the gap signal (spec §8 ledger).

use dsl_compiler::cg::emit::EmittedArtifacts;

fn workspace_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(rel)
}

fn compile_sim(path: &std::path::Path) -> Result<EmittedArtifacts, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let program = dsl_compiler::parse(&src).map_err(|e| format!("parse: {e:?}"))?;
    let comp = dsl_ast::resolve::resolve(program).map_err(|e| format!("resolve: {e:?}"))?;
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp)
        .map_err(|e| format!("lower: {e:?}"))?;
    let schedule_result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    dsl_compiler::cg::emit::emit_cg_program(&schedule_result.schedule, &cg)
        .map_err(|e| format!("emit: {e:?}"))
}

#[allow(dead_code)]
fn kernel_body_containing<'a>(art: &'a EmittedArtifacts, needle: &str) -> Option<&'a str> {
    art.wgsl_files
        .iter()
        .find(|(name, _)| name.contains(needle))
        .map(|(_, body)| body.as_str())
}

#[test]
fn vampire_survivors_compiles() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).unwrap_or_else(|e| panic!("vampire_survivors.sim failed at: {e}"));
    assert!(!art.kernel_index.is_empty(), "no kernels emitted");
    eprintln!(
        "[vampire_survivors] {} kernels: {:?}",
        art.kernel_index.len(),
        art.kernel_index,
    );
}

#[test]
fn enemy_chase_emits_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let body = kernel_body_containing(&art, "ChasePlayer")
        .or_else(|| kernel_body_containing(&art, "physics"))
        .unwrap_or_else(|| panic!("no chase kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        body.contains("spatial_grid_offsets") || body.contains("grid_starts"),
        "expected bounded-neighbour walk in ChasePlayer body; got:\n{body}",
    );
}

#[test]
fn player_control_reads_input() {
    // Plan 3: KitePlayer's autonomous flee was replaced by PlayerControl, which
    // moves the player by the @runtime input channel (cfg.config_ctl_move_*).
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let body = kernel_body_containing(&art, "PlayerControl")
        .unwrap_or_else(|| panic!("no PlayerControl kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        body.contains("config_ctl_move_x") && body.contains("config_ctl_move_y"),
        "PlayerControl should read the runtime input channel (cfg.config_ctl_move_*); got:\n{body}",
    );
}

#[test]
fn bolt_fires_and_damage_applies() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let bolt = kernel_body_containing(&art, "BoltFire")
        .unwrap_or_else(|| panic!("no BoltFire kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        bolt.contains("atomicStore(&event_ring") || bolt.contains("atomicAdd(&event_tail"),
        "BoltFire should emit a Damaged event; got:\n{bolt}",
    );
    let apply = kernel_body_containing(&art, "ApplyDamage")
        .unwrap_or_else(|| panic!("no ApplyDamage kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(apply.contains("agent_hp"), "ApplyDamage should write agent_hp; got:\n{apply}");
    assert!(bolt.contains("view_storage"), "BoltFire amount should read the xp view; got:\n{bolt}");
}

#[test]
fn nova_fires_aoe_neighbour_walk() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let nova = kernel_body_containing(&art, "NovaFire")
        .unwrap_or_else(|| panic!("no NovaFire kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        nova.contains("spatial_grid_offsets") || nova.contains("grid_starts"),
        "NovaFire should iterate enemies in radius via neighbour walk; got:\n{nova}",
    );
    assert!(
        nova.contains("atomicStore(&event_ring") || nova.contains("atomicAdd(&event_tail"),
        "NovaFire should emit Damaged per enemy in radius; got:\n{nova}",
    );
}

#[test]
fn nova_scales_with_ctl_level() {
    // Plan 3: nova damage now scales off the host-driven ctl nova_level input
    // (replacing the old floor(xp/xp_per_level) auto-ramp).
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let nova = kernel_body_containing(&art, "NovaFire").expect("NovaFire kernel");
    assert!(
        nova.contains("config_ctl_nova_level"),
        "NovaFire amount should scale off the ctl nova_level input; got:\n{nova}",
    );
}

#[test]
fn xp_view_folds_kills() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let xp = kernel_body_containing(&art, "xp")
        .unwrap_or_else(|| panic!("no xp fold kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(xp.contains("view_storage"), "xp fold should write view storage; got:\n{xp}");
}

#[test]
fn new_weapons_gated_by_ctl_level() {
    // Plan 3: the auto-ChooseUpgrade probe was replaced by host-driven upgrades.
    // Two new weapons (garlic aura, whip sweep) are gated by their ctl level.
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    let garlic = kernel_body_containing(&art, "GarlicAura")
        .unwrap_or_else(|| panic!("no GarlicAura kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        garlic.contains("config_ctl_garlic_level"),
        "GarlicAura should be gated by the ctl garlic_level input; got:\n{garlic}",
    );
    let whip = kernel_body_containing(&art, "WhipSweep")
        .unwrap_or_else(|| panic!("no WhipSweep kernel; have {:?}", art.wgsl_files.keys().collect::<Vec<_>>()));
    assert!(
        whip.contains("config_ctl_whip_level"),
        "WhipSweep should be gated by the ctl whip_level input; got:\n{whip}",
    );
}

#[test]
fn spawn_verbs_emit_summon_chronicle() {
    let path = workspace_path("assets/sim/vampire_survivors.sim");
    let art = compile_sim(&path).expect("compiles");
    // The apply_ability dispatcher must emit an EffectSummonApplied chronicle
    // (kind 62) into the event ring. Kind tag appears as `62u` in the
    // atomicStore tag write.
    let has_summon_emit = art.wgsl_files.values().any(|body| {
        body.contains("62u") && (body.contains("atomicStore(&event_ring") || body.contains("atomicAdd(&event_tail"))
    });
    assert!(
        has_summon_emit,
        "expected an EffectSummonApplied (kind 62) chronicle emit; kernels: {:?}",
        art.kernel_index,
    );
}
