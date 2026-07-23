//! S12 / DELIVERABLE 2 — MAKING A MID-CAMPAIGN SAVE ACTUALLY RESUMABLE.
//!
//! S9's honest list, item 5: *"Fixture state is not serializable. The
//! save/load this slice proves is HOST-side (`Campaign`). `GeneratedRuntime`
//! has no snapshot/restore, so a real game save would today lose the colony's
//! live positions/inventories/beliefs."* This module adds the missing half.
//!
//! **Why it is written by hand and not taken from the engine.** The engine
//! DOES ship a snapshot facility (`crates/engine/src/snapshot/format.rs`,
//! `save_snapshot`/`load_snapshot`, plus safetensors trajectory recording in
//! `trajectory.rs`) — but it serializes `engine::state::SimState`, the CPU-side
//! wolf-era SoA, and a compiled-`.sim` `GeneratedRuntime` does not own one:
//! its state lives entirely in wgpu buffers it declares itself, one per
//! field/view. There is no reflection over those fields, so the buffer TABLE
//! below is generated from the emitted `runtime_core.rs` and checked in. That
//! is the same shape the rest of this crate already has (the bridge names
//! dozens of `agent_*_buf` fields by hand); a general facility belongs in the
//! compiler's emit — it would be ~20 lines of `build_helper` to emit
//! `fn state_buffers(&self) -> &[(&str, &wgpu::Buffer)]` for every fixture,
//! and that is the right follow-up. Flagged, not faked.
//!
//! **WHAT IS SAVED**: every `agent_*_buf` (all 42 custom fields plus the
//! built-in SoA columns — positions, hp, inventories, needs, mood, claims,
//! raid state) and every `view_storage_*_buf` (which is where the PAIR
//! BELIEFS live: `standing_brawl`, `standing_tended`, `grudge`, `repute` —
//! the minds layer — alongside the tallies), plus the runtime's tick counter
//! and the whole host-side `Bridge` (Campaign, the roster->slot map, the free
//! pool, the staged raid, the counters, the log).
//!
//! **WHAT IS NOT SAVED, and the experiment that says whether it matters**:
//! the GPU event ring (`GeneratedRuntime::event_ring` — `engine::gpu::EventRing`
//! keeps its buffers PRIVATE, so a bridge crate cannot reach them without an
//! engine change), and the per-tick scratch (spatial hash, mask bitmaps,
//! radix histograms, indirect args) which every tick rebuilds from scratch
//! anyway. The ring is the interesting one: it is genuinely cross-tick state
//! (the delayed lossy fold window reads the PRIOR tick's segment — S4 finding
//! 3). The save point this module is built for is a DAWN BOUNDARY, which is
//! where a real game saves, and the proof is empirical, not an argument:
//! `webband_play --headless N --save-at D` writes a save, a FRESH PROCESS
//! resumes it with `--resume`, and the continued run's state digest is
//! compared against an uninterrupted run's. Equal digests mean the omission
//! does not reach sim state; unequal digests would name the gap exactly.

use std::io::{Read, Write};
use std::path::Path;

use crate::{Bridge, GeneratedRuntime, StagedRaid, World};

/// EVERY named `wgpu::Buffer` the generated runtime owns, by name — the
/// agent SoA columns, the materialized-view storage (which is where the pair
/// beliefs live), and the machinery buffers (cfg blocks, mask bitmaps, the
/// spatial hash, the radix scratch, `prev_event_tail`). The machinery is
/// mostly per-tick scratch and costs nothing to carry; carrying it removes a
/// whole class of "is THAT what diverged?" from the proof.
///
/// Generated from the emitted `runtime_core.rs` (`grep -o "pub agent_.*_buf"`
/// / `"pub view_storage_.*_buf"`), so it is exhaustive over the two families
/// as of `webband_colony`'s current shape. A rename or a new field makes this
/// list stale in exactly one detectable way: [`load_fixture`] reports any
/// saved name it cannot place and any live buffer the save did not carry, and
/// `save_fixture` records the count — a silent partial restore is not
/// possible.
pub fn state_buffers(s: &GeneratedRuntime) -> Vec<(&'static str, wgpu::Buffer)> {
    macro_rules! table {
        ($($f:ident),* $(,)?) => { vec![$((stringify!($f), s.$f.clone())),*] };
    }
    table![
        // S5c: `agent_ability_id_buf` REPLACED `agent_atk_buf` when combat
        // moved onto the ported `.ability` programs — a raider body no
        // longer carries its damage as a stat, it carries the registry
        // slot of the program it swings. This table is hand-maintained
        // (see the module docs); a fixture field change lands here.
        agent_ability_id_buf,
        agent_ability_power_buf,
        agent_alive_buf,
        agent_altruism_buf,
        agent_ambition_buf,
        agent_armor_buf,
        agent_ate_meal_tick_buf,
        agent_atk_cd_buf,
        agent_attack_damage_buf,
        agent_attack_range_buf,
        agent_built_buf,
        agent_burnt_buf,
        agent_busy_started_at_tick_buf,
        agent_busy_target_pos_buf,
        agent_busy_target_slot_buf,
        agent_busy_until_tick_buf,
        agent_busy_with_ability_id_buf,
        agent_carry_hide_buf,
        agent_carry_kind_buf,
        agent_carry_n_buf,
        agent_claim_until_buf,
        agent_claimed_buf,
        agent_claimed_job_buf,
        agent_cooldown_next_ready_tick_buf,
        agent_creature_type_buf,
        agent_curiosity_buf,
        agent_damage_taken_mult_expires_at_tick_buf,
        agent_damage_taken_mult_q8_buf,
        agent_defenders_buf,
        agent_demand_plank_buf,
        agent_demand_timber_buf,
        agent_dir_live_buf,
        agent_dir_target_pos_buf,
        agent_directive_kind_buf,
        agent_directive_pos_buf,
        agent_directive_target_buf,
        agent_disguise_expires_at_tick_buf,
        agent_disguise_fake_type_buf,
        agent_downed_buf,
        agent_engaged_with_buf,
        agent_esteem_buf,
        agent_fear_expires_at_tick_buf,
        agent_grid_id_buf,
        agent_growth_buf,
        agent_hp_buf,
        agent_hunger_buf,
        agent_inv_berries_buf,
        agent_inv_grain_buf,
        agent_inv_hide_buf,
        agent_inv_meal_buf,
        agent_inv_plank_buf,
        agent_inv_timber_buf,
        agent_inv_venison_buf,
        agent_job_site_buf,
        agent_level_buf,
        agent_lifesteal_expires_at_tick_buf,
        agent_lifesteal_frac_q8_buf,
        agent_local_pos_buf,
        agent_magic_resist_buf,
        agent_mana_buf,
        agent_max_hp_buf,
        agent_max_mana_buf,
        agent_mood_buf,
        agent_move_speed_buf,
        agent_move_speed_mult_buf,
        agent_move_target_buf,
        agent_movement_mode_buf,
        agent_musters_at_buf,
        agent_need_cheer_buf,
        agent_need_comfort_buf,
        agent_need_food_buf,
        agent_need_plank_buf,
        agent_need_rest_buf,
        agent_need_timber_buf,
        agent_plundered_at_buf,
        agent_pos_buf,
        agent_pri_build_buf,
        agent_pri_chop_buf,
        agent_pri_cook_buf,
        agent_pri_craft_buf,
        agent_pri_forage_buf,
        agent_pri_grow_buf,
        agent_pri_haul_buf,
        agent_pri_hunt_buf,
        agent_prowess_tick_buf,
        agent_purpose_buf,
        agent_purse_buf,
        agent_raid_active_buf,
        agent_raid_on_buf,
        agent_raid_total_buf,
        agent_regrow_at_buf,
        agent_rest_timer_buf,
        agent_risk_tolerance_buf,
        agent_root_expires_at_tick_buf,
        agent_safety_buf,
        agent_scratch_packed_buf,
        agent_shelter_buf,
        agent_shield_hp_buf,
        agent_silence_expires_at_tick_buf,
        agent_slow_expires_at_tick_buf,
        agent_slow_factor_q8_buf,
        agent_social_buf,
        agent_social_drive_buf,
        agent_sown_buf,
        agent_spawn_tick_buf,
        agent_stagger_buf,
        agent_standing_sum_buf,
        agent_starving_days_buf,
        agent_strike_cd_until_buf,
        agent_stun_expires_at_tick_buf,
        agent_taunt_expires_at_tick_buf,
        agent_tended_at_buf,
        agent_thirst_buf,
        agent_thought_sum_buf,
        agent_trader_active_buf,
        agent_travel_dest_x_buf,
        agent_travel_dest_y_buf,
        agent_travel_dest_z_buf,
        agent_vel_buf,
        agent_warned_buf,
        agent_work_done_buf,
        agent_work_left_buf,
        view_storage_brawls_total_anchor_buf,
        view_storage_brawls_total_ids_buf,
        view_storage_brawls_total_primary_buf,
        view_storage_count_ate_raw_anchor_buf,
        view_storage_count_ate_raw_ids_buf,
        view_storage_count_ate_raw_primary_buf,
        view_storage_count_downed_anchor_buf,
        view_storage_count_downed_ids_buf,
        view_storage_count_downed_primary_buf,
        view_storage_count_plunder_anchor_buf,
        view_storage_count_plunder_ids_buf,
        view_storage_count_plunder_primary_buf,
        view_storage_count_prowess_seen_anchor_buf,
        view_storage_count_prowess_seen_ids_buf,
        view_storage_count_prowess_seen_primary_buf,
        view_storage_count_raid_warning_anchor_buf,
        view_storage_count_raid_warning_ids_buf,
        view_storage_count_raid_warning_primary_buf,
        view_storage_count_raider_kills_anchor_buf,
        view_storage_count_raider_kills_ids_buf,
        view_storage_count_raider_kills_primary_buf,
        view_storage_count_slept_rough_anchor_buf,
        view_storage_count_slept_rough_ids_buf,
        view_storage_count_slept_rough_primary_buf,
        view_storage_count_starving_anchor_buf,
        view_storage_count_starving_ids_buf,
        view_storage_count_starving_primary_buf,
        view_storage_count_supper_anchor_buf,
        view_storage_count_supper_ids_buf,
        view_storage_count_supper_primary_buf,
        view_storage_count_tended_anchor_buf,
        view_storage_count_tended_ids_buf,
        view_storage_count_tended_primary_buf,
        view_storage_grudge_anchor_buf,
        view_storage_grudge_ids_buf,
        view_storage_grudge_load_anchor_buf,
        view_storage_grudge_load_ids_buf,
        view_storage_grudge_load_primary_buf,
        view_storage_grudge_primary_buf,
        view_storage_repute_anchor_buf,
        view_storage_repute_ids_buf,
        view_storage_repute_primary_buf,
        view_storage_standing_brawl_anchor_buf,
        view_storage_standing_brawl_ids_buf,
        view_storage_standing_brawl_primary_buf,
        view_storage_standing_tended_anchor_buf,
        view_storage_standing_tended_ids_buf,
        view_storage_standing_tended_primary_buf,
        view_storage_tally_build_anchor_buf,
        view_storage_tally_build_ids_buf,
        view_storage_tally_build_primary_buf,
        view_storage_tally_chop_anchor_buf,
        view_storage_tally_chop_ids_buf,
        view_storage_tally_chop_primary_buf,
        view_storage_tally_cook_anchor_buf,
        view_storage_tally_cook_ids_buf,
        view_storage_tally_cook_primary_buf,
        view_storage_tally_craft_anchor_buf,
        view_storage_tally_craft_ids_buf,
        view_storage_tally_craft_primary_buf,
        view_storage_tally_eat_anchor_buf,
        view_storage_tally_eat_ids_buf,
        view_storage_tally_eat_primary_buf,
        view_storage_tally_forage_anchor_buf,
        view_storage_tally_forage_ids_buf,
        view_storage_tally_forage_primary_buf,
        view_storage_tally_harvest_anchor_buf,
        view_storage_tally_harvest_ids_buf,
        view_storage_tally_harvest_primary_buf,
        view_storage_tally_haul_bench_anchor_buf,
        view_storage_tally_haul_bench_ids_buf,
        view_storage_tally_haul_bench_primary_buf,
        view_storage_tally_haul_cache_anchor_buf,
        view_storage_tally_haul_cache_ids_buf,
        view_storage_tally_haul_cache_primary_buf,
        view_storage_tally_haul_hearth_anchor_buf,
        view_storage_tally_haul_hearth_ids_buf,
        view_storage_tally_haul_hearth_primary_buf,
        view_storage_tally_haul_store_anchor_buf,
        view_storage_tally_haul_store_ids_buf,
        view_storage_tally_haul_store_primary_buf,
        view_storage_tally_hunt_anchor_buf,
        view_storage_tally_hunt_ids_buf,
        view_storage_tally_hunt_primary_buf,
        view_storage_tally_sow_anchor_buf,
        view_storage_tally_sow_ids_buf,
        view_storage_tally_sow_primary_buf,
        view_storage_tally_strikes_in_anchor_buf,
        view_storage_tally_strikes_in_ids_buf,
        view_storage_tally_strikes_in_primary_buf,
        view_storage_tally_strikes_out_anchor_buf,
        view_storage_tally_strikes_out_ids_buf,
        view_storage_tally_strikes_out_primary_buf,
        view_storage_tally_tend_anchor_buf,
        view_storage_tally_tend_ids_buf,
        view_storage_tally_tend_primary_buf,
        view_storage_thought_ate_raw_anchor_buf,
        view_storage_thought_ate_raw_ids_buf,
        view_storage_thought_ate_raw_primary_buf,
        view_storage_thought_came_home_to_ashes_anchor_buf,
        view_storage_thought_came_home_to_ashes_ids_buf,
        view_storage_thought_came_home_to_ashes_primary_buf,
        view_storage_thought_defeat_anchor_buf,
        view_storage_thought_defeat_ids_buf,
        view_storage_thought_defeat_primary_buf,
        view_storage_thought_festival_anchor_buf,
        view_storage_thought_festival_ids_buf,
        view_storage_thought_festival_primary_buf,
        view_storage_thought_goal_served_anchor_buf,
        view_storage_thought_goal_served_ids_buf,
        view_storage_thought_goal_served_primary_buf,
        view_storage_thought_home_refused_anchor_buf,
        view_storage_thought_home_refused_ids_buf,
        view_storage_thought_home_refused_primary_buf,
        view_storage_thought_home_served_anchor_buf,
        view_storage_thought_home_served_ids_buf,
        view_storage_thought_home_served_primary_buf,
        view_storage_thought_hungry_road_anchor_buf,
        view_storage_thought_hungry_road_ids_buf,
        view_storage_thought_hungry_road_primary_buf,
        view_storage_thought_slept_rough_anchor_buf,
        view_storage_thought_slept_rough_ids_buf,
        view_storage_thought_slept_rough_primary_buf,
        view_storage_thought_starving_anchor_buf,
        view_storage_thought_starving_ids_buf,
        view_storage_thought_starving_primary_buf,
        view_storage_thought_victory_anchor_buf,
        view_storage_thought_victory_ids_buf,
        view_storage_thought_victory_primary_buf,
        alive_bitmap_buf,
        cfg_alive_pack_buf,
        cfg_decay_grudge_buf,
        cfg_decay_grudge_load_buf,
        cfg_decay_standing_brawl_buf,
        cfg_decay_standing_tended_buf,
        cfg_decay_thought_ate_raw_buf,
        cfg_decay_thought_came_home_to_ashes_buf,
        cfg_decay_thought_defeat_buf,
        cfg_decay_thought_festival_buf,
        cfg_decay_thought_goal_served_buf,
        cfg_decay_thought_home_refused_buf,
        cfg_decay_thought_home_served_buf,
        cfg_decay_thought_hungry_road_buf,
        cfg_decay_thought_slept_rough_buf,
        cfg_decay_thought_starving_buf,
        cfg_decay_thought_victory_buf,
        cfg_fold_brawls_total_buf,
        cfg_fold_count_ate_raw_buf,
        cfg_fold_count_downed_buf,
        cfg_fold_count_plunder_buf,
        cfg_fold_count_prowess_seen_buf,
        cfg_fold_count_raid_warning_buf,
        cfg_fold_count_raider_kills_buf,
        cfg_fold_count_slept_rough_buf,
        cfg_fold_count_starving_buf,
        cfg_fold_count_supper_buf,
        cfg_fold_count_tended_buf,
        cfg_fold_grudge_buf,
        cfg_fold_grudge_load_buf,
        cfg_fold_repute_buf,
        cfg_fold_standing_brawl_buf,
        cfg_fold_standing_tended_buf,
        cfg_fold_tally_build_buf,
        cfg_fold_tally_chop_buf,
        cfg_fold_tally_cook_buf,
        cfg_fold_tally_craft_buf,
        cfg_fold_tally_eat_buf,
        cfg_fold_tally_forage_buf,
        cfg_fold_tally_harvest_buf,
        cfg_fold_tally_haul_bench_buf,
        cfg_fold_tally_haul_cache_buf,
        cfg_fold_tally_haul_hearth_buf,
        cfg_fold_tally_haul_store_buf,
        cfg_fold_tally_hunt_buf,
        cfg_fold_tally_sow_buf,
        cfg_fold_tally_strikes_in_buf,
        cfg_fold_tally_strikes_out_buf,
        cfg_fold_tally_tend_buf,
        cfg_fold_thought_ate_raw_buf,
        cfg_fold_thought_came_home_to_ashes_buf,
        cfg_fold_thought_defeat_buf,
        cfg_fold_thought_festival_buf,
        cfg_fold_thought_goal_served_buf,
        cfg_fold_thought_home_refused_buf,
        cfg_fold_thought_home_served_buf,
        cfg_fold_thought_hungry_road_buf,
        cfg_fold_thought_slept_rough_buf,
        cfg_fold_thought_starving_buf,
        cfg_fold_thought_victory_buf,
        cfg_fused_pack_agents_buf,
        cfg_kick_snapshot_buf,
        cfg_merge_repute_supper_tale_max_buf,
        cfg_scoring_buf,
        cfg_seed_indirect_0_buf,
        cfg_spatial_build_hash_count_buf,
        cfg_spatial_build_hash_scan_add_buf,
        cfg_spatial_build_hash_scan_carry_buf,
        cfg_spatial_build_hash_scan_local_buf,
        cfg_spatial_build_hash_scatter_buf,
        cfg_upload_sim_cfg_buf,
        event_ring_sort_scratch_buf,
        indirect_args_0_buf,
        mask_0_bitmap_buf,
        mask_10_bitmap_buf,
        mask_11_bitmap_buf,
        mask_12_bitmap_buf,
        mask_13_bitmap_buf,
        mask_14_bitmap_buf,
        mask_15_bitmap_buf,
        mask_16_bitmap_buf,
        mask_17_bitmap_buf,
        mask_18_bitmap_buf,
        mask_19_bitmap_buf,
        mask_1_bitmap_buf,
        mask_20_bitmap_buf,
        mask_21_bitmap_buf,
        mask_22_bitmap_buf,
        mask_23_bitmap_buf,
        mask_24_bitmap_buf,
        mask_25_bitmap_buf,
        mask_26_bitmap_buf,
        mask_27_bitmap_buf,
        mask_28_bitmap_buf,
        mask_29_bitmap_buf,
        mask_2_bitmap_buf,
        mask_30_bitmap_buf,
        mask_31_bitmap_buf,
        mask_32_bitmap_buf,
        mask_33_bitmap_buf,
        mask_34_bitmap_buf,
        mask_35_bitmap_buf,
        mask_36_bitmap_buf,
        mask_37_bitmap_buf,
        mask_38_bitmap_buf,
        mask_39_bitmap_buf,
        mask_3_bitmap_buf,
        mask_40_bitmap_buf,
        mask_41_bitmap_buf,
        mask_42_bitmap_buf,
        mask_43_bitmap_buf,
        mask_44_bitmap_buf,
        mask_4_bitmap_buf,
        mask_5_bitmap_buf,
        mask_6_bitmap_buf,
        mask_7_bitmap_buf,
        mask_8_bitmap_buf,
        mask_9_bitmap_buf,
        prev_event_tail_buf,
        radix_bucket_offsets_buf,
        radix_histogram_buf,
        scoring_output_buf,
        snapshot_kick_buf,
        sort_cfg_buf,
        spatial_chunk_sums_buf,
        spatial_grid_cells_buf,
        spatial_grid_offsets_buf,
        spatial_grid_starts_buf,
        target_histogram_buf,
        target_offsets_buf,
    ]
}

const MAGIC: &[u8; 8] = b"WBFIX01\n";

fn read_bytes(state: &mut GeneratedRuntime, buf: &wgpu::Buffer) -> Vec<u8> {
    let bytes = buf.size();
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("webband_bridge::persist_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = slice.get_mapped_range().to_vec();
    staging.unmap();
    out
}

/// Write the fixture's GPU state to `path`. Returns (buffers written, bytes).
///
/// Buffers without `COPY_SRC` are SKIPPED AND NAMED in the returned report —
/// this crate refuses to pretend it saved something it could not read.
pub fn save_fixture(state: &mut GeneratedRuntime, path: &Path) -> std::io::Result<FixtureSaveReport> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(MAGIC)?;
    f.write_all(&state.tick.to_le_bytes())?;
    f.write_all(&state.agent_count.to_le_bytes())?;
    f.write_all(&state.seed.to_le_bytes())?;
    let table = state_buffers(state);
    let mut skipped: Vec<&'static str> = Vec::new();
    let mut saved: Vec<(&'static str, Vec<u8>)> = Vec::new();
    for (name, buf) in &table {
        if !buf.usage().contains(wgpu::BufferUsages::COPY_SRC) {
            skipped.push(name);
            continue;
        }
        saved.push((name, read_bytes(state, buf)));
    }
    f.write_all(&(saved.len() as u32).to_le_bytes())?;
    let mut bytes = 0u64;
    for (name, data) in &saved {
        let nb = name.as_bytes();
        f.write_all(&(nb.len() as u32).to_le_bytes())?;
        f.write_all(nb)?;
        f.write_all(&(data.len() as u64).to_le_bytes())?;
        f.write_all(data)?;
        bytes += data.len() as u64;
    }
    f.flush()?;
    Ok(FixtureSaveReport { buffers: saved.len(), bytes, skipped })
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixtureSaveReport {
    pub buffers: usize,
    pub bytes: u64,
    /// Buffers the runtime declares but that cannot be copied out.
    pub skipped: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FixtureLoadReport {
    pub buffers: usize,
    pub tick: u64,
    /// Names in the save with no live buffer (a stale table).
    pub unplaced: Vec<String>,
    /// Live buffers the save did not carry.
    pub missing: Vec<&'static str>,
}

/// Restore a fixture save written by [`save_fixture`] onto a freshly
/// constructed runtime (same seed, same agent cap — asserted).
pub fn load_fixture(state: &mut GeneratedRuntime, path: &Path) -> std::io::Result<FixtureLoadReport> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a webband fixture save",
        ));
    }
    let mut w8 = [0u8; 8];
    let mut w4 = [0u8; 4];
    f.read_exact(&mut w8)?;
    let tick = u64::from_le_bytes(w8);
    f.read_exact(&mut w4)?;
    let agent_count = u32::from_le_bytes(w4);
    f.read_exact(&mut w8)?;
    let seed = u64::from_le_bytes(w8);
    assert_eq!(
        agent_count, state.agent_count,
        "a fixture save only restores onto the same agent cap"
    );
    assert_eq!(seed, state.seed, "a fixture save only restores onto the same fixture seed");
    f.read_exact(&mut w4)?;
    let n = u32::from_le_bytes(w4) as usize;

    let table = state_buffers(state);
    let mut seen: Vec<&'static str> = Vec::new();
    let mut unplaced: Vec<String> = Vec::new();
    for _ in 0..n {
        f.read_exact(&mut w4)?;
        let nl = u32::from_le_bytes(w4) as usize;
        let mut nb = vec![0u8; nl];
        f.read_exact(&mut nb)?;
        let name = String::from_utf8_lossy(&nb).to_string();
        f.read_exact(&mut w8)?;
        let dl = u64::from_le_bytes(w8) as usize;
        let mut data = vec![0u8; dl];
        f.read_exact(&mut data)?;
        match table.iter().find(|(n2, _)| *n2 == name) {
            Some((n2, buf)) => {
                assert_eq!(
                    buf.size() as usize,
                    dl,
                    "buffer {name} changed size between save and load"
                );
                state.gpu.queue.write_buffer(buf, 0, &data);
                seen.push(n2);
            }
            None => unplaced.push(name),
        }
    }
    state.gpu.queue.submit(std::iter::empty());
    state.tick = tick;
    let missing: Vec<&'static str> = table
        .iter()
        .map(|(n2, _)| *n2)
        .filter(|n2| !seen.contains(n2))
        .collect();
    Ok(FixtureLoadReport { buffers: seen.len(), tick, unplaced, missing })
}

// ---------------------------------------------------------------------------
// The HOST half of a save: everything the bridge holds that is not on the GPU.

/// The bridge's own state, serialized beside the fixture image. `Campaign`
/// already round-trips exactly (S7b/S9 pin it, `float_roundtrip` and all);
/// everything else here is the seating and bookkeeping a resumed campaign
/// needs to keep driving the same bodies.
///
/// `World` is SAVED, not re-derived: `read_world` filters on `alive`, and a
/// resumed colony has deactivated pool slots, so re-deriving would silently
/// return a shorter colonist list and reseat the roster on the wrong bodies.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BridgeSave {
    pub campaign: webband_app::campaign::Campaign,
    pub world: World,
    pub slot_map: Vec<(String, usize)>,
    pub free_slots: Vec<usize>,
    pub tick: u32,
    pub purse_shadow: f64,
    pub auto_trade: bool,
    pub log: Vec<String>,
    pub hungry_member_dawns: u64,
    pub member_dawns: u64,
    pub raids_staged: u64,
    pub raids_won: u64,
    pub raids_lost: u64,
    pub windfalls: u64,
    pub caravans: u64,
    pub trade_gold: i64,
    pub meals_bought: i64,
    pub joins: u64,
    pub departures: Vec<(i64, String)>,
    pub prev_starving: Vec<f32>,
    pub event_kinds: Vec<String>,
    pub raid_gold: i64,
    pub raid_log: Vec<crate::RaidRecord>,
    pub last_snap: Option<webband_app::campaign::ColonySnapshot>,
    /// A raid staged but not yet settled must survive the save, or the
    /// resumed campaign would leave a live cohort on the board with nobody
    /// to resolve it (the spine's `campaign.raid.is_some() == staged.is_some()`
    /// coherence pin, applied to persistence).
    pub staged: Option<(webband_app::raids::ActiveRaid, Vec<usize>, u32)>,

    // -- S13: the guild layer's own bookkeeping. All `serde(default)`, so a
    // pre-S13 save (and every apolitical one, where they stay empty) loads
    // unchanged. `away` is the load-bearing one: it maps a member id to the
    // body the dispatch deactivated, and without it a resumed campaign would
    // never wake anyone up when their company came home.
    #[serde(default)]
    pub away: Vec<(String, usize)>,
    #[serde(default)]
    pub petitions_opened: u64,
    #[serde(default)]
    pub petitions_answered: u64,
    #[serde(default)]
    pub petitions_lapsed: u64,
    #[serde(default)]
    pub petition_log: Vec<(i64, String, String, String)>,
    #[serde(default)]
    pub stages: Vec<String>,
    #[serde(default)]
    pub achieved: Option<String>,
    #[serde(default)]
    pub band_notices: Vec<(i64, String)>,
    #[serde(default)]
    pub refusals: Vec<String>,
}

impl Bridge {
    /// Write BOTH halves of a save: `<dir>/campaign.json` (the host) and
    /// `<dir>/fixture.bin` (the GPU image).
    pub fn save_all(&mut self, dir: &Path) -> std::io::Result<FixtureSaveReport> {
        std::fs::create_dir_all(dir)?;
        let host = BridgeSave {
            campaign: self.campaign.clone(),
            world: self.w.clone(),
            slot_map: self.slot_map.clone(),
            free_slots: self.free_slots.iter().copied().collect(),
            tick: self.tick,
            purse_shadow: self.purse_shadow,
            auto_trade: self.auto_trade,
            log: self.log.clone(),
            hungry_member_dawns: self.hungry_member_dawns,
            member_dawns: self.member_dawns,
            raids_staged: self.raids_staged,
            raids_won: self.raids_won,
            raids_lost: self.raids_lost,
            windfalls: self.windfalls,
            caravans: self.caravans,
            trade_gold: self.trade_gold,
            meals_bought: self.meals_bought,
            joins: self.joins,
            departures: self.departures.clone(),
            prev_starving: self.prev_starving.clone(),
            event_kinds: self.event_kinds.clone(),
            raid_gold: self.raid_gold,
            raid_log: self.raid_log.clone(),
            last_snap: self.last_snap.clone(),
            staged: self
                .staged
                .as_ref()
                .map(|s| (s.raid.clone(), s.cohort.clone(), s.muster_tick)),
            away: self.away.clone(),
            petitions_opened: self.petitions_opened,
            petitions_answered: self.petitions_answered,
            petitions_lapsed: self.petitions_lapsed,
            petition_log: self.petition_log.clone(),
            stages: self.stages.clone(),
            achieved: self.achieved.clone(),
            band_notices: self.band_notices.clone(),
            refusals: self.refusals.clone(),
        };
        let json = serde_json::to_string(&host)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(dir.join("campaign.json"), json)?;
        save_fixture(&mut self.state, &dir.join("fixture.bin"))
    }

    /// Rebuild a bridge from a `save_all` directory: construct a fresh
    /// runtime (same seed/cap), stamp the saved GPU image onto it, and
    /// restore the host half.
    pub fn load_all(dir: &Path) -> std::io::Result<(Bridge, FixtureLoadReport)> {
        let json = std::fs::read_to_string(dir.join("campaign.json"))?;
        let host: BridgeSave = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut state = GeneratedRuntime::try_new(crate::SEED, crate::AGENTS).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "no wgpu adapter")
        })?;
        let report = load_fixture(&mut state, &dir.join("fixture.bin"))?;
        let b = Bridge {
            state,
            w: host.world,
            campaign: host.campaign,
            slot_map: host.slot_map,
            free_slots: host.free_slots.into(),
            tick: host.tick,
            staged: host.staged.map(|(raid, cohort, muster_tick)| StagedRaid {
                raid,
                cohort,
                muster_tick,
            }),
            purse_shadow: host.purse_shadow,
            auto_trade: host.auto_trade,
            log: host.log,
            hungry_member_dawns: host.hungry_member_dawns,
            member_dawns: host.member_dawns,
            raids_staged: host.raids_staged,
            raids_won: host.raids_won,
            raids_lost: host.raids_lost,
            windfalls: host.windfalls,
            caravans: host.caravans,
            trade_gold: host.trade_gold,
            meals_bought: host.meals_bought,
            joins: host.joins,
            departures: host.departures,
            prev_starving: host.prev_starving,
            event_kinds: host.event_kinds,
            raid_gold: host.raid_gold,
            raid_log: host.raid_log,
            last_snap: host.last_snap,
            away: host.away,
            petitions_opened: host.petitions_opened,
            petitions_answered: host.petitions_answered,
            petitions_lapsed: host.petitions_lapsed,
            petition_log: host.petition_log,
            stages: host.stages,
            achieved: host.achieved,
            band_notices: host.band_notices,
            refusals: host.refusals,
        };
        Ok((b, report))
    }
}
