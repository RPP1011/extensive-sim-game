//! Regression fence on SimCfg's struct layout. Any drift between the
//! Rust Pod layout and the WGSL struct layout silently corrupts
//! GPU-side reads; this test asserts field offsets match.

#![cfg(feature = "gpu")]

use engine_gpu::sim_cfg::SimCfg;
use memoffset::offset_of;

#[test]
fn sim_cfg_field_offsets_are_stable() {
    assert_eq!(std::mem::size_of::<SimCfg>(), 64, "SimCfg must be 64 bytes");
    assert_eq!(offset_of!(SimCfg, tick), 0);
    assert_eq!(offset_of!(SimCfg, world_seed_lo), 4);
    assert_eq!(offset_of!(SimCfg, world_seed_hi), 8);
    assert_eq!(offset_of!(SimCfg, _pad0), 12);
    assert_eq!(offset_of!(SimCfg, engagement_range), 16);
    assert_eq!(offset_of!(SimCfg, attack_damage), 20);
    assert_eq!(offset_of!(SimCfg, attack_range), 24);
    assert_eq!(offset_of!(SimCfg, move_speed), 28);
    assert_eq!(offset_of!(SimCfg, move_speed_mult), 32);
    assert_eq!(offset_of!(SimCfg, kin_radius), 36);
    assert_eq!(offset_of!(SimCfg, cascade_max_iterations), 40);
    assert_eq!(offset_of!(SimCfg, rules_registry_generation), 44);
    assert_eq!(offset_of!(SimCfg, abilities_registry_generation), 48);
    // _reserved: [u32; 3] at bytes 52..=63; size_of ends at 64.
}

#[test]
fn sim_cfg_is_pod_zeroable() {
    let _: SimCfg = bytemuck::Zeroable::zeroed();
}
