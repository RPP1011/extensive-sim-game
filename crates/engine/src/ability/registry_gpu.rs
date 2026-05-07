//! GPU upload helper for `PackedAbilityRegistry` (#132B).
//!
//! Each SoA column in `PackedAbilityRegistry` becomes one
//! `wgpu::Buffer`. The registry is immutable post-build, so all
//! buffers are uploaded once at runtime startup and never written
//! again from the CPU side; shaders bind them as `array<T>` storage
//! buffers (read-only).
//!
//! # Type widening
//!
//! WGSL doesn't natively support `u8` / `u16` storage element types
//! ergonomically — addressing individual bytes requires bit-twiddling
//! across `u32` words. To keep shader-side reads simple, the upload
//! widens narrow columns:
//!
//!   * `Vec<u8>`  → uploaded as `Vec<u32>` (zero-extended)
//!   * `Vec<u16>` → uploaded as `Vec<u32>` (zero-extended)
//!
//! The widening costs 3-4x the byte size of the original column, but
//! the registry is small (one row per ability × stride 6 effects), so
//! the absolute footprint stays trivial (a 64-ability registry's
//! widened columns total well under 1 KB). A future tightening could
//! pack these into native u8/u16 reads via `let byte = (word >> shift)
//! & mask`; this MVP prioritizes readable WGSL kernels over byte
//! frugality.
//!
//! # Binding layout
//!
//! Callers wire each field into a BindGroup at the binding indices
//! their kernel expects. This module deliberately does NOT prescribe a
//! specific BGL — the WGSL emitter (#132C) is the natural owner of the
//! binding-index contract.

use crate::ability::PackedAbilityRegistry;
use crate::gpu::GpuContext;
use bytemuck;
use wgpu::util::DeviceExt;

/// All `wgpu::Buffer` instances backing one `PackedAbilityRegistry`.
/// Field names mirror `PackedAbilityRegistry`'s columns 1:1; widened
/// `u8` / `u16` columns become u32 storage.
///
/// `n_abilities` is duplicated from the source registry so callers
/// can compute strides without holding a reference back to the CPU
/// `PackedAbilityRegistry` (the registry can be dropped after upload).
pub struct PackedAbilityRegistryGpu {
    pub n_abilities: u32,

    // -- Per-ability scalar columns (one entry per ability). --
    pub hints:          wgpu::Buffer,
    pub cooldown_ticks: wgpu::Buffer,
    pub range:          wgpu::Buffer,
    pub gate_flags:     wgpu::Buffer,
    pub delivery_kind:  wgpu::Buffer,

    // -- Effect rows (stride = MAX_EFFECTS_PER_PROGRAM). --
    pub effect_kinds:     wgpu::Buffer,
    pub effect_payload_a: wgpu::Buffer,
    pub effect_payload_b: wgpu::Buffer,

    // -- Tag rows (stride = NUM_ABILITY_TAGS). --
    pub tag_values: wgpu::Buffer,

    // -- Per-effect modifier columns (stride = MAX_EFFECTS_PER_PROGRAM). --
    /// u8 widened to u32 (see module docs).
    pub stackings: wgpu::Buffer,
    /// u16 widened to u32.
    pub chances: wgpu::Buffer,
    /// u8 widened to u32.
    pub lifetime_kinds: wgpu::Buffer,
    pub lifetime_payloads: wgpu::Buffer,
    /// u8 widened to u32.
    pub area_kinds: wgpu::Buffer,
    pub area_args: wgpu::Buffer,
    /// u8 widened to u32.
    pub scaling_stat_refs: wgpu::Buffer,
    pub scaling_percents: wgpu::Buffer,

    // -- Nested-effect rows (stride = MAX_EFFECTS_PER_PROGRAM ×
    //    MAX_NESTED_PER_EFFECT). Wave 1.5#9 — see `PackedAbilityRegistry`.
    pub nested_effect_kinds:     wgpu::Buffer,
    pub nested_effect_payload_a: wgpu::Buffer,
    pub nested_effect_payload_b: wgpu::Buffer,

    // -- When-predicate rows (stride = MAX_EFFECTS_PER_PROGRAM).
    //    Wave 1.5#7 GPU eval — see `PackedAbilityRegistry`.
    /// u8 widened to u32. Sentinel `WHEN_PRED_NONE_SENTINEL` (0xFF) →
    /// no predicate; dispatcher fires unconditionally.
    pub when_pred_binder:  wgpu::Buffer,
    /// u8 widened to u32 (`ScalingStatRef` discriminant 0..=7).
    pub when_pred_field:   wgpu::Buffer,
    /// u8 widened to u32 (`EffectPredicateOp` discriminant 0..=5).
    pub when_pred_op:      wgpu::Buffer,
    pub when_pred_literal: wgpu::Buffer,
}

impl PackedAbilityRegistryGpu {
    /// Upload a `PackedAbilityRegistry` to the GPU as one buffer per
    /// SoA column. All buffers carry `STORAGE` usage (read-only access
    /// from shader); the registry never mutates after construction so
    /// `COPY_DST` is intentionally omitted.
    ///
    /// `label` is a per-runtime prefix attached to each buffer's debug
    /// label (e.g. `"duel_abilities"`). Helps wgpu's frame capture +
    /// device-loss diagnostics distinguish multiple registries when
    /// more than one runtime shares a process.
    pub fn upload(packed: &PackedAbilityRegistry, ctx: &GpuContext, label: &str) -> Self {
        // Helper closures bind `ctx.device` + `label` so each upload
        // line is one short call.
        let mk_u32 = |suffix: &str, data: &[u32]| -> wgpu::Buffer {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{label}::ability_registry::{suffix}")),
                contents: bytemuck::cast_slice(data),
                usage:    wgpu::BufferUsages::STORAGE,
            })
        };
        let mk_f32 = |suffix: &str, data: &[f32]| -> wgpu::Buffer {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("{label}::ability_registry::{suffix}")),
                contents: bytemuck::cast_slice(data),
                usage:    wgpu::BufferUsages::STORAGE,
            })
        };
        // Widen u8 → u32.
        let widen_u8 = |src: &[u8]| -> Vec<u32> {
            src.iter().map(|&b| b as u32).collect()
        };
        // Widen u16 → u32.
        let widen_u16 = |src: &[u16]| -> Vec<u32> {
            src.iter().map(|&w| w as u32).collect()
        };

        Self {
            n_abilities: packed.n_abilities as u32,

            hints:          mk_u32("hints",          &packed.hints),
            cooldown_ticks: mk_u32("cooldown_ticks", &packed.cooldown_ticks),
            range:          mk_f32("range",          &packed.range),
            gate_flags:     mk_u32("gate_flags",     &packed.gate_flags),
            delivery_kind:  mk_u32("delivery_kind",  &packed.delivery_kind),

            effect_kinds:     mk_u32("effect_kinds",     &packed.effect_kinds),
            effect_payload_a: mk_u32("effect_payload_a", &packed.effect_payload_a),
            effect_payload_b: mk_u32("effect_payload_b", &packed.effect_payload_b),

            tag_values: mk_f32("tag_values", &packed.tag_values),

            stackings:         mk_u32("stackings",         &widen_u8(&packed.stackings)),
            chances:           mk_u32("chances",           &widen_u16(&packed.chances)),
            lifetime_kinds:    mk_u32("lifetime_kinds",    &widen_u8(&packed.lifetime_kinds)),
            lifetime_payloads: mk_f32("lifetime_payloads", &packed.lifetime_payloads),
            area_kinds:        mk_u32("area_kinds",        &widen_u8(&packed.area_kinds)),
            area_args:         mk_f32("area_args",         &packed.area_args),
            scaling_stat_refs: mk_u32("scaling_stat_refs", &widen_u8(&packed.scaling_stat_refs)),
            scaling_percents:  mk_f32("scaling_percents",  &packed.scaling_percents),

            nested_effect_kinds:     mk_u32("nested_effect_kinds",     &packed.nested_effect_kinds),
            nested_effect_payload_a: mk_u32("nested_effect_payload_a", &packed.nested_effect_payload_a),
            nested_effect_payload_b: mk_u32("nested_effect_payload_b", &packed.nested_effect_payload_b),

            when_pred_binder:  mk_u32("when_pred_binder",  &widen_u8(&packed.when_pred_binder)),
            when_pred_field:   mk_u32("when_pred_field",   &widen_u8(&packed.when_pred_field)),
            when_pred_op:      mk_u32("when_pred_op",      &widen_u8(&packed.when_pred_op)),
            when_pred_literal: mk_f32("when_pred_literal", &packed.when_pred_literal),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::program::{EffectOp, Gate};
    use crate::ability::registry::{AbilityRegistry, AbilityRegistryBuilder};
    use crate::ability::AbilityProgram;

    /// Build a minimal one-ability registry for upload tests.
    fn one_ability_registry() -> AbilityRegistry {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 30.0 }],
        );
        let mut b = AbilityRegistryBuilder::new();
        let _ = b.register(prog);
        b.build()
    }

    #[test]
    fn upload_packed_registry_creates_one_buffer_per_column() {
        // GPU-touching test — skip silently when no adapter (CI without
        // a wgpu backend). The buffer-count assertion runs unconditionally
        // through the type system: if a column's buffer field is missing
        // from `PackedAbilityRegistryGpu`, this file would fail to compile.
        let ctx = match GpuContext::new_blocking() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping: no GPU adapter available");
                return;
            }
        };
        let registry = one_ability_registry();
        let packed = PackedAbilityRegistry::pack(&registry);
        let gpu = PackedAbilityRegistryGpu::upload(&packed, &ctx, "test");

        assert_eq!(gpu.n_abilities, 1);
        // Spot-check that each buffer has nonzero size (column populated).
        // wgpu::Buffer doesn't expose len() directly post-creation; we
        // rely on construction succeeding without panic as the smoke
        // signal here.
        let _ = (
            &gpu.hints, &gpu.cooldown_ticks, &gpu.range, &gpu.gate_flags,
            &gpu.delivery_kind, &gpu.effect_kinds, &gpu.effect_payload_a,
            &gpu.effect_payload_b, &gpu.tag_values, &gpu.stackings,
            &gpu.chances, &gpu.lifetime_kinds, &gpu.lifetime_payloads,
            &gpu.area_kinds, &gpu.area_args, &gpu.scaling_stat_refs,
            &gpu.scaling_percents,
            &gpu.nested_effect_kinds, &gpu.nested_effect_payload_a,
            &gpu.nested_effect_payload_b,
            &gpu.when_pred_binder, &gpu.when_pred_field,
            &gpu.when_pred_op, &gpu.when_pred_literal,
        );
    }
}
