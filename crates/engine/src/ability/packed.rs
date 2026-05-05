//! Wave 1.9 — SoA packing of `AbilityRegistry` for GPU consumption.
//!
//! `AbilityRegistry` (host-side) is a `Vec<AbilityProgram>` where each
//! program carries a `SmallVec` of effects + a `SmallVec` of tag pairs.
//! That layout is fine for the CPU cast cascade, but the GPU dispatch
//! kernels (Wave 2+) need fixed-stride flat columns they can address by
//! `(ability_id, slot_index)` without chasing pointers.
//!
//! `PackedAbilityRegistry::pack` walks the frozen registry once at
//! startup and emits a Struct-of-Arrays layout: one row per ability,
//! multiple `Vec<u32>` / `Vec<f32>` columns. Effects + tags are flattened
//! to row-major arrays with a stride pinned by the per-program max
//! constants (`MAX_EFFECTS_PER_PROGRAM` for effects,
//! `NUM_ABILITY_TAGS == AbilityTag::COUNT` for tags). Empty effect slots
//! are tagged with `EFFECT_KIND_EMPTY` so a GPU dispatch loop can break
//! early; the `hint:` column uses `HINT_NONE_SENTINEL` as the
//! "no hint authored" marker (distinct from any `AbilityHint`
//! discriminant).
//!
//! Effect-payload encoding lives in this module's `pack_effect` helper —
//! the `effect_payload_a` / `effect_payload_b` slots are interpreted
//! per `effect_kinds[i]`. The encoding is pinned by
//! `crates/engine/src/schema_hash.rs` (the `PackedAbilityRegistry:SoA{...}`
//! line) so any layout change forces a coordinated bump of the engine
//! schema hash + the WGSL constants once kernels exist.
//!
//! Constitution touch-points:
//! * P1 (compiler-first): packing runs at startup, not on the tick path.
//! * P2 (schema-hash): every layout change here MUST bump
//!   `crates/engine/.schema_hash`. The `pack_*` test set guards the
//!   in-Rust contract; the schema-hash test guards the cross-backend
//!   contract.
//! * P10 (no panics on hot path): `pack` runs once at startup, not on
//!   the deterministic tick path. Internal-invariant `panic!`/`expect`
//!   sites here would only fire on a registry with structurally invalid
//!   contents — that would itself be a compiler bug, not a runtime
//!   issue.
//! * P5 (determinism): pack is a pure function of the registry — no
//!   `HashMap` iteration leakage, no time-of-day inputs, no thread state.

use super::program::{
    Area, Delivery, EffectOp, MAX_EFFECTS_PER_PROGRAM, MAX_TAGS_PER_PROGRAM, TargetSelector,
};
use super::{AbilityProgram, AbilityRegistry, AbilityTag};

/// Sentinel for the `hints` column when an ability has no `hint:` set.
/// Distinct from any `AbilityHint` discriminant (0..=3 today). Pinned at
/// `0xFFFF_FFFF` so a GPU `u32 == HINT_NONE_SENTINEL` test is one cmp.
pub const HINT_NONE_SENTINEL: u32 = u32::MAX;

/// Number of tag columns per ability — matches `AbilityTag::COUNT`.
/// Stride for `tag_values` row addressing: `tag_values[ab * NUM_ABILITY_TAGS + tag]`.
pub const NUM_ABILITY_TAGS: usize = AbilityTag::COUNT;

/// Effect-op kind tag for an empty slot (program had fewer than
/// `MAX_EFFECTS_PER_PROGRAM` effects). Distinct from any `EffectOp`
/// discriminant (0..=15 today, including the Wave 2 piece 1 control
/// verbs Root/Silence/Fear/Taunt and the Wave 2 piece 2 movement verbs
/// Dash/Blink/Knockback/Pull). GPU dispatch loops break early on this
/// sentinel.
pub const EFFECT_KIND_EMPTY: u32 = 0xFF;

// Compile-time guard: `MAX_TAGS_PER_PROGRAM` and `NUM_ABILITY_TAGS` must
// stay aligned. Both are derived from `AbilityTag::COUNT` today; a future
// refactor that decouples them would bump the schema hash and need a
// fresh integration plan.
const _: () = assert!(MAX_TAGS_PER_PROGRAM == NUM_ABILITY_TAGS);

/// SoA layout of the frozen `AbilityRegistry`, ready for GPU upload.
///
/// Each `Vec<T>` is one column; row-N maps to slot-N
/// (`AbilityId::new(N+1).slot()`). `effect_kinds` / `effect_payload_*`
/// use a row-major flat layout with stride `MAX_EFFECTS_PER_PROGRAM`;
/// `tag_values` uses stride `NUM_ABILITY_TAGS`.
///
/// Field ordering + payload encoding is pinned by the
/// `PackedAbilityRegistry:SoA{...}` line in
/// `crates/engine/src/schema_hash.rs`. Renaming, reordering, or changing
/// any payload encoding forces a schema-hash bump.
pub struct PackedAbilityRegistry {
    /// Number of abilities packed — equals `registry.len()`. Cached as a
    /// scalar so callers binding GPU buffers do not re-derive it from
    /// per-column lengths.
    pub n_abilities: usize,

    // -- Per-ability scalar columns (one entry per ability). --

    /// `AbilityHint::discriminant() as u32`, or `HINT_NONE_SENTINEL` when
    /// the program has no hint authored.
    pub hints: Vec<u32>,

    /// `gate.cooldown_ticks`. Bit-for-bit copy from the program.
    pub cooldown_ticks: Vec<u32>,

    /// Range derived from `Area::SingleTarget { range }`. Other Area
    /// shapes (Cone/Circle/AoE — Wave 2+) will demand new columns; a
    /// future refactor adds them with a coordinated schema-hash bump.
    pub range: Vec<f32>,

    /// Bitfield: bit 0 = `gate.hostile_only`, bit 1 = `gate.line_of_sight`.
    /// Future bits reserved (e.g. `requires_los_to_origin`).
    pub gate_flags: Vec<u32>,

    /// `Delivery` discriminant. Pinned at `Instant=0` today, future
    /// `Projectile` / `Zone` lands with their resolver code.
    pub delivery_kind: Vec<u32>,

    // -- Effect rows (flat, stride = MAX_EFFECTS_PER_PROGRAM = 4). --

    /// `EffectOp` discriminant per slot, or `EFFECT_KIND_EMPTY` when the
    /// program had fewer than `MAX_EFFECTS_PER_PROGRAM` effects.
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub effect_kinds: Vec<u32>,

    /// First payload word per effect slot. Encoding depends on the slot's
    /// `effect_kinds[i]` value; see `pack_effect` for the per-kind table.
    /// Length: same as `effect_kinds`.
    pub effect_payload_a: Vec<u32>,

    /// Second payload word per effect slot. Only meaningful for kinds
    /// that need two words (`Slow`, `CastAbility`). Zero otherwise so the
    /// `pack_partial_effect_row_pads_with_sentinel` test asserts a
    /// stable column.
    /// Length: same as `effect_kinds`.
    pub effect_payload_b: Vec<u32>,

    // -- Tag rows (flat, stride = NUM_ABILITY_TAGS = 6). --

    /// Per-tag numeric power ratings, row-major.
    /// `tag_values[ab * NUM_ABILITY_TAGS + tag.index()]`. Default `0.0`
    /// for any tag not present on the program.
    /// Length: `n_abilities * NUM_ABILITY_TAGS`.
    pub tag_values: Vec<f32>,
}

impl PackedAbilityRegistry {
    /// Pack a frozen `AbilityRegistry` into the SoA layout. Pure function
    /// of the registry — no global state, no RNG, no time-of-day inputs.
    /// Runs once at startup (typically right after
    /// `dsl_compiler::ability_registry::build_registry`).
    pub fn pack(registry: &AbilityRegistry) -> Self {
        let n = registry.len();

        // Reserve exact capacities so the resulting Vecs have no slack.
        let mut hints = Vec::with_capacity(n);
        let mut cooldown_ticks = Vec::with_capacity(n);
        let mut range = Vec::with_capacity(n);
        let mut gate_flags = Vec::with_capacity(n);
        let mut delivery_kind = Vec::with_capacity(n);

        let effect_total = n * MAX_EFFECTS_PER_PROGRAM;
        let mut effect_kinds = Vec::with_capacity(effect_total);
        let mut effect_payload_a = Vec::with_capacity(effect_total);
        let mut effect_payload_b = Vec::with_capacity(effect_total);

        let tag_total = n * NUM_ABILITY_TAGS;
        let mut tag_values = vec![0.0_f32; tag_total];

        for slot in 0..n {
            // `AbilityId` is 1-based; the registry's `get` accepts an id,
            // so reconstruct it from the slot. The registry guarantees
            // every slot in `0..len()` is occupied.
            let id = super::AbilityId::new((slot as u32) + 1)
                .expect("slot+1 is non-zero");
            let program = registry
                .get(id)
                .expect("registry slot in 0..len() must resolve to a program");

            pack_program_columns(
                program,
                &mut hints,
                &mut cooldown_ticks,
                &mut range,
                &mut gate_flags,
                &mut delivery_kind,
                &mut effect_kinds,
                &mut effect_payload_a,
                &mut effect_payload_b,
            );
            pack_program_tags(program, slot, &mut tag_values);
        }

        Self {
            n_abilities: n,
            hints,
            cooldown_ticks,
            range,
            gate_flags,
            delivery_kind,
            effect_kinds,
            effect_payload_a,
            effect_payload_b,
            tag_values,
        }
    }

    /// Number of abilities packed — alias for `n_abilities` to match
    /// `AbilityRegistry::len`.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_abilities
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_abilities == 0
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Pack one `AbilityProgram`'s scalar + effect columns. Tags are packed
/// separately by `pack_program_tags` because they need an explicit slot
/// index for row addressing into the pre-zeroed `tag_values` buffer.
#[allow(clippy::too_many_arguments)]
fn pack_program_columns(
    program: &AbilityProgram,
    hints: &mut Vec<u32>,
    cooldown_ticks: &mut Vec<u32>,
    range: &mut Vec<f32>,
    gate_flags: &mut Vec<u32>,
    delivery_kind: &mut Vec<u32>,
    effect_kinds: &mut Vec<u32>,
    effect_payload_a: &mut Vec<u32>,
    effect_payload_b: &mut Vec<u32>,
) {
    // -- Hint column. --
    hints.push(match program.hint {
        Some(h) => h.discriminant() as u32,
        None => HINT_NONE_SENTINEL,
    });

    // -- Gate columns. --
    cooldown_ticks.push(program.gate.cooldown_ticks);
    let mut flags: u32 = 0;
    if program.gate.hostile_only {
        flags |= 1 << 0;
    }
    if program.gate.line_of_sight {
        flags |= 1 << 1;
    }
    gate_flags.push(flags);

    // -- Area column. --
    let r = match program.area {
        Area::SingleTarget { range } => range,
    };
    range.push(r);

    // -- Delivery column. --
    delivery_kind.push(pack_delivery(program.delivery));

    // -- Effect rows (stride = MAX_EFFECTS_PER_PROGRAM). --
    for i in 0..MAX_EFFECTS_PER_PROGRAM {
        let (kind, a, b) = match program.effects.get(i) {
            Some(op) => pack_effect(*op),
            None => (EFFECT_KIND_EMPTY, 0, 0),
        };
        effect_kinds.push(kind);
        effect_payload_a.push(a);
        effect_payload_b.push(b);
    }
}

/// Splat one program's `(tag, value)` smallvec into the row-major
/// `tag_values` buffer. Slots not present remain at the pre-zeroed `0.0`.
fn pack_program_tags(program: &AbilityProgram, slot: usize, tag_values: &mut [f32]) {
    let base = slot * NUM_ABILITY_TAGS;
    for &(tag, value) in program.tags.iter() {
        tag_values[base + tag.index()] = value;
    }
}

/// Encode a `Delivery` to its u32 discriminant. Today only `Instant=0`
/// exists; new variants land with their resolver and bump the hash.
#[inline]
fn pack_delivery(d: Delivery) -> u32 {
    match d {
        Delivery::Instant => 0,
    }
}

/// Encode one `EffectOp` to `(kind, payload_a, payload_b)`.
///
/// Per-kind encoding (mirrored by the schema-hash string):
/// * `Damage` / `Heal` / `Shield`  -> `(disc, f32::to_bits(amount), 0)`
/// * `Stun`                        -> `(disc, duration_ticks, 0)`
/// * `Slow`                        -> `(disc, duration_ticks, factor_q8 as i16 as u32)`
/// * `TransferGold`                -> `(disc, amount as i32 as u32, 0)`
/// * `ModifyStanding`              -> `(disc, delta as i16 as u32, 0)`
/// * `CastAbility`                 -> `(disc, ability.raw(), selector as u32)`
/// * `Root` / `Silence` / `Fear` / `Taunt`
///                                 -> `(disc, duration_ticks, 0)`  (Wave 2 piece 1; same shape as `Stun`)
/// * `Dash` / `Blink` / `Knockback` / `Pull`
///                                 -> `(disc, f32::to_bits(distance), 0)`  (Wave 2 piece 2; same shape as `Damage`)
///
/// Sign-bearing payloads use sign-preserving bitcasts (`as i16 as u32`)
/// so a GPU shader doing `bitcast<i32>(payload_a)` recovers the signed
/// value losslessly.
#[inline]
fn pack_effect(op: EffectOp) -> (u32, u32, u32) {
    // The discriminant matches `#[repr(u8)]` ordinals on `EffectOp`; the
    // schema_hash string pins those ordinals.
    match op {
        EffectOp::Damage { amount } => (0, amount.to_bits(), 0),
        EffectOp::Heal { amount } => (1, amount.to_bits(), 0),
        EffectOp::Shield { amount } => (2, amount.to_bits(), 0),
        EffectOp::Stun { duration_ticks } => (3, duration_ticks, 0),
        EffectOp::Slow { duration_ticks, factor_q8 } => {
            (4, duration_ticks, factor_q8 as i32 as u32)
        }
        EffectOp::TransferGold { amount } => (5, amount as u32, 0),
        EffectOp::ModifyStanding { delta } => (6, delta as i32 as u32, 0),
        EffectOp::CastAbility { ability, selector } => {
            (7, ability.raw(), pack_selector(selector))
        }
        // Wave 2 piece 1 — control verbs share `Stun`'s shape exactly.
        EffectOp::Root { duration_ticks } => (8, duration_ticks, 0),
        EffectOp::Silence { duration_ticks } => (9, duration_ticks, 0),
        EffectOp::Fear { duration_ticks } => (10, duration_ticks, 0),
        EffectOp::Taunt { duration_ticks } => (11, duration_ticks, 0),
        // Wave 2 piece 2 — movement verbs share `Damage`'s shape exactly.
        // `distance` is bit-cast to u32 via `f32::to_bits` so a GPU shader
        // doing `bitcast<f32>(payload_a)` recovers the value losslessly.
        EffectOp::Dash      { distance } => (12, distance.to_bits(), 0),
        EffectOp::Blink     { distance } => (13, distance.to_bits(), 0),
        EffectOp::Knockback { distance } => (14, distance.to_bits(), 0),
        EffectOp::Pull      { distance } => (15, distance.to_bits(), 0),
    }
}

#[inline]
fn pack_selector(s: TargetSelector) -> u32 {
    match s {
        TargetSelector::Target => 0,
        TargetSelector::Caster => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ability::{AbilityHint, AbilityId, AbilityRegistryBuilder};
    use crate::ability::program::Gate;

    fn build(programs: Vec<AbilityProgram>) -> AbilityRegistry {
        let mut b = AbilityRegistryBuilder::new();
        for p in programs {
            b.register(p);
        }
        b.build()
    }

    #[test]
    fn pack_empty_registry() {
        let r = AbilityRegistry::new();
        let p = PackedAbilityRegistry::pack(&r);
        assert_eq!(p.n_abilities, 0);
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert!(p.hints.is_empty());
        assert!(p.cooldown_ticks.is_empty());
        assert!(p.range.is_empty());
        assert!(p.gate_flags.is_empty());
        assert!(p.delivery_kind.is_empty());
        assert!(p.effect_kinds.is_empty());
        assert!(p.effect_payload_a.is_empty());
        assert!(p.effect_payload_b.is_empty());
        assert!(p.tag_values.is_empty());
    }

    #[test]
    fn pack_single_damage() {
        // Single-target Damage 15 with cooldown 10, hostile_only=true.
        // Asserts EVERY column slot for a single ability so a future
        // layout drift surfaces here before the schema-hash test.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 15.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.n_abilities, 1);
        assert_eq!(p.len(), 1);
        assert!(!p.is_empty());

        // Per-ability scalars.
        assert_eq!(p.hints, vec![HINT_NONE_SENTINEL]);
        assert_eq!(p.cooldown_ticks, vec![10]);
        assert_eq!(p.range, vec![5.0]);
        assert_eq!(p.gate_flags, vec![0b01]);
        assert_eq!(p.delivery_kind, vec![0]);

        // Effect row — slot 0 holds Damage(15.0), slots 1..4 are empty.
        assert_eq!(p.effect_kinds.len(), MAX_EFFECTS_PER_PROGRAM);
        assert_eq!(p.effect_kinds[0], 0); // Damage discriminant
        assert_eq!(p.effect_payload_a[0], 15.0_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.effect_kinds[i], EFFECT_KIND_EMPTY);
            assert_eq!(p.effect_payload_a[i], 0);
            assert_eq!(p.effect_payload_b[i], 0);
        }

        // No tags authored -> entire row is zero.
        assert_eq!(p.tag_values.len(), NUM_ABILITY_TAGS);
        for v in &p.tag_values {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn pack_full_effect_row() {
        // Mix four distinct kinds so each slot exercises a different
        // discriminant + payload encoding.
        let prog = AbilityProgram::new_single_target(
            3.0,
            Gate { cooldown_ticks: 5, hostile_only: true, line_of_sight: true },
            [
                EffectOp::Damage { amount: 10.0 },
                EffectOp::Heal { amount: 5.0 },
                EffectOp::Shield { amount: 7.5 },
                EffectOp::Stun { duration_ticks: 20 },
            ],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.effect_kinds, vec![0, 1, 2, 3]);
        assert_eq!(
            p.effect_payload_a,
            vec![
                10.0_f32.to_bits(),
                5.0_f32.to_bits(),
                7.5_f32.to_bits(),
                20,
            ],
        );
        for v in &p.effect_payload_b {
            assert_eq!(*v, 0);
        }
        // Confirm both gate-flag bits round-trip.
        assert_eq!(p.gate_flags, vec![0b11]);
    }

    #[test]
    fn pack_partial_effect_row_pads_with_sentinel() {
        let prog = AbilityProgram::new_single_target(
            2.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 1.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.effect_kinds[0], 0);
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.effect_kinds[i], EFFECT_KIND_EMPTY,
                "slot {i} must be the empty sentinel");
        }
        // Empty slots also zero their payload columns.
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.effect_payload_a[i], 0);
            assert_eq!(p.effect_payload_b[i], 0);
        }
    }

    #[test]
    fn pack_hint_none_sentinel() {
        let p_none = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 1.0 }],
        );
        let p_def = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Shield { amount: 1.0 }],
        )
        .with_hint(AbilityHint::Defense);

        let reg = build(vec![p_none, p_def]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.hints[0], HINT_NONE_SENTINEL);
        // AbilityHint::Defense == 1 per `#[repr(u8)]`.
        assert_eq!(p.hints[1], 1);
    }

    #[test]
    fn pack_tag_row_default_zero_and_set_value() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 1.0 }],
        )
        .with_tags([(AbilityTag::Magical, 42.0)]);

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // ab=0, NUM_ABILITY_TAGS=6. Magical is index 1.
        assert_eq!(p.tag_values.len(), NUM_ABILITY_TAGS);
        assert_eq!(p.tag_values[0 * NUM_ABILITY_TAGS + AbilityTag::Magical.index()], 42.0);
        // Every other tag column is the default 0.0.
        for tag in AbilityTag::all() {
            if tag == AbilityTag::Magical {
                continue;
            }
            assert_eq!(
                p.tag_values[0 * NUM_ABILITY_TAGS + tag.index()], 0.0,
                "tag {tag:?} must default to 0.0",
            );
        }
    }

    #[test]
    fn pack_slow_payload_b_holds_factor_q8() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Slow { duration_ticks: 30, factor_q8: -64 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Slow discriminant == 4.
        assert_eq!(p.effect_kinds[0], 4);
        assert_eq!(p.effect_payload_a[0], 30);
        // -64 as i16 sign-extended into u32 == 0xFFFFFFC0.
        assert_eq!(p.effect_payload_b[0], 0xFFFF_FFC0);
    }

    #[test]
    fn pack_cast_ability_payload() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::CastAbility {
                ability: AbilityId::new(7).unwrap(),
                selector: TargetSelector::Caster,
            }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // CastAbility discriminant == 7.
        assert_eq!(p.effect_kinds[0], 7);
        assert_eq!(p.effect_payload_a[0], 7);
        // Caster selector == 1.
        assert_eq!(p.effect_payload_b[0], 1);
    }

    // -- Wave 2 piece 1 — control verb pack tests. Each mirrors the
    // Stun shape exactly: discriminant + duration in `payload_a`,
    // `payload_b` zero. ---------------------------------------------------
    #[test]
    fn pack_root_payload() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Root { duration_ticks: 20 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Root discriminant == 8.
        assert_eq!(p.effect_kinds[0], 8);
        assert_eq!(p.effect_payload_a[0], 20);
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_silence_payload() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Silence { duration_ticks: 30 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Silence discriminant == 9.
        assert_eq!(p.effect_kinds[0], 9);
        assert_eq!(p.effect_payload_a[0], 30);
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_fear_payload() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Fear { duration_ticks: 15 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Fear discriminant == 10.
        assert_eq!(p.effect_kinds[0], 10);
        assert_eq!(p.effect_payload_a[0], 15);
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_taunt_payload() {
        let prog = AbilityProgram::new_single_target(
            1.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Taunt { duration_ticks: 40 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Taunt discriminant == 11.
        assert_eq!(p.effect_kinds[0], 11);
        assert_eq!(p.effect_payload_a[0], 40);
        assert_eq!(p.effect_payload_b[0], 0);
    }

    // -- Wave 2 piece 2 — movement verb pack tests. Each mirrors the
    // Damage shape exactly: discriminant + `f32::to_bits(distance)` in
    // `payload_a`, `payload_b` zero. ---------------------------------------
    #[test]
    fn pack_dash_payload() {
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Dash { distance: 4.5 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Dash discriminant == 12.
        assert_eq!(p.effect_kinds[0], 12);
        assert_eq!(p.effect_payload_a[0], 4.5_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_blink_payload() {
        let prog = AbilityProgram::new_single_target(
            6.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Blink { distance: 6.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Blink discriminant == 13.
        assert_eq!(p.effect_kinds[0], 13);
        assert_eq!(p.effect_payload_a[0], 6.0_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_knockback_payload() {
        let prog = AbilityProgram::new_single_target(
            2.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Knockback { distance: 3.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Knockback discriminant == 14.
        assert_eq!(p.effect_kinds[0], 14);
        assert_eq!(p.effect_payload_a[0], 3.0_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_pull_payload() {
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Pull { distance: 2.5 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Pull discriminant == 15.
        assert_eq!(p.effect_kinds[0], 15);
        assert_eq!(p.effect_payload_a[0], 2.5_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_is_deterministic() {
        // Pack twice; assert column-by-column equality so a future
        // refactor that leaks `HashMap` iteration order surfaces here.
        let prog_a = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 10, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 15.0 }],
        )
        .with_hint(AbilityHint::Damage)
        .with_tags([(AbilityTag::Physical, 1.0), (AbilityTag::Magical, 2.0)]);

        let prog_b = AbilityProgram::new_single_target(
            2.0,
            Gate { cooldown_ticks: 5, hostile_only: false, line_of_sight: false },
            [
                EffectOp::Heal { amount: 7.0 },
                EffectOp::Shield { amount: 3.0 },
            ],
        )
        .with_tags([(AbilityTag::Heal, 9.5)]);

        let reg = build(vec![prog_a, prog_b]);
        let p1 = PackedAbilityRegistry::pack(&reg);
        let p2 = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p1.n_abilities, p2.n_abilities);
        assert_eq!(p1.hints, p2.hints);
        assert_eq!(p1.cooldown_ticks, p2.cooldown_ticks);
        assert_eq!(p1.range, p2.range);
        assert_eq!(p1.gate_flags, p2.gate_flags);
        assert_eq!(p1.delivery_kind, p2.delivery_kind);
        assert_eq!(p1.effect_kinds, p2.effect_kinds);
        assert_eq!(p1.effect_payload_a, p2.effect_payload_a);
        assert_eq!(p1.effect_payload_b, p2.effect_payload_b);
        assert_eq!(p1.tag_values, p2.tag_values);
    }
}
