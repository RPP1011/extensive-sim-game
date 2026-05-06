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
    Area, Delivery, EffectAreaShape, EffectOp, EffectScaling, LifetimeMode,
    MAX_EFFECTS_PER_PROGRAM, MAX_SCALINGS_PER_EFFECT, MAX_TAGS_PER_PROGRAM, StackingMode,
    TargetSelector,
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
/// discriminant (0..=19 today, including the Wave 2 piece 1 control
/// verbs Root/Silence/Fear/Taunt, the Wave 2 piece 2 movement verbs
/// Dash/Blink/Knockback/Pull, the Wave 2 piece 3 advanced verbs
/// Execute/SelfDamage, and the Wave 2 piece 4 buff verbs
/// LifeSteal/DamageModify). GPU dispatch loops break early on this
/// sentinel.
pub const EFFECT_KIND_EMPTY: u32 = 0xFF;

/// Sentinel for the `stackings` column when an effect did not carry an
/// explicit `stacking <mode>` modifier in the source DSL. Apply
/// handlers should treat this slot as `StackingMode::Refresh` per
/// `project_buff_stacking_rule.md`. Distinct from any `StackingMode`
/// discriminant (0..=2 today: Refresh/Stack/Extend) so a GPU shader
/// doing `payload == STACKING_NONE_SENTINEL` is one cmp.
pub const STACKING_NONE_SENTINEL: u8 = 0xFF;

/// Sentinel for the `chances` column when an effect has no `chance`
/// modifier. The Q16 fixed-point encoding of `chance N%` packs
/// `0..=65534` (0%..~100%) into the column, with `65535` (`u16::MAX`)
/// reserved as the "no modifier authored" marker. Apply handlers
/// should treat `CHANCE_NONE_SENTINEL` as "always fires" (no RNG
/// gate) — distinct from `0` (encoded "0%": never fires).
///
/// Sentinel-vs-100% conflict resolution: the lowering pass clamps the
/// q16 encoding to `0..=65534` so `chance 100%` lowers to
/// `Some(65534)` (one less than `u16::MAX`). Authors who want
/// "always" should omit the modifier — both produce the same runtime
/// behavior, but the column stays a single-cmp test.
pub const CHANCE_NONE_SENTINEL: u16 = u16::MAX;

/// Sentinel for the `lifetime_kinds` column when an effect has no
/// lifetime modifier (`until_caster_dies` / `damageable_hp(N)` /
/// `break_on_damage`). Distinct from any `LifetimeMode` discriminant
/// (0..=2 today). Apply handlers should treat this as "effect persists
/// indefinitely (or for the verb's own duration if the verb has one)";
/// distinct from discriminant `0` (UntilCasterDies).
///
/// Companion `lifetime_payloads` column stores `0.0` at the matching
/// slot whenever the kind is the sentinel — payload is ONLY meaningful
/// when `lifetime_kinds[i] == 1` (DamageableHp).
pub const LIFETIME_KIND_NONE_SENTINEL: u8 = 0xFF;

/// Sentinel for the `area_kinds` column when an effect has no
/// `in <shape>(args)` modifier. Distinct from any `ShapeKind`
/// discriminant (0..=11 today). Apply handlers should treat this slot
/// as "single-target effect (use program.area)"; companion
/// `area_args` slot stores `[0.0; 4]` whenever the kind is the
/// sentinel — args are ONLY meaningful when
/// `area_kinds[i] != SHAPE_KIND_NONE_SENTINEL`.
pub const SHAPE_KIND_NONE_SENTINEL: u8 = 0xFF;

/// Sentinel for the `scaling_stat_refs` column when a per-effect
/// scaling slot is unused (either the effect carried no `+ N% stat_ref`
/// modifier at all, or the slot index exceeds the number of scalings
/// authored on that effect). Distinct from any `ScalingStatRef`
/// discriminant (0..=7 today). Apply handlers should treat this slot
/// as "no scaling for this slot — flat amount only"; companion
/// `scaling_percents` slot stores `0.0` whenever the stat-ref is the
/// sentinel — percents are ONLY meaningful when
/// `scaling_stat_refs[i] != SCALING_STAT_NONE_SENTINEL`.
pub const SCALING_STAT_NONE_SENTINEL: u8 = 0xFF;

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

    // -- Stacking rows (flat, stride = MAX_EFFECTS_PER_PROGRAM = 4). --

    /// Per-effect stacking mode encoded as `StackingMode as u8`, or
    /// `STACKING_NONE_SENTINEL` (0xFF) when the source effect did not
    /// carry an explicit `stacking <mode>` modifier. Apply handlers
    /// should treat the sentinel as `StackingMode::Refresh` per
    /// `project_buff_stacking_rule.md`.
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub stackings: Vec<u8>,

    // -- Chance rows (flat, stride = MAX_EFFECTS_PER_PROGRAM = 4). --

    /// Per-effect probability gate, q16 fixed-point. Valid values
    /// `0..=65534` encode `0% ..≈ 100%`; `CHANCE_NONE_SENTINEL` (0xFFFF
    /// = u16::MAX) marks slots where the source effect did not carry a
    /// `chance N%` modifier. Apply handlers should treat the sentinel
    /// as "always fires" (no RNG gate); for non-sentinel values they
    /// compare `per_agent_u32(seed, agent_id, tick, purpose) & 0xFFFF`
    /// against the slot.
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub chances: Vec<u16>,

    // -- Lifetime rows (flat, stride = MAX_EFFECTS_PER_PROGRAM = 4). --

    /// Per-effect lifetime kind discriminant
    /// (`LifetimeMode::discriminant() as u8`), or
    /// `LIFETIME_KIND_NONE_SENTINEL` (0xFF) when the source effect did
    /// not carry one of the three lifetime modifiers. Apply handlers
    /// should treat the sentinel as "effect persists indefinitely (or
    /// for the verb's own duration if the verb has one)"; for
    /// non-sentinel values they switch on `0` (UntilCasterDies) /
    /// `1` (DamageableHp — read companion payload column) /
    /// `2` (BreakOnDamage).
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub lifetime_kinds: Vec<u8>,

    /// Per-effect lifetime payload — only meaningful when
    /// `lifetime_kinds[i] == 1` (DamageableHp), in which case it holds
    /// the initial hp value of the damage budget. `0.0` for every other
    /// slot (sentinel slots, UntilCasterDies, BreakOnDamage). Stored
    /// alongside `lifetime_kinds` because `LifetimeMode::DamageableHp`
    /// is the first per-effect SoA modifier with variant data — flat
    /// `Vec<u8>` couldn't carry the f32 hp pool by itself.
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub lifetime_payloads: Vec<f32>,

    // -- Area shape rows (flat, stride = MAX_EFFECTS_PER_PROGRAM = 4). --

    /// Per-effect shape kind discriminant
    /// (`ShapeKind::discriminant() as u8`), or `SHAPE_KIND_NONE_SENTINEL`
    /// (0xFF) when the source effect did not carry an `in <shape>(args)`
    /// modifier. Apply handlers should treat the sentinel as "single-
    /// target effect (use program.area)". Spec §8 catalog: 5 disc-
    /// family (Circle/Cone/Line/Ring/Spread = 0..=4) + 7 volume-family
    /// (Box/Sphere/Column/Wall/Cylinder/Dome/Hull = 5..=11).
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM`.
    pub area_kinds: Vec<u8>,

    /// Per-effect shape args — flat `4 × f32` per effect slot, addressed
    /// `area_args[(slot * MAX_EFFECTS_PER_PROGRAM + i) * 4 .. + 4]`.
    /// `[0.0; 4]` for every slot whose kind is the sentinel. Args are
    /// the source-order positional values of the shape's constructor
    /// (per spec §8); shapes with fewer than 4 args zero-pad the tail
    /// — only `Wall` consumes all four (len/h/thick/facing).
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM * 4`.
    pub area_args: Vec<f32>,

    // -- Scaling rows (flat, stride =
    //    MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT). ---------

    /// Per-effect-per-scaling stat-ref discriminant
    /// (`ScalingStatRef::discriminant() as u8`), or
    /// `SCALING_STAT_NONE_SENTINEL` (0xFF) when the slot is unused
    /// (either the effect carried no `+ N% stat_ref` modifier, or the
    /// scaling-slot index exceeds the number of scalings authored on
    /// that effect). Multiple scalings per effect are allowed (e.g.
    /// `damage 50 + 30% AP + 20% AD`); the inner stride is
    /// `MAX_SCALINGS_PER_EFFECT` (2 today). Apply handlers should treat
    /// the sentinel as "no scaling for this slot"; for non-sentinel
    /// values they switch on `0..=7`
    /// (AttackDamage/AbilityPower/MaxHp/Hp/Armor/MagicResist/MoveSpeed/Mana).
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT`.
    pub scaling_stat_refs: Vec<u8>,

    /// Per-effect-per-scaling percent — only meaningful when
    /// `scaling_stat_refs[i] != SCALING_STAT_NONE_SENTINEL`, in which
    /// case it holds the fraction of the referenced stat to add to the
    /// effect's flat amount (e.g. `0.30` for "+ 30% AP"). `0.0` for
    /// every sentinel slot. Stored alongside `scaling_stat_refs`
    /// because `EffectScaling` is the third per-effect SoA modifier
    /// with variant data — flat `Vec<u8>` couldn't carry the f32
    /// percent by itself.
    /// Length: `n_abilities * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT`.
    pub scaling_percents: Vec<f32>,
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

        // Stackings: pre-fill with the "no stacking modifier" sentinel
        // so empty effect slots and effects without a `stacking <mode>`
        // modifier share a single resting value. Per-effect overrides
        // are written by `pack_program_stackings`.
        let stacking_total = n * MAX_EFFECTS_PER_PROGRAM;
        let mut stackings = vec![STACKING_NONE_SENTINEL; stacking_total];

        // Chances: pre-fill with the "no chance modifier" sentinel so
        // empty effect slots and effects without `chance N%` share a
        // single resting value. Per-effect overrides are written by
        // `pack_program_chances`.
        let chance_total = n * MAX_EFFECTS_PER_PROGRAM;
        let mut chances = vec![CHANCE_NONE_SENTINEL; chance_total];

        // Lifetimes: pre-fill kinds with the none-sentinel and
        // payloads with `0.0` so empty effect slots + effects without
        // a lifetime modifier share a single resting state. Per-effect
        // overrides land in `pack_program_lifetimes`.
        let lifetime_total = n * MAX_EFFECTS_PER_PROGRAM;
        let mut lifetime_kinds = vec![LIFETIME_KIND_NONE_SENTINEL; lifetime_total];
        let mut lifetime_payloads = vec![0.0_f32; lifetime_total];

        // Area shapes: pre-fill kinds with the none-sentinel and args
        // with `0.0` so empty effect slots + effects without an
        // `in <shape>(args)` modifier share a single resting state.
        // Per-effect overrides land in `pack_program_areas`. Args are
        // 4×f32 per slot (Wall consumes all four; other shapes zero-pad
        // the tail).
        let area_kinds_total = n * MAX_EFFECTS_PER_PROGRAM;
        let area_args_total = n * MAX_EFFECTS_PER_PROGRAM * 4;
        let mut area_kinds = vec![SHAPE_KIND_NONE_SENTINEL; area_kinds_total];
        let mut area_args = vec![0.0_f32; area_args_total];

        // Scalings: pre-fill stat_refs with the none-sentinel and
        // percents with `0.0` so empty effect slots + effects without a
        // `+ N% stat_ref` modifier (and unused inner slots when an
        // effect uses fewer scalings than MAX_SCALINGS_PER_EFFECT) all
        // share a single resting state. Per-effect-per-scaling
        // overrides land in `pack_program_scalings`.
        let scaling_total = n * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT;
        let mut scaling_stat_refs = vec![SCALING_STAT_NONE_SENTINEL; scaling_total];
        let mut scaling_percents = vec![0.0_f32; scaling_total];

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
            pack_program_stackings(program, slot, &mut stackings);
            pack_program_chances(program, slot, &mut chances);
            pack_program_lifetimes(
                program,
                slot,
                &mut lifetime_kinds,
                &mut lifetime_payloads,
            );
            pack_program_areas(program, slot, &mut area_kinds, &mut area_args);
            pack_program_scalings(
                program,
                slot,
                &mut scaling_stat_refs,
                &mut scaling_percents,
            );
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
            stackings,
            chances,
            lifetime_kinds,
            lifetime_payloads,
            area_kinds,
            area_args,
            scaling_stat_refs,
            scaling_percents,
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
    delivery_kind.push(pack_delivery(&program.delivery));

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

/// Splat one program's per-effect stacking modes into the row-major
/// `stackings` buffer. Slots already pre-filled with
/// `STACKING_NONE_SENTINEL`; only `Some(StackingMode)` entries
/// overwrite. `program.stackings` is index-parallel to `program.effects`
/// when populated; an empty `program.stackings` slice means no effect
/// carried a `stacking <mode>` modifier (every slot stays at the
/// sentinel).
fn pack_program_stackings(program: &AbilityProgram, slot: usize, stackings: &mut [u8]) {
    let base = slot * MAX_EFFECTS_PER_PROGRAM;
    for (i, mode) in program.stackings.iter().enumerate() {
        // Defensive bounds: `lower_ability_decl` enforces
        // `program.stackings.len() <= MAX_EFFECTS_PER_PROGRAM`, but a
        // future hand-built program could violate it; clamp instead of
        // panicking on the startup pack path.
        if i >= MAX_EFFECTS_PER_PROGRAM {
            break;
        }
        if let Some(m) = mode {
            stackings[base + i] = pack_stacking(*m);
        }
    }
}

/// Splat one program's per-effect chance gates into the row-major
/// `chances` buffer. Slots already pre-filled with
/// `CHANCE_NONE_SENTINEL`; only `Some(q16)` entries overwrite.
/// `program.chances` is index-parallel to `program.effects` when
/// populated; an empty `program.chances` slice means no effect carried
/// a `chance N%` modifier (every slot stays at the sentinel).
///
/// Defensive clamp: the lowering pass already keeps q16 in
/// `0..=65534` (one less than `u16::MAX`) so the sentinel stays
/// unambiguous, but if a hand-built program slips through with
/// `Some(u16::MAX)` we coerce it to `65534` rather than collide with
/// "no modifier authored".
fn pack_program_chances(program: &AbilityProgram, slot: usize, chances: &mut [u16]) {
    let base = slot * MAX_EFFECTS_PER_PROGRAM;
    for (i, ch) in program.chances.iter().enumerate() {
        if i >= MAX_EFFECTS_PER_PROGRAM {
            break;
        }
        if let Some(q16) = ch {
            // Reserve `u16::MAX` as the none-sentinel even on the
            // hand-built path.
            let v = if *q16 == CHANCE_NONE_SENTINEL { CHANCE_NONE_SENTINEL - 1 } else { *q16 };
            chances[base + i] = v;
        }
    }
}

/// Splat one program's per-effect lifetime modifiers into the row-major
/// `lifetime_kinds` + `lifetime_payloads` buffers. Slots already
/// pre-filled (`LIFETIME_KIND_NONE_SENTINEL` / `0.0`); only
/// `Some(LifetimeMode)` entries overwrite. `program.lifetimes` is
/// index-parallel to `program.effects` when populated; an empty
/// `program.lifetimes` slice means no effect carried a lifetime
/// modifier (every slot stays at the sentinel + `0.0`).
///
/// `LifetimeMode::DamageableHp` is the first per-effect SoA modifier
/// with variant data — the kind discriminant goes in `lifetime_kinds`,
/// the f32 hp pool in `lifetime_payloads`. Variants without a payload
/// (UntilCasterDies / BreakOnDamage) write `0.0` into the payload
/// column so the column is dense + non-sentinel-bearing.
fn pack_program_lifetimes(
    program:           &AbilityProgram,
    slot:              usize,
    lifetime_kinds:    &mut [u8],
    lifetime_payloads: &mut [f32],
) {
    let base = slot * MAX_EFFECTS_PER_PROGRAM;
    for (i, lt) in program.lifetimes.iter().enumerate() {
        // Defensive bounds parallel to `pack_program_stackings` /
        // `pack_program_chances` — the lowering pass enforces
        // `program.lifetimes.len() <= MAX_EFFECTS_PER_PROGRAM`, but a
        // hand-built program could violate it; clamp instead of
        // panicking on the startup pack path.
        if i >= MAX_EFFECTS_PER_PROGRAM {
            break;
        }
        if let Some(mode) = lt {
            // Bind through the named type so the `LifetimeMode` import
            // is genuinely used at lib level (without this, method
            // dispatch alone wouldn't keep the import alive under
            // `-D unused-imports`). The encoding routes through the
            // type's pinned helpers — rename or reorder the variants
            // and the schema-hash test will guard the discriminant
            // contract.
            let m: LifetimeMode = *mode;
            lifetime_kinds[base + i] = m.discriminant();
            lifetime_payloads[base + i] = m.payload_f32();
        }
    }
}

/// Splat one program's per-effect area shapes into the row-major
/// `area_kinds` + `area_args` buffers. Slots already pre-filled
/// (`SHAPE_KIND_NONE_SENTINEL` / `[0.0; 4]`); only `Some(EffectAreaShape)`
/// entries overwrite. `program.per_effect_areas` is index-parallel to
/// `program.effects` when populated; an empty `program.per_effect_areas`
/// slice means no effect carried an `in <shape>(args)` modifier (every
/// slot stays at the sentinel + `[0.0; 4]`).
///
/// `EffectAreaShape` is the second per-effect SoA modifier with variant
/// data (mirrors the `LifetimeMode::DamageableHp` shape) — the kind
/// discriminant goes in `area_kinds`, the 4×f32 args block in
/// `area_args`. Slots without the modifier write `[0.0; 4]` into the
/// args column so the column is dense + non-sentinel-bearing.
fn pack_program_areas(
    program:    &AbilityProgram,
    slot:       usize,
    area_kinds: &mut [u8],
    area_args:  &mut [f32],
) {
    let kind_base = slot * MAX_EFFECTS_PER_PROGRAM;
    let arg_base = slot * MAX_EFFECTS_PER_PROGRAM * 4;
    for (i, area) in program.per_effect_areas.iter().enumerate() {
        // Defensive bounds parallel to the other per-effect packers —
        // the lowering pass enforces
        // `program.per_effect_areas.len() <= MAX_EFFECTS_PER_PROGRAM`,
        // but a hand-built program could violate it; clamp instead of
        // panicking on the startup pack path.
        if i >= MAX_EFFECTS_PER_PROGRAM {
            break;
        }
        if let Some(shape) = area {
            // Bind through the named type so the `EffectAreaShape`
            // import is genuinely used at lib level (parallel to
            // `pack_program_lifetimes`'s LifetimeMode bind).
            let s: EffectAreaShape = *shape;
            area_kinds[kind_base + i] = s.kind.discriminant();
            let a = arg_base + i * 4;
            area_args[a]     = s.args[0];
            area_args[a + 1] = s.args[1];
            area_args[a + 2] = s.args[2];
            area_args[a + 3] = s.args[3];
        }
    }
}

/// Splat one program's per-effect-per-scaling modifiers into the
/// row-major `scaling_stat_refs` + `scaling_percents` buffers. Slots
/// already pre-filled (`SCALING_STAT_NONE_SENTINEL` / `0.0`); only
/// authored entries overwrite. `program.scalings_per_effect` is
/// outer-index-parallel to `program.effects` when populated; an empty
/// outer slice means no effect carried a `+ N% stat_ref` modifier
/// (every slot stays at the sentinel + `0.0`). Inner SmallVec is
/// bounded at `MAX_SCALINGS_PER_EFFECT` (2 today).
///
/// `EffectScaling` is the third per-effect SoA modifier with variant
/// data — its (stat_ref, percent) pair lives in the kind + payload
/// columns parallel to `EffectAreaShape`'s pattern. Stride is
/// `MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT` so a GPU shader
/// addresses slot `(ability, effect, scaling)` as
/// `slot * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT
///  + effect * MAX_SCALINGS_PER_EFFECT + scaling`.
fn pack_program_scalings(
    program:           &AbilityProgram,
    slot:              usize,
    scaling_stat_refs: &mut [u8],
    scaling_percents:  &mut [f32],
) {
    let base = slot * MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT;
    for (eff_i, inner) in program.scalings_per_effect.iter().enumerate() {
        // Defensive bounds parallel to the other per-effect packers —
        // the lowering pass enforces
        // `program.scalings_per_effect.len() <= MAX_EFFECTS_PER_PROGRAM`,
        // but a hand-built program could violate it; clamp instead of
        // panicking on the startup pack path.
        if eff_i >= MAX_EFFECTS_PER_PROGRAM {
            break;
        }
        for (sc_i, sc) in inner.iter().enumerate() {
            // Same defensive clamp on the inner stride.
            if sc_i >= MAX_SCALINGS_PER_EFFECT {
                break;
            }
            // Bind through the named type so the `EffectScaling` import
            // is genuinely used at lib level (parallel to the
            // `EffectAreaShape` bind in `pack_program_areas`).
            let entry: EffectScaling = *sc;
            let off = base + eff_i * MAX_SCALINGS_PER_EFFECT + sc_i;
            scaling_stat_refs[off] = entry.stat_ref.discriminant();
            scaling_percents[off]  = entry.percent;
        }
    }
}

/// Encode a `StackingMode` to its u8 discriminant. Pinned by
/// `crates/engine/src/schema_hash.rs` (the `StackingMode:` line).
#[inline]
fn pack_stacking(m: StackingMode) -> u8 {
    match m {
        StackingMode::Refresh => 0,
        StackingMode::Stack => 1,
        StackingMode::Extend => 2,
    }
}

/// Encode a `Delivery` to its u32 discriminant.
///
/// `Instant`             → 0 (no payload)
/// `Method { kind, raw }` → 1 + (kind ordinal << 8)
///                          (low byte signals "method delivery"; next
///                          byte carries the DeliveryMethodKind ordinal,
///                          stable per the enum's `#[repr(u8)]`).
///
/// `raw` is CPU-only metadata and does not pack — apply handlers
/// re-fetch from `AbilityProgram.delivery` (CPU-side) when needed.
#[inline]
fn pack_delivery(d: &Delivery) -> u32 {
    match d {
        Delivery::Instant => 0,
        Delivery::Method { kind, raw: _, hooks: _ } => 1 | ((*kind as u32) << 8),
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
/// * `Execute`                     -> `(disc, f32::to_bits(hp_threshold), 0)`  (Wave 2 piece 3; same shape as `Damage`)
/// * `SelfDamage`                  -> `(disc, f32::to_bits(amount), 0)`        (Wave 2 piece 3; same shape as `Damage`)
/// * `LifeSteal`                   -> `(disc, duration_ticks, fraction_q8 as i16 as u32)`   (Wave 2 piece 4; same shape as `Slow`)
/// * `DamageModify`                -> `(disc, duration_ticks, multiplier_q8 as i16 as u32)` (Wave 2 piece 4; same shape as `Slow`)
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
        // Wave 2 piece 3 — advanced verbs (`Execute` / `SelfDamage`)
        // also share `Damage`'s shape: a single f32 payload bit-cast
        // through `f32::to_bits`. No new SoA fields — `Execute` reads
        // `hot_hp` and `SelfDamage` re-emits a `Damaged` event the
        // existing ApplyDamage handler drains.
        EffectOp::Execute    { hp_threshold } => (16, hp_threshold.to_bits(), 0),
        EffectOp::SelfDamage { amount }       => (17, amount.to_bits(), 0),
        // Wave 2 piece 4 — buff verbs share `Slow`'s payload shape exactly:
        // `payload_a = duration_ticks`, `payload_b = q8_magnitude as i16 as
        // u32` (sign-preserving widen so a GPU shader can `bitcast<i32>`).
        EffectOp::LifeSteal    { duration_ticks, fraction_q8 } => {
            (18, duration_ticks, fraction_q8 as i32 as u32)
        }
        EffectOp::DamageModify { duration_ticks, multiplier_q8 } => {
            (19, duration_ticks, multiplier_q8 as i32 as u32)
        }
        // DoT/HoT — payload_a is amount-per-tick (f32 bits), payload_b
        // is duration_ticks. Apply handlers iterate `duration_ticks`
        // tick-events emitting Damaged/Healed each one. Same shape as
        // SelfDamage but two columns.
        EffectOp::DamageOverTime { amount, duration_ticks } => {
            (20, amount.to_bits(), duration_ticks)
        }
        EffectOp::HealOverTime { amount, duration_ticks } => {
            (21, amount.to_bits(), duration_ticks)
        }
        // TimedShield — same shape as DoT/HoT (amount + duration).
        EffectOp::TimedShield { amount, duration_ticks } => {
            (22, amount.to_bits(), duration_ticks)
        }
        // Buff — payload_a packs (stat ordinal in low byte | magnitude
        // q8 in high 16 bits, sign-preserved); payload_b is duration.
        // Same compact pattern as Slow's payload, but with an added
        // u8 stat selector. Kind=23.
        EffectOp::Buff { stat, magnitude_q8, duration_ticks } => {
            let pa = (stat as u32) | ((magnitude_q8 as i32 as u32) << 8);
            (23, pa, duration_ticks)
        }
        // Summon — payload_a is the template_hash (FxHash of the
        // template ident from `summon "<template>"`). payload_b packs
        // count in the high byte and lifetime_ticks in the low 24 bits
        // (24 bits of ticks ≈ 4.97 years at 10 Hz — well past any
        // sane minion lifetime). A GPU shader recovers via:
        //   count          = (payload_b >> 24) & 0xFF;
        //   lifetime_ticks = payload_b & 0x00FF_FFFF;
        // Apply handler not wired in this slice (see `EffectOp::Summon`
        // doc) — payload format is fixed now so the GPU pack column
        // does not bump again when the runtime arrives.
        EffectOp::Summon { template_hash, count, lifetime_ticks } => {
            let lifetime = lifetime_ticks & 0x00FF_FFFF;
            let pb = ((count as u32) << 24) | lifetime;
            (24, template_hash, pb)
        }
        // Non-combat verbs phase 1 — harvest / place_voxel.
        // payload_a is the FxHash of the resource / voxel ident;
        // payload_b is the harvest amount (widened from u16 to u32 for
        // the SoA column, zero-extended). PlaceVoxel has no second
        // payload — the cast's target position carries the placement
        // location; payload_b stays 0.
        EffectOp::Harvest    { kind_hash, amount }    => (25, kind_hash, amount as u32),
        EffectOp::PlaceVoxel { kind_hash }            => (26, kind_hash, 0),
        // Wave 2 piece 7 — `stealth for <duration>`. Duration in
        // payload_a; payload_b unused. Apply handlers tie the per-
        // agent stealth flag to a tick-stamp the same way Stun does.
        EffectOp::Stealth    { duration_ticks }       => (27, duration_ticks, 0),
        // Wave 2 piece 8 — remaining CC verbs. Same shape as Stun:
        // duration in payload_a, payload_b unused.
        EffectOp::Charm      { duration_ticks }       => (28, duration_ticks, 0),
        EffectOp::Grounded   { duration_ticks }       => (29, duration_ticks, 0),
        EffectOp::Suppress   { duration_ticks }       => (30, duration_ticks, 0),
        // Reflect packs duration in payload_a and the fraction (q8,
        // bit-cast through u16 then zero-extended) in payload_b
        // — same convention as `LifeSteal` / `DamageModify`.
        EffectOp::Reflect    { duration_ticks, fraction_q8 } =>
            (31, duration_ticks, (fraction_q8 as u16) as u32),
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
        assert!(p.stackings.is_empty());
        assert!(p.chances.is_empty());
        assert!(p.lifetime_kinds.is_empty());
        assert!(p.lifetime_payloads.is_empty());
        assert!(p.area_kinds.is_empty());
        assert!(p.area_args.is_empty());
        assert!(p.scaling_stat_refs.is_empty());
        assert!(p.scaling_percents.is_empty());
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

        // MAX_EFFECTS_PER_PROGRAM=6 (#131-followup), so the 4-effect
        // program packs to a 6-stride row with two trailing empties.
        assert_eq!(
            p.effect_kinds,
            vec![0, 1, 2, 3, EFFECT_KIND_EMPTY as u32, EFFECT_KIND_EMPTY as u32],
        );
        assert_eq!(
            p.effect_payload_a,
            vec![
                10.0_f32.to_bits(),
                5.0_f32.to_bits(),
                7.5_f32.to_bits(),
                20,
                0,
                0,
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

    // -- Wave 2 piece 3 — advanced verb pack tests. Each mirrors the
    // Damage shape exactly: discriminant + `f32::to_bits(...)` in
    // `payload_a`, `payload_b` zero. ---------------------------------------
    #[test]
    fn pack_execute_payload() {
        let prog = AbilityProgram::new_single_target(
            4.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Execute { hp_threshold: 25.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Execute discriminant == 16.
        assert_eq!(p.effect_kinds[0], 16);
        assert_eq!(p.effect_payload_a[0], 25.0_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    #[test]
    fn pack_self_damage_payload() {
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::SelfDamage { amount: 7.5 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // SelfDamage discriminant == 17.
        assert_eq!(p.effect_kinds[0], 17);
        assert_eq!(p.effect_payload_a[0], 7.5_f32.to_bits());
        assert_eq!(p.effect_payload_b[0], 0);
    }

    // -- Wave 2 piece 4 — buff verb pack tests. Each mirrors the `Slow`
    // shape exactly: discriminant + `duration_ticks` in `payload_a`,
    // sign-extended q8 magnitude in `payload_b`. --------------------------
    #[test]
    fn pack_lifesteal_payload() {
        // `lifesteal 0.5 (4s = 40 ticks)` → fraction_q8 = 128 (0.5 * 256).
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::LifeSteal { duration_ticks: 40, fraction_q8: 128 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // LifeSteal discriminant == 18.
        assert_eq!(p.effect_kinds[0], 18);
        assert_eq!(p.effect_payload_a[0], 40);
        // 128 sign-extends to itself in u32 (positive value).
        assert_eq!(p.effect_payload_b[0], 128);
    }

    #[test]
    fn pack_lifesteal_payload_sign_extends_negative_q8() {
        // Negative fraction is nonsensical at the spec layer (no anti-heal),
        // but the bitcast contract still has to round-trip. Confirms the
        // `as i16 as u32` widen sets the high u32 bits per i16's sign so
        // GPU `bitcast<i32>(payload_b)` recovers `-1` losslessly.
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::LifeSteal { duration_ticks: 10, fraction_q8: -1 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.effect_kinds[0], 18);
        assert_eq!(p.effect_payload_a[0], 10);
        // -1 as i16 sign-extended into u32 == 0xFFFFFFFF.
        assert_eq!(p.effect_payload_b[0], 0xFFFF_FFFF);
    }

    #[test]
    fn pack_damage_modify_payload() {
        // `damage_modify 1.5 (3s = 30 ticks)` → multiplier_q8 = 384 (1.5 * 256).
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::DamageModify { duration_ticks: 30, multiplier_q8: 384 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // DamageModify discriminant == 19.
        assert_eq!(p.effect_kinds[0], 19);
        assert_eq!(p.effect_payload_a[0], 30);
        // 384 fits as a positive i16 → u32 == 384.
        assert_eq!(p.effect_payload_b[0], 384);
    }

    #[test]
    fn pack_damage_modify_payload_sign_extends_negative_q8() {
        // Sign-preservation guard for the multiplier bitcast.
        let prog = AbilityProgram::new_single_target(
            0.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::DamageModify { duration_ticks: 5, multiplier_q8: -64 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.effect_kinds[0], 19);
        assert_eq!(p.effect_payload_a[0], 5);
        // -64 as i16 sign-extended into u32 == 0xFFFFFFC0.
        assert_eq!(p.effect_payload_b[0], 0xFFFF_FFC0);
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
        assert_eq!(p1.stackings, p2.stackings);
        assert_eq!(p1.chances, p2.chances);
        assert_eq!(p1.lifetime_kinds, p2.lifetime_kinds);
        assert_eq!(p1.lifetime_payloads, p2.lifetime_payloads);
        assert_eq!(p1.area_kinds, p2.area_kinds);
        assert_eq!(p1.area_args, p2.area_args);
        assert_eq!(p1.scaling_stat_refs, p2.scaling_stat_refs);
        assert_eq!(p1.scaling_percents, p2.scaling_percents);
    }

    // -- Wave 1.5#3 — stacking-mode pack tests. The `stackings` column
    // mirrors `effect_kinds`'s row-major layout (stride =
    // MAX_EFFECTS_PER_PROGRAM); slots without a `stacking <mode>`
    // modifier carry `STACKING_NONE_SENTINEL` (0xFF). Apply handlers
    // treat the sentinel as `StackingMode::Refresh`. -----------------
    #[test]
    fn pack_stackings_default_all_sentinel() {
        // No effect carries `stacking <mode>`; every column slot must be
        // the none-sentinel — including the empty effect tail (slots
        // 1..MAX_EFFECTS_PER_PROGRAM). This guards the pre-fill path so
        // the apply-handler doesn't need to special-case empty slots.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.stackings.len(), MAX_EFFECTS_PER_PROGRAM);
        for (i, s) in p.stackings.iter().enumerate() {
            assert_eq!(
                *s, STACKING_NONE_SENTINEL,
                "slot {i} must default to STACKING_NONE_SENTINEL",
            );
        }
    }

    #[test]
    fn pack_stackings_per_effect_override() {
        // Two effects: slot 0 carries `stacking stack`, slot 1 has no
        // stacking modifier. Empty slots 2..MAX stay at the sentinel.
        // Discriminants: Refresh=0, Stack=1, Extend=2.
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 10.0 },
                EffectOp::Stun { duration_ticks: 20 },
            ],
        );
        let mut sv: SmallVec<[Option<StackingMode>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        sv.push(Some(StackingMode::Stack));
        sv.push(None);
        prog.stackings = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.stackings.len(), MAX_EFFECTS_PER_PROGRAM);
        // Stack discriminant == 1.
        assert_eq!(p.stackings[0], 1, "slot 0 has `stacking stack`");
        assert_eq!(p.stackings[1], STACKING_NONE_SENTINEL, "slot 1 has no modifier");
        for i in 2..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(
                p.stackings[i], STACKING_NONE_SENTINEL,
                "empty slot {i} must stay at sentinel",
            );
        }
    }

    // -- Wave 1.5#5 — chance-modifier pack tests. The `chances` column
    // mirrors `effect_kinds`'s row-major layout (stride =
    // MAX_EFFECTS_PER_PROGRAM); slots without a `chance N%` modifier
    // carry `CHANCE_NONE_SENTINEL` (0xFFFF). Apply handlers treat the
    // sentinel as "always fires". Q16 fixed-point: `0..=65534` encode
    // `0%..~100%`; `65535` reserved as the sentinel. ----------------
    #[test]
    fn pack_chances_default_all_sentinel() {
        // No effect carries `chance N%`; every column slot must be the
        // none-sentinel — including the empty effect tail (slots
        // 1..MAX_EFFECTS_PER_PROGRAM). This guards the pre-fill path so
        // apply handlers don't need to special-case empty slots.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.chances.len(), MAX_EFFECTS_PER_PROGRAM);
        for (i, c) in p.chances.iter().enumerate() {
            assert_eq!(
                *c, CHANCE_NONE_SENTINEL,
                "slot {i} must default to CHANCE_NONE_SENTINEL",
            );
        }
    }

    #[test]
    fn pack_chances_per_effect_override() {
        // Two effects: slot 0 carries `chance 25%` (q16 = 16384), slot
        // 1 has no chance modifier. Empty slots 2..MAX stay at the
        // sentinel.
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [
                EffectOp::Damage { amount: 10.0 },
                EffectOp::Stun { duration_ticks: 20 },
            ],
        );
        let mut sv: SmallVec<[Option<u16>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        // 0.25 * 65534 = 16383.5 → 16384 after round.
        sv.push(Some(16384));
        sv.push(None);
        prog.chances = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.chances.len(), MAX_EFFECTS_PER_PROGRAM);
        assert_eq!(p.chances[0], 16384, "slot 0 has `chance 25%`");
        assert_eq!(p.chances[1], CHANCE_NONE_SENTINEL, "slot 1 has no modifier");
        for i in 2..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(
                p.chances[i], CHANCE_NONE_SENTINEL,
                "empty slot {i} must stay at sentinel",
            );
        }
    }

    // -- Wave 1.5#8 — lifetime-modifier pack tests. The `lifetime_kinds`
    // + `lifetime_payloads` pair mirrors `effect_kinds`'s row-major
    // layout (stride = MAX_EFFECTS_PER_PROGRAM); slots without a
    // lifetime modifier carry `LIFETIME_KIND_NONE_SENTINEL` (0xFF) and
    // `0.0`. Apply handlers treat the sentinel as "effect persists
    // indefinitely (or for the verb's own duration if the verb has
    // one)". `LifetimeMode::DamageableHp` is the first per-effect SoA
    // modifier carrying variant data — its hp pool lives in the
    // companion `lifetime_payloads` column. -----------------
    #[test]
    fn pack_lifetimes_default_all_sentinel() {
        // No effect carries a lifetime modifier; every kind slot must
        // be the none-sentinel + every payload slot `0.0` — including
        // the empty effect tail (slots 1..MAX_EFFECTS_PER_PROGRAM). This
        // guards the pre-fill path so apply handlers don't need to
        // special-case empty slots.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.lifetime_kinds.len(), MAX_EFFECTS_PER_PROGRAM);
        assert_eq!(p.lifetime_payloads.len(), MAX_EFFECTS_PER_PROGRAM);
        for (i, k) in p.lifetime_kinds.iter().enumerate() {
            assert_eq!(
                *k, LIFETIME_KIND_NONE_SENTINEL,
                "slot {i} kind must default to LIFETIME_KIND_NONE_SENTINEL",
            );
        }
        for (i, v) in p.lifetime_payloads.iter().enumerate() {
            assert_eq!(*v, 0.0, "slot {i} payload must default to 0.0");
        }
    }

    #[test]
    fn pack_lifetimes_until_caster_dies() {
        // Single effect carries `until_caster_dies` (discriminant 0).
        // Payload is `0.0` (variant has no payload). Empty tail slots
        // stay at the sentinel + `0.0`.
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Shield { amount: 50.0 }],
        );
        let mut sv: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        sv.push(Some(LifetimeMode::UntilCasterDies));
        prog.lifetimes = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // UntilCasterDies discriminant == 0.
        assert_eq!(p.lifetime_kinds[0], 0, "slot 0 kind = UntilCasterDies");
        assert_eq!(p.lifetime_payloads[0], 0.0, "UntilCasterDies has no payload");
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(
                p.lifetime_kinds[i], LIFETIME_KIND_NONE_SENTINEL,
                "empty slot {i} kind must stay at sentinel",
            );
            assert_eq!(
                p.lifetime_payloads[i], 0.0,
                "empty slot {i} payload must stay at 0.0",
            );
        }
    }

    #[test]
    fn pack_lifetimes_damageable_hp_carries_payload() {
        // Single effect carries `damageable_hp(100.0)` (discriminant 1)
        // — the FIRST per-effect SoA modifier with variant data, so
        // both columns must round-trip.
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Shield { amount: 50.0 }],
        );
        let mut sv: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        sv.push(Some(LifetimeMode::DamageableHp(100.0)));
        prog.lifetimes = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // DamageableHp discriminant == 1.
        assert_eq!(p.lifetime_kinds[0], 1, "slot 0 kind = DamageableHp");
        assert_eq!(p.lifetime_payloads[0], 100.0, "DamageableHp payload = 100.0");
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.lifetime_kinds[i], LIFETIME_KIND_NONE_SENTINEL);
            assert_eq!(p.lifetime_payloads[i], 0.0);
        }
    }

    #[test]
    fn pack_lifetimes_break_on_damage_no_payload() {
        // Single effect carries `break_on_damage` (discriminant 2)
        // — payload column stays `0.0`. Two-effect program with the
        // second effect un-modified guards the per-slot independence
        // of the kinds + payloads columns.
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [
                EffectOp::Shield { amount: 50.0 },
                EffectOp::Damage { amount: 20.0 },
            ],
        );
        let mut sv: SmallVec<[Option<LifetimeMode>; MAX_EFFECTS_PER_PROGRAM]> = SmallVec::new();
        sv.push(Some(LifetimeMode::BreakOnDamage));
        sv.push(None);
        prog.lifetimes = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // BreakOnDamage discriminant == 2.
        assert_eq!(p.lifetime_kinds[0], 2, "slot 0 kind = BreakOnDamage");
        assert_eq!(p.lifetime_payloads[0], 0.0, "BreakOnDamage has no payload");
        assert_eq!(
            p.lifetime_kinds[1], LIFETIME_KIND_NONE_SENTINEL,
            "slot 1 has no lifetime modifier",
        );
        assert_eq!(p.lifetime_payloads[1], 0.0);
        for i in 2..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.lifetime_kinds[i], LIFETIME_KIND_NONE_SENTINEL);
            assert_eq!(p.lifetime_payloads[i], 0.0);
        }
    }

    // -- Wave 1.5#2 — `in <shape>(args)` modifier pack tests. The
    // `area_kinds` + `area_args` pair mirrors `effect_kinds`'s
    // row-major layout (stride = MAX_EFFECTS_PER_PROGRAM for kinds;
    // stride = MAX_EFFECTS_PER_PROGRAM * 4 for args). Slots without
    // an `in <shape>` modifier carry `SHAPE_KIND_NONE_SENTINEL`
    // (0xFF) and `[0.0; 4]`. Apply handlers treat the sentinel as
    // "single-target effect (use program.area)". `EffectAreaShape` is
    // the second per-effect SoA modifier with variant data — its
    // 4×f32 args block lives in the companion `area_args` column.
    // Spec §8 catalog: 5 disc-family (Circle/Cone/Line/Ring/Spread =
    // 0..=4) + 7 volume-family (Box/Sphere/Column/Wall/Cylinder/
    // Dome/Hull = 5..=11). -------------------------------------------
    #[test]
    fn pack_areas_default_all_sentinel() {
        // No effect carries `in <shape>`; every kind slot must be the
        // none-sentinel + every args slot `0.0` — including the empty
        // effect tail (slots 1..MAX_EFFECTS_PER_PROGRAM). This guards
        // the pre-fill path so apply handlers don't need to special-
        // case empty slots.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        assert_eq!(p.area_kinds.len(), MAX_EFFECTS_PER_PROGRAM);
        assert_eq!(p.area_args.len(), MAX_EFFECTS_PER_PROGRAM * 4);
        for (i, k) in p.area_kinds.iter().enumerate() {
            assert_eq!(
                *k, SHAPE_KIND_NONE_SENTINEL,
                "slot {i} kind must default to SHAPE_KIND_NONE_SENTINEL",
            );
        }
        for (i, v) in p.area_args.iter().enumerate() {
            assert_eq!(*v, 0.0, "slot {i} arg must default to 0.0");
        }
    }

    #[test]
    fn pack_areas_circle_one_arg() {
        // Single effect carries `in circle(2.5)` (discriminant 0).
        // Args [2.5, 0, 0, 0]. Empty tail slots stay at sentinel +
        // `[0.0; 4]`.
        use crate::ability::program::{EffectAreaShape, ShapeKind};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        let mut sv: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
            SmallVec::new();
        sv.push(Some(EffectAreaShape {
            kind: ShapeKind::Circle,
            args: [2.5, 0.0, 0.0, 0.0],
        }));
        prog.per_effect_areas = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Circle discriminant == 0.
        assert_eq!(p.area_kinds[0], 0, "slot 0 kind = Circle");
        assert_eq!(p.area_args[0], 2.5, "slot 0 arg 0 = radius");
        assert_eq!(p.area_args[1], 0.0);
        assert_eq!(p.area_args[2], 0.0);
        assert_eq!(p.area_args[3], 0.0);
        for i in 1..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.area_kinds[i], SHAPE_KIND_NONE_SENTINEL);
            for j in 0..4 {
                assert_eq!(p.area_args[i * 4 + j], 0.0);
            }
        }
    }

    #[test]
    fn pack_areas_cone_two_args() {
        // `in cone(8.0, 45.0)` (discriminant 1) — args [r, angle, 0, 0].
        use crate::ability::program::{EffectAreaShape, ShapeKind};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        let mut sv: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
            SmallVec::new();
        sv.push(Some(EffectAreaShape {
            kind: ShapeKind::Cone,
            args: [8.0, 45.0, 0.0, 0.0],
        }));
        prog.per_effect_areas = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Cone discriminant == 1.
        assert_eq!(p.area_kinds[0], 1, "slot 0 kind = Cone");
        assert_eq!(p.area_args[0], 8.0, "slot 0 arg 0 = radius");
        assert_eq!(p.area_args[1], 45.0, "slot 0 arg 1 = angle_deg");
        assert_eq!(p.area_args[2], 0.0);
        assert_eq!(p.area_args[3], 0.0);
    }

    #[test]
    fn pack_areas_wall_four_args() {
        // `Wall` is the only shape that consumes ALL four args
        // (len/h/thick/facing). Round-trip exercises every column slot.
        use crate::ability::program::{EffectAreaShape, ShapeKind};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        let mut sv: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
            SmallVec::new();
        sv.push(Some(EffectAreaShape {
            kind: ShapeKind::Wall,
            args: [10.0, 3.0, 0.5, 90.0],
        }));
        prog.per_effect_areas = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Wall discriminant == 8.
        assert_eq!(p.area_kinds[0], 8, "slot 0 kind = Wall");
        assert_eq!(p.area_args[0], 10.0,  "slot 0 arg 0 = len");
        assert_eq!(p.area_args[1], 3.0,   "slot 0 arg 1 = h");
        assert_eq!(p.area_args[2], 0.5,   "slot 0 arg 2 = thick");
        assert_eq!(p.area_args[3], 90.0,  "slot 0 arg 3 = facing_deg");
    }

    #[test]
    fn pack_areas_per_effect_independence() {
        // Two-effect program: slot 0 has `in sphere(3.0)`, slot 1 has
        // no shape modifier. Empty slots 2..MAX stay at the sentinel +
        // `[0.0; 4]`. Guards per-slot independence of the kind +
        // args columns parallel to `pack_lifetimes_break_on_damage_no_payload`.
        use crate::ability::program::{EffectAreaShape, ShapeKind};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [
                EffectOp::Damage { amount: 30.0 },
                EffectOp::Heal   { amount: 10.0 },
            ],
        );
        let mut sv: SmallVec<[Option<EffectAreaShape>; MAX_EFFECTS_PER_PROGRAM]> =
            SmallVec::new();
        sv.push(Some(EffectAreaShape {
            kind: ShapeKind::Sphere,
            args: [3.0, 0.0, 0.0, 0.0],
        }));
        sv.push(None);
        prog.per_effect_areas = sv;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Sphere discriminant == 6.
        assert_eq!(p.area_kinds[0], 6, "slot 0 kind = Sphere");
        assert_eq!(p.area_args[0], 3.0, "slot 0 radius");
        assert_eq!(p.area_args[1], 0.0);
        assert_eq!(p.area_args[2], 0.0);
        assert_eq!(p.area_args[3], 0.0);
        assert_eq!(
            p.area_kinds[1], SHAPE_KIND_NONE_SENTINEL,
            "slot 1 has no shape modifier",
        );
        for j in 0..4 {
            assert_eq!(p.area_args[1 * 4 + j], 0.0);
        }
        for i in 2..MAX_EFFECTS_PER_PROGRAM {
            assert_eq!(p.area_kinds[i], SHAPE_KIND_NONE_SENTINEL);
            for j in 0..4 {
                assert_eq!(p.area_args[i * 4 + j], 0.0);
            }
        }
    }

    // -- Wave 1.5#4 — `+ N% stat_ref` modifier pack tests. The
    // `scaling_stat_refs` + `scaling_percents` pair mirrors
    // `effect_kinds`'s row-major layout but adds an inner stride of
    // `MAX_SCALINGS_PER_EFFECT` (2 today) so multiple scalings per
    // effect can be addressed (e.g. `damage 50 + 30% AP + 20% AD`).
    // Slots without a scaling carry `SCALING_STAT_NONE_SENTINEL`
    // (0xFF) and `0.0`. Apply handlers treat the sentinel as "no
    // scaling for this slot — flat amount only". Stat-ref ordinals:
    // AttackDamage=0, AbilityPower=1, MaxHp=2, Hp=3, Armor=4,
    // MagicResist=5, MoveSpeed=6, Mana=7. -----------------------
    #[test]
    fn pack_scalings_default_all_sentinel() {
        // No effect carries `+ N% stat_ref`; every stat-ref slot must
        // be the none-sentinel + every percent slot `0.0` — including
        // the empty effect tail (slots 1..MAX_EFFECTS_PER_PROGRAM) and
        // every inner scaling slot (0..MAX_SCALINGS_PER_EFFECT). This
        // guards the pre-fill path so apply handlers don't need to
        // special-case empty slots.
        let prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [EffectOp::Damage { amount: 10.0 }],
        );
        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        let total = MAX_EFFECTS_PER_PROGRAM * MAX_SCALINGS_PER_EFFECT;
        assert_eq!(p.scaling_stat_refs.len(), total);
        assert_eq!(p.scaling_percents.len(), total);
        for (i, s) in p.scaling_stat_refs.iter().enumerate() {
            assert_eq!(
                *s, SCALING_STAT_NONE_SENTINEL,
                "slot {i} stat-ref must default to SCALING_STAT_NONE_SENTINEL",
            );
        }
        for (i, v) in p.scaling_percents.iter().enumerate() {
            assert_eq!(*v, 0.0, "slot {i} percent must default to 0.0");
        }
    }

    #[test]
    fn pack_scalings_single_on_first_effect() {
        // Single effect carries one scaling: `+ 30% AP` →
        // (AbilityPower=1, 0.30). The first inner slot (effect 0,
        // scaling 0) must hold the discriminant + percent; every
        // other slot stays at sentinel + `0.0`.
        use crate::ability::program::{EffectScaling, ScalingStatRef};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        let mut outer: SmallVec<
            [SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
        > = SmallVec::new();
        let mut inner: SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]> = SmallVec::new();
        inner.push(EffectScaling { stat_ref: ScalingStatRef::AbilityPower, percent: 0.30 });
        outer.push(inner);
        prog.scalings_per_effect = outer;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // AbilityPower discriminant == 1.
        assert_eq!(p.scaling_stat_refs[0], 1, "effect 0 scaling 0 = AbilityPower");
        assert_eq!(p.scaling_percents[0], 0.30, "effect 0 scaling 0 percent");
        // Inner slot 1 of effect 0 (unused) stays at sentinel + 0.0.
        assert_eq!(p.scaling_stat_refs[1], SCALING_STAT_NONE_SENTINEL);
        assert_eq!(p.scaling_percents[1], 0.0);
        // Effects 1..MAX stay at sentinel + 0.0.
        for eff_i in 1..MAX_EFFECTS_PER_PROGRAM {
            for sc_i in 0..MAX_SCALINGS_PER_EFFECT {
                let off = eff_i * MAX_SCALINGS_PER_EFFECT + sc_i;
                assert_eq!(
                    p.scaling_stat_refs[off], SCALING_STAT_NONE_SENTINEL,
                    "effect {eff_i} scaling {sc_i} stat-ref must stay at sentinel",
                );
                assert_eq!(
                    p.scaling_percents[off], 0.0,
                    "effect {eff_i} scaling {sc_i} percent must stay at 0.0",
                );
            }
        }
    }

    #[test]
    fn pack_scalings_two_on_first_effect() {
        // Single effect carries two scalings: `+ 30% AP + 20% AD` →
        // [(AbilityPower=1, 0.30), (AttackDamage=0, 0.20)]. Both inner
        // slots of effect 0 must round-trip; slot 0 of effects 1..MAX
        // and any inner padding stay at sentinel + 0.0.
        use crate::ability::program::{EffectScaling, ScalingStatRef};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 50.0 }],
        );
        let mut outer: SmallVec<
            [SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
        > = SmallVec::new();
        let mut inner: SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]> = SmallVec::new();
        inner.push(EffectScaling { stat_ref: ScalingStatRef::AbilityPower, percent: 0.30 });
        inner.push(EffectScaling { stat_ref: ScalingStatRef::AttackDamage, percent: 0.20 });
        outer.push(inner);
        prog.scalings_per_effect = outer;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Inner slot 0: AbilityPower (1) / 0.30.
        assert_eq!(p.scaling_stat_refs[0], 1);
        assert_eq!(p.scaling_percents[0], 0.30);
        // Inner slot 1: AttackDamage (0) / 0.20.
        assert_eq!(p.scaling_stat_refs[1], 0);
        assert_eq!(p.scaling_percents[1], 0.20);
        // Every other effect slot stays at sentinel + 0.0.
        for eff_i in 1..MAX_EFFECTS_PER_PROGRAM {
            for sc_i in 0..MAX_SCALINGS_PER_EFFECT {
                let off = eff_i * MAX_SCALINGS_PER_EFFECT + sc_i;
                assert_eq!(p.scaling_stat_refs[off], SCALING_STAT_NONE_SENTINEL);
                assert_eq!(p.scaling_percents[off], 0.0);
            }
        }
    }

    #[test]
    fn pack_scalings_per_effect_independence() {
        // Two-effect program: effect 0 has `+ 25% AD`, effect 1 has no
        // scaling. Empty effects 2..MAX stay at sentinel + 0.0. Guards
        // outer-slot independence parallel to
        // `pack_areas_per_effect_independence`.
        use crate::ability::program::{EffectScaling, ScalingStatRef};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
            [
                EffectOp::Damage { amount: 30.0 },
                EffectOp::Heal   { amount: 10.0 },
            ],
        );
        let mut outer: SmallVec<
            [SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
        > = SmallVec::new();
        let mut inner0: SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]> = SmallVec::new();
        inner0.push(EffectScaling { stat_ref: ScalingStatRef::AttackDamage, percent: 0.25 });
        outer.push(inner0);
        // Effect 1 — empty inner SmallVec (no scaling).
        outer.push(SmallVec::new());
        prog.scalings_per_effect = outer;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // Effect 0 inner slot 0: AttackDamage (0) / 0.25.
        assert_eq!(p.scaling_stat_refs[0], 0, "effect 0 scaling 0 = AttackDamage");
        assert_eq!(p.scaling_percents[0], 0.25);
        // Effect 0 inner slot 1 (unused): sentinel + 0.0.
        assert_eq!(p.scaling_stat_refs[1], SCALING_STAT_NONE_SENTINEL);
        assert_eq!(p.scaling_percents[1], 0.0);
        // Effect 1 (empty inner): both inner slots sentinel + 0.0.
        for sc_i in 0..MAX_SCALINGS_PER_EFFECT {
            let off = 1 * MAX_SCALINGS_PER_EFFECT + sc_i;
            assert_eq!(
                p.scaling_stat_refs[off], SCALING_STAT_NONE_SENTINEL,
                "effect 1 (no scaling) scaling {sc_i} stat-ref must be sentinel",
            );
            assert_eq!(p.scaling_percents[off], 0.0);
        }
        // Effects 2..MAX (empty effect tail).
        for eff_i in 2..MAX_EFFECTS_PER_PROGRAM {
            for sc_i in 0..MAX_SCALINGS_PER_EFFECT {
                let off = eff_i * MAX_SCALINGS_PER_EFFECT + sc_i;
                assert_eq!(p.scaling_stat_refs[off], SCALING_STAT_NONE_SENTINEL);
                assert_eq!(p.scaling_percents[off], 0.0);
            }
        }
    }

    #[test]
    fn pack_scalings_percent_edge_case_above_one() {
        // Percent > 1.0 (e.g. `+ 150% MaxHP` → 1.5) must round-trip
        // unchanged — apply handlers will multiply
        // `target.max_hp * 1.5` and add to the flat amount. Guards
        // that the f32 percent column does no clamping.
        use crate::ability::program::{EffectScaling, ScalingStatRef};
        use smallvec::SmallVec;
        let mut prog = AbilityProgram::new_single_target(
            5.0,
            Gate { cooldown_ticks: 0, hostile_only: true, line_of_sight: false },
            [EffectOp::Damage { amount: 20.0 }],
        );
        let mut outer: SmallVec<
            [SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]>; MAX_EFFECTS_PER_PROGRAM],
        > = SmallVec::new();
        let mut inner: SmallVec<[EffectScaling; MAX_SCALINGS_PER_EFFECT]> = SmallVec::new();
        inner.push(EffectScaling { stat_ref: ScalingStatRef::MaxHp, percent: 1.5 });
        outer.push(inner);
        prog.scalings_per_effect = outer;

        let reg = build(vec![prog]);
        let p = PackedAbilityRegistry::pack(&reg);

        // MaxHp discriminant == 2.
        assert_eq!(p.scaling_stat_refs[0], 2, "stat-ref = MaxHp");
        assert_eq!(p.scaling_percents[0], 1.5, "percent must round-trip 1.5");
    }
}
