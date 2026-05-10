//! Plan G G2.5 — interrupt detection primitives.
//!
//! Engine-side mirror of the AST `InterruptKind` enum (see
//! `crates/dsl_ast/src/ast.rs::InterruptKind`). The AST holds the
//! interrupts metadata at parse time; this module defines the
//! POD bitmask + decision function the busy-resolution kernel
//! consults at runtime to decide whether a Damaged / Stunned / etc.
//! event interrupts an in-progress cast.
//!
//! ## Bit layout
//!
//! [`InterruptKind`]'s discriminant is the bit position in the
//! [`InterruptMask`]. The `Standard` set (= the engine-wide default
//! per design 2026-05-09) is the union of `Damage | Stun |
//! CasterDied | TargetDied`. `Movement` is opt-in only.
//!
//! ```text
//! bit 0  Damage
//! bit 1  Stun
//! bit 2  CasterDied
//! bit 3  TargetDied
//! bit 4  Movement
//!
//! STANDARD = 0b00001111 = 15
//! NONE     = 0
//! ALL      = 0b00011111 = 31
//! ```
//!
//! ## P11 determinism
//!
//! `should_interrupt` is a pure function of `(busy_state, event_kind,
//! current_tick)`; no RNG, no hidden state. The order events arrive
//! in still matters at the consumer level (multiple interrupts in
//! one tick all clear the same busy state once), but the per-event
//! decision is order-independent.

use crate::ids::AgentId;

/// Sources that can interrupt a cast / busy activity. Engine-side
/// POD mirror of `dsl_ast::ast::InterruptKind`. Discriminants encode
/// bit positions in [`InterruptMask`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum InterruptKind {
    /// Caster took a `Damaged` event this tick.
    Damage = 0,
    /// Caster received a `Stunned` event this tick.
    Stun = 1,
    /// Caster died (`Defeated`).
    CasterDied = 2,
    /// The cast's target died — no-op for AOE / self-cast / position-target casts.
    TargetDied = 3,
    /// Caster moved (pre-tick / post-tick `agent_pos` differ).
    Movement = 4,
}

impl InterruptKind {
    /// Bit position in the packed [`InterruptMask`].
    #[inline]
    pub const fn bit(self) -> u8 {
        self as u8
    }

    /// Single-kind mask (one bit set).
    #[inline]
    pub const fn as_mask(self) -> InterruptMask {
        InterruptMask(1u8 << (self as u8))
    }
}

/// Packed bitmask of interrupt kinds. `u8` carries 8 bits, room for
/// 5 today + 3 reserved for future kinds (Cancel, Charm, Suppress
/// candidates per the Plan G plan doc — none implemented yet).
///
/// Constructed via [`InterruptMask::standard`] / [`InterruptMask::none`]
/// / [`InterruptMask::all`] / `InterruptKind::as_mask` + bitwise ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct InterruptMask(pub u8);

impl InterruptMask {
    /// `interrupts: none` — the cast is uninterruptible. Used for
    /// Bind Soul-style casts that must complete regardless of caster
    /// state.
    pub const NONE: Self = Self(0);

    /// `interrupts: standard` — the engine-wide default per design
    /// 2026-05-09: `{ Damage, Stun, CasterDied, TargetDied }`.
    /// Movement is OPT-IN only (not part of standard).
    pub const STANDARD: Self =
        Self((1 << InterruptKind::Damage as u8)
            | (1 << InterruptKind::Stun as u8)
            | (1 << InterruptKind::CasterDied as u8)
            | (1 << InterruptKind::TargetDied as u8));

    /// All known kinds (Damage | Stun | CasterDied | TargetDied | Movement).
    pub const ALL: Self =
        Self((1 << InterruptKind::Damage as u8)
            | (1 << InterruptKind::Stun as u8)
            | (1 << InterruptKind::CasterDied as u8)
            | (1 << InterruptKind::TargetDied as u8)
            | (1 << InterruptKind::Movement as u8));

    #[inline]
    pub const fn none() -> Self { Self::NONE }
    #[inline]
    pub const fn standard() -> Self { Self::STANDARD }
    #[inline]
    pub const fn all() -> Self { Self::ALL }

    /// Test whether `kind` is in the set.
    #[inline]
    pub const fn contains(self, kind: InterruptKind) -> bool {
        (self.0 & (1u8 << kind as u8)) != 0
    }

    /// Set `kind` in the mask (returns a new mask; original unchanged).
    #[inline]
    pub const fn with(self, kind: InterruptKind) -> Self {
        Self(self.0 | (1u8 << kind as u8))
    }

    /// Clear `kind` from the mask.
    #[inline]
    pub const fn without(self, kind: InterruptKind) -> Self {
        Self(self.0 & !(1u8 << kind as u8))
    }

    /// Raw u8 — the SoA encoding used by the busy-resolution kernel.
    #[inline]
    pub const fn raw(self) -> u8 {
        self.0
    }
}

/// Snapshot of an agent's busy state, as seen by the
/// busy-resolution / interrupt-detection kernel. Mirrors the four
/// busy SoA columns added in Plan G G2.1 +
/// [`InterruptMask`] derived from the casting ability's spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BusyState {
    /// `agents.busy_until_tick(self)`. `0` = idle.
    pub busy_until_tick: u32,
    /// `agents.busy_with_ability_id(self)`. `0` = idle.
    pub busy_with_ability_id: u32,
    /// `agents.busy_target_slot(self)`. `0` for self-cast / AOE.
    pub busy_target_slot: u32,
    /// Interrupt mask the casting ability declared (`interrupts: …`
    /// at parse time, lowered to a u8 bitmask by G2.5's lowering
    /// addition). Per-fixture sim consumers stamp this onto a new
    /// SoA column in the same shape as the other three.
    pub interrupt_mask: InterruptMask,
}

impl BusyState {
    /// Convenience: is the agent currently mid-cast at `tick`?
    #[inline]
    pub const fn is_busy_at(&self, tick: u32) -> bool {
        self.busy_until_tick > 0 && tick < self.busy_until_tick
    }
}

/// Decide whether an interrupt-source event cancels an in-progress
/// cast. Returns `true` when ALL of:
///
///   * The agent is currently busy at `current_tick` (cast hasn't
///     resolved yet — interrupting AT the resolve tick is too late;
///     the busy-resolution kernel has already fired).
///   * `kind` is in the cast's `interrupt_mask`.
///
/// The caller (per-fixture sim rule today, registry-driven dispatcher
/// later) is responsible for:
///   * Filtering events to the right TARGET (Damaged at the busy
///     agent — for Damage / Stun / CasterDied; Damaged at the busy
///     agent's TARGET — for TargetDied).
///   * Clearing busy SoA + emitting `CastInterrupted` chronicle.
///
/// **Movement edge case.** `InterruptKind::Movement` requires the
/// caller to have already detected position drift (pre-tick vs
/// post-tick `agent_pos`). This function only checks the mask; it
/// does NOT distinguish "moved by 0.001 because of jitter" from
/// "moved 5m to break the channel". The fixture decides the
/// threshold.
///
/// **`TargetDied` edge case.** For self-cast or AOE / position-
/// target casts the cast has no agent target. Callers MUST guard
/// `kind == TargetDied` against `busy_target_slot == 0` before
/// invoking this function — passing `TargetDied` for a self-cast
/// fires a false-positive interrupt here.
#[inline]
pub fn should_interrupt(
    busy:        BusyState,
    kind:        InterruptKind,
    current_tick: u32,
) -> bool {
    busy.is_busy_at(current_tick) && busy.interrupt_mask.contains(kind)
}

/// Per-fixture / per-tick interrupt event accumulator. Useful when
/// MULTIPLE interrupt-source events hit the same busy agent in one
/// tick (e.g. caster takes Damaged + Stunned in the same tick) —
/// only ONE `CastInterrupted` chronicle should fire. The first
/// matching kind wins; later events on the same agent are no-ops.
///
/// Today's per-fixture sim consumers do this naturally (the busy
/// state read in the second event sees `busy_until_tick == 0`
/// after the first interrupt cleared it), but the helper is here
/// for the registry-driven dispatcher slice.
#[derive(Debug, Default)]
pub struct InterruptDecider {
    decided: Vec<AgentId>,
}

impl InterruptDecider {
    pub fn new() -> Self { Self::default() }

    /// Returns `Some(kind)` on the FIRST interrupt for `agent` in
    /// this batch; `None` for subsequent events on the same agent.
    /// Callers that don't need this dedup can call
    /// [`should_interrupt`] directly.
    pub fn try_interrupt(
        &mut self,
        agent: AgentId,
        busy:  BusyState,
        kind:  InterruptKind,
        tick:  u32,
    ) -> Option<InterruptKind> {
        if self.decided.contains(&agent) {
            return None;
        }
        if !should_interrupt(busy, kind, tick) {
            return None;
        }
        self.decided.push(agent);
        Some(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy_with(mask: InterruptMask, until: u32) -> BusyState {
        BusyState {
            busy_until_tick:      until,
            busy_with_ability_id: 1,
            busy_target_slot:     0,
            interrupt_mask:       mask,
        }
    }

    #[test]
    fn standard_mask_contains_damage_stun_caster_target_died() {
        let m = InterruptMask::standard();
        assert!(m.contains(InterruptKind::Damage));
        assert!(m.contains(InterruptKind::Stun));
        assert!(m.contains(InterruptKind::CasterDied));
        assert!(m.contains(InterruptKind::TargetDied));
        assert!(!m.contains(InterruptKind::Movement),
            "Movement is opt-in, not part of standard");
    }

    #[test]
    fn none_mask_contains_nothing() {
        let m = InterruptMask::none();
        assert!(!m.contains(InterruptKind::Damage));
        assert!(!m.contains(InterruptKind::Stun));
        assert!(!m.contains(InterruptKind::CasterDied));
        assert!(!m.contains(InterruptKind::TargetDied));
        assert!(!m.contains(InterruptKind::Movement));
        assert_eq!(m.raw(), 0);
    }

    #[test]
    fn all_mask_contains_every_kind() {
        let m = InterruptMask::all();
        for k in [
            InterruptKind::Damage, InterruptKind::Stun,
            InterruptKind::CasterDied, InterruptKind::TargetDied,
            InterruptKind::Movement,
        ] {
            assert!(m.contains(k), "ALL must contain {k:?}");
        }
    }

    #[test]
    fn standard_plus_movement_round_trips() {
        // `interrupts: standard + { movement }` per the AST grammar.
        let m = InterruptMask::standard().with(InterruptKind::Movement);
        assert!(m.contains(InterruptKind::Damage));
        assert!(m.contains(InterruptKind::Movement));
        assert_eq!(m, InterruptMask::all());
    }

    #[test]
    fn standard_minus_damage_round_trips() {
        // `interrupts: standard - { damage }` per the AST grammar
        // (forager keeps moving even under fire).
        let m = InterruptMask::standard().without(InterruptKind::Damage);
        assert!(!m.contains(InterruptKind::Damage));
        assert!(m.contains(InterruptKind::Stun));
        assert!(m.contains(InterruptKind::CasterDied));
        assert!(m.contains(InterruptKind::TargetDied));
    }

    #[test]
    fn should_interrupt_fires_when_busy_and_kind_in_mask() {
        let busy = busy_with(InterruptMask::standard(), /*until*/ 10);
        // current_tick = 5 → mid-cast, Damage in standard → interrupts.
        assert!(should_interrupt(busy, InterruptKind::Damage, 5));
    }

    #[test]
    fn should_interrupt_returns_false_when_idle() {
        let idle = BusyState {
            busy_until_tick:      0,
            busy_with_ability_id: 0,
            busy_target_slot:     0,
            interrupt_mask:       InterruptMask::standard(),
        };
        assert!(!should_interrupt(idle, InterruptKind::Damage, 5),
            "idle agent (busy_until_tick == 0) cannot be interrupted");
    }

    #[test]
    fn should_interrupt_returns_false_at_resolve_tick() {
        // The cast resolves on the EXACT tick busy_until_tick fires.
        // An interrupt arriving at the resolve tick is too late —
        // the busy-resolution kernel already picked the agent up.
        let busy = busy_with(InterruptMask::standard(), /*until*/ 10);
        assert!(!should_interrupt(busy, InterruptKind::Damage, 10),
            "interrupt at resolve tick is too late — kernel already fired");
    }

    #[test]
    fn should_interrupt_returns_false_when_kind_not_in_mask() {
        // `interrupts: standard` does NOT include Movement.
        let busy = busy_with(InterruptMask::standard(), /*until*/ 10);
        assert!(!should_interrupt(busy, InterruptKind::Movement, 5),
            "Movement is opt-in; not in standard");
    }

    #[test]
    fn should_interrupt_uninterruptible_cast_never_interrupts() {
        // `interrupts: none` — Bind Soul style.
        let busy = busy_with(InterruptMask::none(), /*until*/ 10);
        for k in [
            InterruptKind::Damage, InterruptKind::Stun,
            InterruptKind::CasterDied, InterruptKind::TargetDied,
            InterruptKind::Movement,
        ] {
            assert!(!should_interrupt(busy, k, 5),
                "InterruptMask::none() must reject {k:?}");
        }
    }

    #[test]
    fn interrupt_decider_dedups_per_agent_per_batch() {
        let mut d = InterruptDecider::new();
        let busy = busy_with(InterruptMask::standard(), 10);
        let a = AgentId::new(1).unwrap();
        // First interrupt fires.
        assert_eq!(d.try_interrupt(a, busy, InterruptKind::Damage, 5),
            Some(InterruptKind::Damage));
        // Second event on the SAME agent in the same batch is a no-op
        // (the per-fixture sim consumer already cleared busy state).
        assert_eq!(d.try_interrupt(a, busy, InterruptKind::Stun, 5), None);
    }

    #[test]
    fn interrupt_decider_per_agent_independent() {
        let mut d = InterruptDecider::new();
        let busy = busy_with(InterruptMask::standard(), 10);
        let a = AgentId::new(1).unwrap();
        let b = AgentId::new(2).unwrap();
        assert!(d.try_interrupt(a, busy, InterruptKind::Damage, 5).is_some());
        assert!(d.try_interrupt(b, busy, InterruptKind::Damage, 5).is_some(),
            "different agents have independent decisions");
    }

    #[test]
    fn bit_layout_is_stable() {
        // Pin the bit positions — the SoA encoding (u8 per agent in
        // `agents.busy_interrupt_mask` once that column lands) bakes
        // them into the GPU layout. A future renumber would silently
        // corrupt every saved snapshot.
        assert_eq!(InterruptKind::Damage.bit(),     0);
        assert_eq!(InterruptKind::Stun.bit(),       1);
        assert_eq!(InterruptKind::CasterDied.bit(), 2);
        assert_eq!(InterruptKind::TargetDied.bit(), 3);
        assert_eq!(InterruptKind::Movement.bit(),   4);
        assert_eq!(InterruptMask::STANDARD.raw(),   0b0000_1111);
        assert_eq!(InterruptMask::ALL.raw(),        0b0001_1111);
    }
}
