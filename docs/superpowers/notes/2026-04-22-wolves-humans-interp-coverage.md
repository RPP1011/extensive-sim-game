# Wolves+Humans IR Interpreter Coverage Inventory

Source: `assets/sim/{masks,scoring,physics,views,entities}.sim`
Generated: 2026-04-25

---

## Overview

This document bounds the interpreter's required coverage for P1b of the IR
interpreter plan (`docs/superpowers/plans/2026-04-22-ir-interpreter.md`). Only
constructs that appear in the wolves+humans fixture rule files are listed as
in-scope. Everything else is explicitly P1c+ scope.

---

## MaskIR variants used

Masks used by wolves+humans (all masks in `masks.sim` apply universally):

- `Hold` — `mask Hold when agents.alive(self)` (line 34)
  - `IrExpr::NamespaceCall { ns: Agents, method: "alive", args: [self] }`

- `MoveToward(target)` — `from query.nearby_agents(...)` (lines 47–49)
  - `IrActionHeadShape::Positional` with one positional param (target: AgentId)
  - `MaskIR::candidate_source`: `IrExpr::NamespaceCall { ns: Query, method: "nearby_agents", args: [agents.pos(self), config.movement.max_move_radius] }`
  - Predicate binary conjunction (BinOp::And):
    - `IrExpr::NamespaceCall { ns: Agents, method: "alive", args: [target] }`
    - `IrExpr::Binary(BinOp::NotEq, target, self)`

- `Flee` — `mask Flee when agents.alive(self)` (line 61)
  - Same as Hold: single `NamespaceCall { ns: Agents, method: "alive" }`

- `Eat`, `Drink`, `Rest` — all `when agents.alive(self)` (lines 67–69)
  - Same pattern as Hold/Flee.

- `Attack(target)` — `from query.nearby_agents(...)` with 3-conjunct predicate (lines 75–79)
  - `IrActionHeadShape::Positional` with one positional param (target: AgentId)
  - `MaskIR::candidate_source`: `IrExpr::NamespaceCall { ns: Query, method: "nearby_agents", args: [agents.pos(self), config.combat.attack_range] }`
  - Predicate:
    - `BinOp::And` chain over three sub-predicates:
      - `NamespaceCall { ns: Agents, method: "alive", args: [target] }`
      - `ViewCall(is_hostile, [self, target])` — calls `view is_hostile(a, b)`
      - `IrExpr::Binary(BinOp::Lt, BuiltinCall(Distance, [agents.pos(self), agents.pos(target)]), LitFloat(2.0))`

- `Cast(ability: AbilityId)` — caster-side predicate (lines 99–104)
  - `IrActionHeadShape::Positional` with one typed param `(ability: AbilityId)`
  - Predicate (BinOp::And chain over 5 sub-predicates):
    - `NamespaceCall { ns: Agents, method: "alive", args: [self] }`
    - `IrExpr::Unary(UnOp::Not, ViewCall(is_stunned, [self]))` — negated view call
    - `NamespaceCall { ns: Abilities, method: "known", args: [self, ability] }`
    - `NamespaceCall { ns: Abilities, method: "cooldown_ready", args: [self, ability] }`
    - `IrExpr::Binary(BinOp::Eq, NamespaceCall { ns: Agents, method: "engaged_with", args: [self] }, EnumVariant { ty: "Option", variant: "None" })`

### BinOp variants required by MaskIR
- `BinOp::And` — predicate conjunction (all multi-clause masks)
- `BinOp::NotEq` — `target != self` (MoveToward)
- `BinOp::Lt` — `distance(...) < 2.0` (Attack)
- `BinOp::Eq` — `agents.engaged_with(self) == None` (Cast)

### UnOp variants required by MaskIR
- `UnOp::Not` — `!view::is_stunned(self)` (Cast)

### IrExpr variants required by MaskIR
- `IrExpr::LitFloat` — `2.0` (Attack distance threshold)
- `IrExpr::LitBool` — not directly; `None` variant used as `EnumVariant`
- `IrExpr::EnumVariant` — `None` / `TargetSelector::Target` (Cast, cast physics)
- `IrExpr::NamespaceCall { ns: Agents, ... }` — `alive`, `pos`, `engaged_with`
- `IrExpr::NamespaceCall { ns: Query, ... }` — `nearby_agents`
- `IrExpr::NamespaceCall { ns: Abilities, ... }` — `known`, `cooldown_ready`
- `IrExpr::NamespaceField { ns: Config, field: "movement.max_move_radius" }` — config reads
- `IrExpr::NamespaceField { ns: Config, field: "combat.attack_range" }` — config reads
- `IrExpr::ViewCall(is_hostile, ...)` — lazy view call
- `IrExpr::ViewCall(is_stunned, ...)` — lazy view call
- `IrExpr::BuiltinCall(Builtin::Distance, ...)` — pairwise distance
- `IrExpr::Binary(...)` — all BinOp variants listed above
- `IrExpr::Unary(...)` — UnOp::Not

---

## ScoringIR variants used

The `scoring { ... }` block covers all `MicroKind` rows (lines 26–297 of
`scoring.sim`). The wolves+humans fixture exercises both trivial (constant-zero)
rows and live multi-modifier rows.

### Trivial rows (base literal only, never fire meaningfully)
- `Hold = 0.1` — `IrExpr::LitFloat(0.1)`
- `MoveToward = 0.3` — `IrExpr::LitFloat(0.3)`
- `Cast = 0.0`, `UseItem = 0.0`, `Harvest = 0.0`, `Drink = 0.0`, `Rest = 0.0`
- `PlaceTile = 0.0`, `PlaceVoxel = 0.0`, `HarvestVoxel = 0.0`
- `Converse = 0.0`, `ShareStory = 0.0`, `Communicate = 0.0`, `Ask = 0.0`, `Remember = 0.0`

### Active rows with modifiers

**Flee** (lines 111–116):
```
Flee = 0.0
  + (if self.hp < 30.0 { 0.6 } else { 0.0 })
  + (if self.hp < 50.0 { 0.4 } else { 0.0 })
  + (if self.hp_pct < 0.3 { 0.6 } else { 0.0 })
  + (view::threat_level(self, _) per_unit 0.01)
  + (if view::threat_level(self, _) > 50.0 { 0.3 } else { 0.0 })
  + (if view::kin_fear(self, _) > 0.5 { 0.4 } else { 0.0 })
```
- `IrExpr::Field { field_name: "hp" }` on `self` — raw hp read
- `IrExpr::Field { field_name: "hp_pct" }` on `self` — percentage hp read
- `IrExpr::If { cond, then_expr, else_expr }` — conditional modifier
- `IrExpr::LitFloat` — thresholds 30.0, 50.0, 0.3, 0.5, 50.0; deltas 0.6, 0.4, 0.3, 0.4, 0.0
- `IrExpr::PerUnit { expr, delta }` — `view::threat_level(self, _) per_unit 0.01` gradient modifier
- `IrExpr::ViewCall(threat_level, [self, Wildcard])` — materialized view call with wildcard second arg
- `IrExpr::ViewCall(kin_fear, [self, Wildcard])` — materialized view call with wildcard

**Attack(target)** (lines 231–238):
```
Attack(target) = 0.0
  + (if self.hp_pct >= 0.8 { 0.5 } else { 0.0 })
  + (if target.hp_pct < 0.3 { 0.4 } else { 0.0 })
  + (if target.hp_pct < 0.5 { 0.2 } else { 0.0 })
  + (view::threat_level(self, target) per_unit 0.01)
  + (if view::threat_level(self, target) > 20.0 { 0.3 } else { 0.0 })
  + (if view::my_enemies(self, target) > 0.5 { 0.4 } else { 0.0 })
  + (if view::pack_focus(self, target) > 0.5 { 0.4 } else { 0.0 })
  + (if view::rally_boost(self, _) > 0.3 { 0.3 } else { 0.0 })
```
- `IrExpr::Field { field_name: "hp_pct" }` on `self` and on `target` — target-side field read
- `IrExpr::ViewCall(threat_level, [self, target])` — specific-slot (non-wildcard) pair call
- `IrExpr::ViewCall(threat_level, [self, target])` with `per_unit 0.01` gradient
- `IrExpr::ViewCall(my_enemies, [self, target])`
- `IrExpr::ViewCall(pack_focus, [self, target])`
- `IrExpr::ViewCall(rally_boost, [self, Wildcard])` — wildcard second arg

**Eat** (line 260):
```
Eat = 0.0 + (if self.hp_pct < 0.3 { 0.8 } else { 0.0 })
```
- Same `IrExpr::Field { field_name: "hp_pct" }` + `IrExpr::If` pattern.

### BinOp variants required by ScoringIR
- `BinOp::Lt` — `self.hp < 30.0`, `hp_pct < 0.3`, etc.
- `BinOp::LtEq` — (not explicitly observed; `<` only in scoring.sim)
- `BinOp::GtEq` — `self.hp_pct >= 0.8`
- `BinOp::Gt` — `view::threat_level(...) > 50.0`, `> 20.0`, `> 0.5`
- `BinOp::Add` — the implicit accumulation of base + modifier deltas

### IrExpr variants required by ScoringIR
- `IrExpr::LitFloat` — all base values and thresholds
- `IrExpr::Field { base: self, field_name: "hp" }` — raw hp field on scoring agent
- `IrExpr::Field { base: self, field_name: "hp_pct" }` — pct field on scoring agent
- `IrExpr::Field { base: target, field_name: "hp_pct" }` — pct field on target (Attack row only)
- `IrExpr::If { cond, then_expr (LitFloat), else_expr (LitFloat(0.0)) }` — scoring modifier gate
- `IrExpr::PerUnit { expr: ViewCall, delta: LitFloat }` — gradient scoring modifier
- `IrExpr::ViewCall` with:
  - specific args `(self, target)` — threat_level, my_enemies, pack_focus
  - wildcard second arg `(self, _)` — threat_level (Flee), kin_fear, rally_boost
- `IrExpr::Binary(BinOp::Lt | Gt | GtEq, ...)` — comparisons in If conditions

### ScoringIR structure variants used
- `ScoringEntryIR` — standard per-agent rows (all rows in wolves+humans are standard)
- `IrActionHeadShape::None` — most rows (Hold, Flee, Eat, etc.)
- `IrActionHeadShape::Positional` — `Attack(target)` has one positional param

---

## PhysicsIR variants used

All rules in `physics.sim` apply in the wolves+humans fixture since they respond
to events in the standard combat flow.

### Statement variants used (`IrStmt`)

- `IrStmt::If { cond, then_body, else_body }` — guards on `agents.alive`, `a > 0.0`, `new_hp <= 0.0`, etc.
- `IrStmt::Let { name, value }` — local bindings: `let shield = agents.shield_hp(t)`, `let new_hp = max(...)`, etc.
- `IrStmt::Emit(IrEmit)` — emit events: `AgentAttacked`, `AgentDied`, `ChronicleEntry`, `EngagementCommitted`, `EngagementBroken`, etc.
- `IrStmt::For { binder, iter, body }` — `for op in abilities.effects(ab)`, `for kin in query.nearby_kin(dead, 12.0)`
- `IrStmt::Match { scrutinee, arms }` — `match op { Damage { amount } => ..., Heal { amount } => ..., ... }` (cast dispatch)
- `IrStmt::Expr(IrExprNode)` — `agents.kill(t)`, `agents.record_cast_cooldowns(caster, ab, t)`, etc. (side-effect calls as statements)

### IrExpr variants used in physics bodies
- `IrExpr::NamespaceCall { ns: Agents, method: "alive" | "shield_hp" | "hp" | "max_hp" | "set_hp" | "set_shield_hp" | "kill" | "attack_damage" | "stun_expires_at_tick" | "set_stun_expires_at_tick" | "slow_expires_at_tick" | "slow_factor_q8" | "set_slow_expires_at_tick" | "set_slow_factor_q8" | "engaged_with_or" | "clear_engaged_with" | "set_engaged_with" | "record_memory" | "record_cast_cooldowns" | "add_gold" | "sub_gold" | "adjust_standing" }` — state mutation + read calls
- `IrExpr::NamespaceCall { ns: Query, method: "nearest_hostile_to_or" | "nearby_kin" }` — spatial queries
- `IrExpr::NamespaceCall { ns: Abilities, method: "is_known" | "effects" }` — ability registry access
- `IrExpr::NamespaceField { ns: Config, field: "combat.engagement_range" | "cascade.max_iterations" }` — config reads
- `IrExpr::BuiltinCall(Builtin::Min, ...)` — `min(shield, a)`, `min(cur_hp + a, max_hp)`
- `IrExpr::BuiltinCall(Builtin::Max, ...)` — `max(cur_hp - residual, 0.0)`, `max(cur_hp - damage, 0.0)`
- `IrExpr::BuiltinCall(Builtin::SaturatingAdd, ...)` — `saturating_add(t, duration_ticks)` (stun/slow)
- `IrExpr::LitFloat` — `0.0`, `1.0` (comparisons and accumulations)
- `IrExpr::LitInt` — `0`, `1`, `2` (break reasons, depth increment, depth threshold)
- `IrExpr::Binary(BinOp::Gt, ...)` — `a > 0.0`, `e > cur_exp`
- `IrExpr::Binary(BinOp::Lt | LtEq, ...)` — `residual > 0.0`, `new_hp <= 0.0`, `hp_pct < 0.5`
- `IrExpr::Binary(BinOp::Eq, ...)` — `a != 0`, `from != to`, `old != new`, `partner != dead`
- `IrExpr::Binary(BinOp::NotEq, ...)` — `old != new`, `new != mover`, `stranded != new`, etc.
- `IrExpr::Binary(BinOp::Add, ...)` — `depth + 1`, `cur_hp + a`
- `IrExpr::Binary(BinOp::Sub, ...)` — `a - absorbed`, `cur_hp - residual`
- `IrExpr::Binary(BinOp::Div, ...)` — `cur_hp / max_hp` (hp_pct computation in chronicle_wound, rally_on_wound)
- `IrExpr::If { cond, then_expr, else_expr }` — `if sel == TargetSelector::Target { target } else { caster }` (cast dispatch)
- `IrExpr::EnumVariant { ty: "TargetSelector", variant: "Target" }` — comparison in cast dispatch
- `IrExpr::Local(...)` — references to `let`-bound locals throughout all handlers

### IrPattern variants used in physics bodies
- `IrPattern::Struct { name: "Damage" | "Heal" | "Shield" | "Stun" | "Slow" | "TransferGold" | "ModifyStanding" | "CastAbility", bindings }` — match arms in cast dispatch
- `IrPhysicsPattern::Kind(IrEventPattern { name: "EffectDamageApplied" | "AgentCast" | etc })` — event pattern on `on` handler

### Physics event patterns (events listened to)
All handlers in wolves+humans listen to these events:
- `EffectDamageApplied`, `EffectHealApplied`, `EffectShieldApplied`, `EffectStunApplied`, `EffectSlowApplied` (from their respective effect handlers)
- `EffectGoldTransfer`, `EffectStandingDelta`
- `OpportunityAttackTriggered`
- `RecordMemory`
- `AgentCast` — the big cast dispatch rule
- `AgentDied` — `engagement_on_death`, `fear_spread_on_death`, `chronicle_death`
- `AgentMoved` — `engagement_on_move`
- `AgentAttacked` — `chronicle_attack`, `chronicle_wound`, `rally_on_wound`
- `EngagementCommitted` — `pack_focus_on_engagement`, `chronicle_engagement`
- `EngagementBroken` — `chronicle_break`
- `FearSpread` — `chronicle_rout`
- `AgentFled` — `chronicle_flee`
- `RallyCall` — `chronicle_rally`

### @phase annotations
- `@phase(event)` — rules that fire during event dispatch (damage, heal, shield, stun, slow, gold, standing, cast, engagement_on_move, engagement_on_death, fear_spread_on_death, pack_focus_on_engagement, rally_on_wound)
- `@phase(post)` — chronicle rules that fire after the tick's event-dispatch finishes (all 8 chronicle_* rules)

---

## ViewIR variants used

### @lazy views
- `is_hostile(a, b) -> bool` (line 24) — body: `IrExpr::NamespaceCall { ns: Agents, method: "is_hostile_to", args: [a, b] }`
- `is_stunned(a) -> bool` (line 37) — body: `IrExpr::Binary(BinOp::Lt, NamespaceField { ns: World, field: "tick" }, NamespaceCall { ns: Agents, method: "stun_expires_at_tick", args: [a] })`
- `slow_factor(a) -> i16` (line 49) — body: `IrExpr::If { cond: Binary(Lt, world.tick, agents.slow_expires_at_tick(a)), then_expr: agents.slow_factor_q8(a), else_expr: LitInt(0) }`

### @materialized views with fold bodies
- `threat_level(a, b) -> f32` (line 75)
  - `on AgentAttacked { actor: b, target: a }` → `self += 1.0`
  - `on EffectDamageApplied { actor: b, target: a }` → `self += 1.0`
  - `@decay(rate = 0.98, per = tick)`
  - `storage = per_entity_topk(K = 8)`
  - `clamp: [0.0, 1000.0]`

- `engaged_with(a) -> Agent` (line 96)
  - `on EngagementCommitted { actor: a, target: b }` → `self += 1`
  - `on EngagementBroken { actor: a, former_target: b }` → `self += 1`
  - `storage = per_entity_topk` (K=1 implied single-slot)
  - No decay, no clamp

- `my_enemies(observer, attacker) -> f32` (line 124)
  - `on AgentAttacked { actor: attacker, target: observer }` → `self += 1.0`
  - No decay
  - `clamp: [0.0, 1.0]`
  - `storage = per_entity_topk(K = 8)`

- `kin_fear(observer, dead_kin) -> f32` (line 167)
  - `on FearSpread { observer: observer, dead_kin: dead_kin }` → `self += 1.0`
  - `@decay(rate = 0.891, per = tick)`
  - `clamp: [0.0, 10.0]`
  - `storage = per_entity_topk(K = 8)`

- `pack_focus(observer, target) -> f32` (line 207)
  - `on PackAssist { observer: observer, target: target }` → `self += 1.0`
  - `@decay(rate = 0.933, per = tick)`
  - `clamp: [0.0, 10.0]`
  - `storage = per_entity_topk(K = 8)`

- `rally_boost(observer, wounded_kin) -> f32` (line 253)
  - `on RallyCall { observer: observer, wounded_kin: wounded_kin }` → `self += 1.0`
  - `@decay(rate = 0.891, per = tick)`
  - `clamp: [0.0, 10.0]`
  - `storage = per_entity_topk(K = 8)`

- `standing(a, b) -> i32` (line 293)
  - `on EffectStandingDelta { a: a, b: b, delta: delta }` → `self += delta`
  - No decay
  - `clamp: [-1000, 1000]`
  - `@symmetric_pair_topk(K = 8)` — distinct storage shape

- `memory(observer, source) -> f32` (line 325)
  - `on RecordMemory { observer: observer, source: source }` → `self += 1.0`
  - No decay, no clamp
  - `@per_entity_ring(K = 64)`

### ViewBodyIR variants required
- `ViewBodyIR::Expr(IrExprNode)` — @lazy views (is_hostile, is_stunned, slow_factor)
- `ViewBodyIR::Fold { initial, handlers, clamp }` — all @materialized views

### ViewKind variants required
- `ViewKind::Lazy` — three @lazy views
- `ViewKind::Materialized(StorageHint::PerEntityTopK { k: 8, ... })` — threat_level, my_enemies, kin_fear, pack_focus, rally_boost
- `ViewKind::Materialized(StorageHint::PerEntityTopK { k: 1, ... })` — engaged_with
- `ViewKind::Materialized(StorageHint::SymmetricPairTopK { k: 8 })` — standing
- `ViewKind::Materialized(StorageHint::PerEntityRing { k: 64 })` — memory

### DecayHint variants required
- `DecayHint { rate: 0.98, per: DecayUnit::Tick }` — threat_level
- `DecayHint { rate: 0.891, per: DecayUnit::Tick }` — kin_fear, rally_boost
- `DecayHint { rate: 0.933, per: DecayUnit::Tick }` — pack_focus
- No decay — my_enemies, engaged_with, standing, memory

### IrStmt variants in view fold bodies
- `IrStmt::SelfUpdate { op: "+=", value }` — every fold handler uses `self += ...`

---

## Stdlib functions and namespace calls used

| Namespace | Method / Field | Used in | Notes |
|-----------|----------------|---------|-------|
| `agents` | `alive(id)` | masks, physics | bool — dead-agent guard |
| `agents` | `pos(id)` | masks | Vec3 — agent position |
| `agents` | `hp(id)` | physics | f32 — current hp |
| `agents` | `max_hp(id)` | physics | f32 — max hp |
| `agents` | `set_hp(id, v)` | physics | mutation |
| `agents` | `shield_hp(id)` | physics | f32 — current shield |
| `agents` | `set_shield_hp(id, v)` | physics | mutation |
| `agents` | `kill(id)` | physics | marks agent dead |
| `agents` | `attack_damage(id)` | physics | f32 — opportunity attack |
| `agents` | `stun_expires_at_tick(id)` | physics, views | u32 |
| `agents` | `set_stun_expires_at_tick(id, v)` | physics | mutation |
| `agents` | `slow_expires_at_tick(id)` | physics, views | u32 |
| `agents` | `slow_factor_q8(id)` | physics, views | i16 — q8 fixed-point |
| `agents` | `set_slow_expires_at_tick(id, v)` | physics | mutation |
| `agents` | `set_slow_factor_q8(id, v)` | physics | mutation |
| `agents` | `engaged_with(id)` | masks | `Option<AgentId>` |
| `agents` | `engaged_with_or(id, sentinel)` | physics | AgentId with sentinel fallback |
| `agents` | `clear_engaged_with(id)` | physics | mutation |
| `agents` | `set_engaged_with(id, partner)` | physics | mutation |
| `agents` | `is_hostile_to(a, b)` | views | bool — creature hostility |
| `agents` | `record_memory(o, s, f, c, t)` | physics | cold_memory push |
| `agents` | `record_cast_cooldowns(caster, ab, t)` | physics | GCD + slot cooldown |
| `agents` | `add_gold(id, a)` | physics | mutation |
| `agents` | `sub_gold(id, a)` | physics | mutation |
| `agents` | `adjust_standing(a, b, delta)` | physics | mutation |
| `query` | `nearby_agents(pos, radius)` | masks | candidate source for MoveToward, Attack |
| `query` | `nearest_hostile_to_or(id, r, sentinel)` | physics | engagement_on_move |
| `query` | `nearby_kin(id, radius)` | physics | fear_spread, pack_focus, rally_on_wound |
| `abilities` | `known(self, ability)` | masks | bool — spellbook membership |
| `abilities` | `cooldown_ready(self, ability)` | masks | bool — per-slot cooldown |
| `abilities` | `is_known(ab)` | physics | bool — registry membership |
| `abilities` | `effects(ab)` | physics | iterable EffectOp list |
| `world` | `.tick` | views, physics | u32 — current tick |
| `config` | `movement.max_move_radius` | masks | f32 |
| `config` | `combat.attack_range` | masks | f32 |
| `config` | `combat.engagement_range` | physics | f32 |
| `cascade` | `max_iterations` | physics | u32 — recursion ceiling |
| `view::` | `is_hostile(a, b)` | masks | lazy view dispatch |
| `view::` | `is_stunned(a)` | masks | lazy view dispatch |
| `view::` | `threat_level(a, b)` | scoring | materialized view read |
| `view::` | `kin_fear(a, _)` | scoring | wildcard materialized read |
| `view::` | `my_enemies(a, b)` | scoring | materialized view read |
| `view::` | `pack_focus(a, b)` | scoring | materialized view read |
| `view::` | `rally_boost(a, _)` | scoring | wildcard materialized read |

### Standalone builtins used
| Builtin | Sites |
|---------|-------|
| `distance(pos_a, pos_b)` | masks.sim line 79 (Attack range check) |
| `min(a, b)` | physics.sim lines 28, 58 (shield absorption, heal clamp) |
| `max(a, b)` | physics.sim lines 34, 58, 167 (hp floor, heal cap, opportunity attack) |
| `saturating_add(a, b)` | physics.sim lines 237, 241 (stun/slow expiry ticks) |

---

## Field reads observed

### Self-side fields (scoring agent)
- `self.hp` — raw current hp (`f32`); Flee row, lines 111–112
- `self.hp_pct` — computed hp percentage; Flee (line 113), Attack (line 231), Eat (line 260)

### Target-side fields (scored candidate target)
- `target.hp_pct` — hp percentage on the opponent; Attack row, lines 232–233

### Entity (creatures) taxonomy fields — entities.sim
Entities are not evaluated at runtime by the interpreter (EntityIR is pure
data). The hostility matrix is encoded as `CreatureType::is_hostile_to` emitted
by the entity compiler; accessed at runtime via `agents.is_hostile_to(a, b)`
(which the lazy view `is_hostile` wraps).

Creature fields present in wolves+humans:
- `creature_type: CreatureType` (Human, Wolf, Deer, Dragon)
- `capabilities.herds_when_fleeing: bool` — used in flee-direction logic (Flee step.rs arm)
- `predator_prey.prey_of` / `predator_prey.preys_on` — hostility matrix source

---

## NOT used — explicitly P1c+ scope

The following IR constructs appear in `ir.rs` / `ast.rs` but have zero usage
in any wolves+humans rule file:

- `IrExpr::PerUnit` with a non-view expression (only ViewCall + per_unit is used)
- `IrExpr::Quantifier` (`forall` / `exists`) — not in any wolves+humans rule
- `IrExpr::Fold` (`count`, `sum`, `max`, `min` fold forms) — not used
- `IrExpr::In` / `IrExpr::Contains` — not used
- `IrExpr::List` / `IrExpr::Tuple` — not used directly in rule bodies
- `IrExpr::AbilityTag` / `IrExpr::AbilityHint` / `IrExpr::AbilityHintLit` / `IrExpr::AbilityRange` / `IrExpr::AbilityOnCooldown` — ability evaluation Phase 2 rows; no `per_ability` row in scoring.sim
- `ScoringIR::per_ability_rows` (`PerAbilityRowIR`) — empty in wolves+humans
- `IrActionHeadShape::Named` — not used in any mask or scoring head
- `IrPhysicsPattern::Tag { ... }` — tag-pattern dispatch; all handlers use `Kind` patterns
- `BinOp::Or`, `BinOp::Mod` — not found in any rule body
- `BinOp::Mul` — not used (per_unit gradient uses `PerUnit` variant, not raw Mul)
- `BinOp::LtEq` — not explicitly observed (only `Lt`, `Gt`, `GtEq` appear)
- `UnOp::Neg` — not used
- `Builtin::Count`, `Builtin::Sum`, `Builtin::PlanarDistance`, `Builtin::ZSeparation`, `Builtin::Entity`, `Builtin::Clamp`, `Builtin::Abs`, `Builtin::Floor`, `Builtin::Ceil`, `Builtin::Round`, `Builtin::Ln`, `Builtin::Log2`, `Builtin::Log10`, `Builtin::Sqrt` — not used
- `StorageHint::PairMap` / `StorageHint::LazyCached` — not used by any wolves+humans view
- `IrStmt::For` with a filter clause — `for ... in ... { ... }` bodies have no `where` filter in these rules
- `VerbIR`, `InvariantIR`, `ProbeIR`, `MetricIR` — not relevant to interpreter eval path
- `NamespaceId::Terrain`, `NamespaceId::Membership`, `NamespaceId::Relationship`, `NamespaceId::TheoryOfMind`, `NamespaceId::Group`, `NamespaceId::Quest`, `NamespaceId::Voxel`, `NamespaceId::Rng` — not referenced in any wolves+humans rule
- `IrExpr::Index`, `IrExpr::Ctor` (non-event form), `IrExpr::Raw` — not used

---

## Summary table

| Rule class | IrStmt forms | IrExpr forms | BinOp | UnOp | Builtins | Namespaces |
|------------|-------------|-------------|-------|------|----------|------------|
| MaskIR | (predicate only — no statements, just expr) | Binary, Unary, NamespaceCall, ViewCall, BuiltinCall(Distance), LitFloat, EnumVariant | And, NotEq, Lt, Eq | Not | distance | agents, query, abilities, config, view:: |
| ScoringIR | (expression tree only) | LitFloat, Field(self/target), If, PerUnit, ViewCall, Binary | Lt, Gt, GtEq, Add | — | — | view:: |
| PhysicsIR | Let, If, Emit, For, Match, Expr (side-effect call) | NamespaceCall(agents/query/abilities), BuiltinCall(min/max/saturating_add), Binary, If, LitFloat, LitInt, EnumVariant, Local | Gt, GtEq, Lt, LtEq, Eq, NotEq, Add, Sub, Div | — | min, max, saturating_add | agents, query, abilities, config, cascade |
| ViewIR | SelfUpdate(+=) | NamespaceCall(agents/world), Binary(Lt), LitInt(0) | Lt | — | — | agents, world |
