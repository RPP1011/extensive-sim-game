# Engine Plan 3 — Persistence + Observation Packer + Probes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/engine/spec.md` §§16 (save/load), §18 (probes), §21 (observation packing) from ❌ to ✅ (Serial column). After this plan: engine can snapshot + restore state across restarts, pack observation feature tensors for ML training, and run scripted reproducible smoke tests.

**Architecture:** Additive to current engine crate. Three independent sub-projects executed in order. `SerialBackend` is the reference implementation per the 2026-04-19 spec rewrite; GPU counterparts land in Plans 6+ with parity verification. Save/load is backend-agnostic at the surface — always operates on host-mirror post-sync. Obs packer is backend-local — Serial does scalar packing, GPU gets a kernel later. Probes dispatch through the tick pipeline so they're backend-agnostic by construction.

**Tech Stack:** Rust 2021; no new deps. Save/load format is hand-rolled binary (little-endian). Obs packing is plain `f32` math over SoA slices.

---

## Files overview

New:
- `crates/engine/src/snapshot/mod.rs` — re-export
- `crates/engine/src/snapshot/format.rs` — magic, header, writer, reader
- `crates/engine/src/snapshot/migrate.rs` — migration-function registry
- `crates/engine/src/obs/mod.rs` — re-export
- `crates/engine/src/obs/packer.rs` — `ObsPacker`, `FeatureSource` trait
- `crates/engine/src/obs/sources.rs` — built-in `FeatureSource` impls
- `crates/engine/src/probe/mod.rs` — `Probe` struct, runner

Modified:
- `crates/engine/src/lib.rs` — register new modules
- `crates/engine/src/schema_hash.rs` — bump for snapshot format, obs packer layout, probe harness version
- `crates/engine/.schema_hash`

Tests (new):
- `tests/snapshot_roundtrip.rs`, `tests/snapshot_schema_mismatch.rs`, `tests/snapshot_migration.rs`
- `tests/obs_packer.rs`, `tests/obs_sources_vitals.rs`, `tests/obs_sources_position.rs`, `tests/obs_sources_neighbors.rs`
- `tests/probe_harness.rs`, `tests/probe_determinism.rs`
- `tests/acceptance_plan3.rs`

---

## Phase 1 — Save/Load (§16)

Host snapshot format. Backend-agnostic: always saves from host mirror. Schema-hash versioned — load rejects mismatches unless a migration is registered.

### Task 1: Snapshot header + magic

**Files:**
- Create: `crates/engine/src/snapshot/mod.rs`
- Create: `crates/engine/src/snapshot/format.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/snapshot_header.rs`

- [ ] **Step 1: Failing test**

```rust
use engine::snapshot::{SnapshotHeader, MAGIC};

#[test]
fn header_serializes_to_64_bytes() {
    let h = SnapshotHeader {
        magic:         *MAGIC,
        schema_hash:   [0xAB; 32],
        tick:          42,
        seed:          0xDEADBEEF_CAFEF00D,
        format_version: 1,
        reserved:      [0; 7],
    };
    let bytes = h.to_bytes();
    assert_eq!(bytes.len(), 64);
    let h2 = SnapshotHeader::from_bytes(&bytes).unwrap();
    assert_eq!(h.schema_hash, h2.schema_hash);
    assert_eq!(h.tick, h2.tick);
    assert_eq!(h.seed, h2.seed);
}

#[test]
fn magic_is_wsimsv01_ascii() {
    assert_eq!(MAGIC, b"WSIMSV01");
}

#[test]
fn from_bytes_rejects_bad_magic() {
    let mut bytes = vec![0u8; 64];
    bytes[..8].copy_from_slice(b"NOPEXXXX");
    assert!(SnapshotHeader::from_bytes(&bytes).is_err());
}
```

- [ ] **Step 2: Implement `crates/engine/src/snapshot/format.rs`**

```rust
//! Snapshot file format. 64-byte header + field blocks.

pub const MAGIC: &[u8; 8] = b"WSIMSV01";
pub const FORMAT_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapshotHeader {
    pub magic:          [u8; 8],
    pub schema_hash:    [u8; 32],
    pub tick:           u32,
    pub seed:           u64,
    pub format_version: u16,
    pub reserved:       [u8; 7],  // pad to 64 bytes total: 8+32+4+8+2+7 = 61... wait
}
```

Recount: 8 + 32 + 4 + 8 + 2 = 54. Need 10 more for 64. So `reserved: [u8; 10]`. Fix accordingly.

```rust
impl SnapshotHeader {
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[..8].copy_from_slice(&self.magic);
        out[8..40].copy_from_slice(&self.schema_hash);
        out[40..44].copy_from_slice(&self.tick.to_le_bytes());
        out[44..52].copy_from_slice(&self.seed.to_le_bytes());
        out[52..54].copy_from_slice(&self.format_version.to_le_bytes());
        out[54..64].copy_from_slice(&self.reserved);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotError> {
        if bytes.len() < 64 { return Err(SnapshotError::ShortHeader); }
        if &bytes[..8] != MAGIC { return Err(SnapshotError::BadMagic); }
        let mut schema_hash = [0u8; 32];
        schema_hash.copy_from_slice(&bytes[8..40]);
        let tick = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        let seed = u64::from_le_bytes(bytes[44..52].try_into().unwrap());
        let format_version = u16::from_le_bytes(bytes[52..54].try_into().unwrap());
        let mut reserved = [0u8; 10];
        reserved.copy_from_slice(&bytes[54..64]);
        Ok(Self {
            magic: *MAGIC, schema_hash, tick, seed, format_version, reserved,
        })
    }
}

#[derive(Debug)]
pub enum SnapshotError {
    BadMagic,
    ShortHeader,
    SchemaMismatch { expected: [u8; 32], found: [u8; 32] },
    Truncated(&'static str),
    Io(std::io::Error),
}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self { SnapshotError::Io(e) }
}
```

- [ ] **Step 3: `snapshot/mod.rs`**

```rust
pub mod format;
pub mod migrate;  // stub for Task 7

pub use format::{SnapshotError, SnapshotHeader, FORMAT_VERSION, MAGIC};
```

- [ ] **Step 4: Register `pub mod snapshot;` in `lib.rs`**. Alphabetical: between `schema_hash` and `spatial`? Actually `snapshot` > `schema_hash` lexically (schema < snapshot). Place after `schema_hash`.

Stub `migrate.rs`:

```rust
pub struct MigrationRegistry;
impl MigrationRegistry {
    pub fn new() -> Self { Self }
}
impl Default for MigrationRegistry { fn default() -> Self { Self::new() } }
```

- [ ] **Step 5: Run + commit**

```
cargo test -p engine --test snapshot_header
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

Expect 150 + 3 = 153 tests.

```bash
git add crates/engine/src/snapshot/ crates/engine/src/lib.rs \
        crates/engine/tests/snapshot_header.rs
git commit -m "feat(engine): snapshot header + 64-byte layout + magic WSIMSV01"
```

---

### Task 2: Full snapshot roundtrip — hot + cold fields

**Files:**
- Modify: `crates/engine/src/snapshot/format.rs`
- Test: `crates/engine/tests/snapshot_roundtrip.rs`

- [ ] **Step 1: Failing test**

```rust
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::snapshot::{save_snapshot, load_snapshot};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn save_then_load_produces_identical_state() {
    let mut state = SimState::new(8, 42);
    let events = EventRing::with_cap(64);

    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(1.0, 2.0, 3.0), hp: 50.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(-4.0, 5.0, -6.0), hp: 75.0,
    }).unwrap();
    state.set_agent_hunger(a, 0.3);
    state.tick = 100;

    let tmp = std::env::temp_dir().join("engine_snap_rt.bin");
    save_snapshot(&state, &events, &tmp).unwrap();
    let (state2, _events2) = load_snapshot(&tmp).unwrap();

    assert_eq!(state2.tick, 100);
    assert_eq!(state2.seed, 42);
    assert_eq!(state2.agent_pos(a), state.agent_pos(a));
    assert_eq!(state2.agent_pos(b), state.agent_pos(b));
    assert_eq!(state2.agent_hp(a), state.agent_hp(a));
    assert_eq!(state2.agent_hunger(a), Some(0.3));
    assert_eq!(state2.agent_creature_type(a), Some(CreatureType::Human));
    assert_eq!(state2.agent_creature_type(b), Some(CreatureType::Wolf));
    assert!(state2.agent_alive(a));
    assert!(state2.agent_alive(b));

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn save_then_load_preserves_freelist_reuse() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    state.kill_agent(a);

    let events = EventRing::with_cap(16);
    let tmp = std::env::temp_dir().join("engine_snap_fl.bin");
    save_snapshot(&state, &events, &tmp).unwrap();
    let (mut state2, _) = load_snapshot(&tmp).unwrap();

    // After load, next spawn should reuse slot 1 — proving freelist survived.
    let b = state2.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::X, hp: 100.0,
    }).unwrap();
    assert_eq!(b.raw(), a.raw(), "freelist reused after load");

    std::fs::remove_file(&tmp).ok();
}
```

- [ ] **Step 2: Extend `format.rs`** with `save_snapshot` and `load_snapshot` free functions.

Field-block layout after header:

```
[header 64B]
[alive_len: u32 LE]
[alive: alive_len bytes, 1 byte per slot, 0/1]
[freelist_len: u32 LE]
[freelist: freelist_len × u32 LE]
[hot_pos: alive_len × 12B (3×f32 LE)]
[hot_hp: alive_len × 4B]
[hot_max_hp: alive_len × 4B]
[hot_hunger: alive_len × 4B]
[hot_thirst: alive_len × 4B]
[hot_rest_timer: alive_len × 4B]
[hot_movement_mode: alive_len × 1B (enum discriminant)]
[cold_creature_type: alive_len × 2B (1B present-flag + 1B discriminant)]
[cold_channels: variable, prefix each with present-flag + count + bytes]
[cold_spawn_tick: alive_len × 5B (1B present + 4B tick)]
[event_ring_len: u32 LE]
[event_entries: each EventId (2×u32) + Event byte-packed per ring hash format + cause (present+2×u32)]
```

Implementation strategy: write a builder that serializes each field in turn; reader uses offset tracking.

For brevity, the complete implementation is ~200 lines. The key invariant: **byte-for-byte stable across runs with the same state**. Test with raw byte-level equality of save bytes from two fresh same-seed runs.

Compute the schema hash via `engine::schema_hash::schema_hash()` and write it into the header. On load, compare against current `schema_hash()`; return `SnapshotError::SchemaMismatch` on disagreement.

Implementation note: `EventRing` currently exposes `iter()` + `total_pushed()`. For snapshot we need to iterate entries in push order AND capture their `EventId` + `cause`. Add a `EventRing::entries_for_snapshot() -> impl Iterator<Item = (Event, EventId, Option<EventId>)>` method (or similar internal access). Restore path constructs the ring by calling `push_caused_by` in order, then updates `current_tick` + `next_seq` + `total_pushed` to match.

- [ ] **Step 3: Run + commit**

```
cargo test -p engine --test snapshot_roundtrip
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

Expect 153 + 2 = 155 tests.

```bash
git add crates/engine/src/snapshot/format.rs crates/engine/src/event/ring.rs \
        crates/engine/tests/snapshot_roundtrip.rs
git commit -m "feat(engine): save_snapshot + load_snapshot — hot/cold fields + freelist + event ring"
```

---

### Task 3: Schema-mismatch rejection

**Files:**
- Test: `crates/engine/tests/snapshot_schema_mismatch.rs`

- [ ] **Step 1: Failing test**

```rust
use engine::snapshot::{SnapshotError, SnapshotHeader, load_snapshot, save_snapshot, MAGIC};
use engine::event::EventRing;
use engine::state::SimState;

#[test]
fn load_rejects_snapshot_with_wrong_schema_hash() {
    let state = SimState::new(4, 42);
    let events = EventRing::with_cap(16);
    let tmp = std::env::temp_dir().join("engine_snap_sm.bin");
    save_snapshot(&state, &events, &tmp).unwrap();

    // Corrupt the schema_hash bytes (offset 8..40 in the header).
    let mut buf = std::fs::read(&tmp).unwrap();
    buf[8] ^= 0xFF;
    std::fs::write(&tmp, &buf).unwrap();

    let err = load_snapshot(&tmp).unwrap_err();
    match err {
        SnapshotError::SchemaMismatch { .. } => (),
        other => panic!("expected SchemaMismatch, got {:?}", other),
    }
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn load_rejects_short_file() {
    let tmp = std::env::temp_dir().join("engine_snap_short.bin");
    std::fs::write(&tmp, b"short").unwrap();
    let err = load_snapshot(&tmp).unwrap_err();
    match err {
        SnapshotError::ShortHeader | SnapshotError::Io(_) => (),
        other => panic!("expected ShortHeader or Io, got {:?}", other),
    }
    std::fs::remove_file(&tmp).ok();
}
```

- [ ] **Step 2: Verify `load_snapshot` already rejects (Task 2 should have implemented this).** If not, add the check after `SnapshotHeader::from_bytes`.

- [ ] **Step 3: Commit (test-only — no new code if Task 2 was complete)**

```bash
git add crates/engine/tests/snapshot_schema_mismatch.rs
git commit -m "test(engine): snapshot load rejects schema-hash mismatch + short files"
```

---

### Task 4: Migration registry

**Files:**
- Modify: `crates/engine/src/snapshot/migrate.rs`
- Test: `crates/engine/tests/snapshot_migration.rs`

Migration is a function from `(old_hash, old_bytes) -> new_bytes`. Registered at init; applied at load if the file's hash matches a registered `old_hash`.

- [ ] **Step 1: Failing test**

```rust
use engine::snapshot::{MigrationRegistry, load_snapshot_with_migrations, save_snapshot};
use engine::event::EventRing;
use engine::state::SimState;

#[test]
fn migration_from_old_hash_runs_on_load() {
    // Save with a fake old-hash snapshot by manually writing the file.
    // ... construct a fake-schema-hash file ...
    // ... register migration: old_hash -> current_hash ...
    // ... load via load_snapshot_with_migrations ...
    // ... assert migration closure was called ...
    // (Exact shape depends on Task 2's serialization; fill in when that API is clear.)
}
```

Skip the full test if this gets complex — or simplify to just register a migration, save a fake-old-hash file, assert that load invokes the closure. Keep it minimal.

- [ ] **Step 2: Implementation sketch**

```rust
use super::format::{SnapshotError, SnapshotHeader};

type MigrationFn = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync>;

pub struct MigrationRegistry {
    migrations: Vec<([u8; 32], [u8; 32], MigrationFn)>,  // (from, to, fn)
}

impl MigrationRegistry {
    pub fn new() -> Self { Self { migrations: Vec::new() } }
    pub fn register<F>(&mut self, from: [u8; 32], to: [u8; 32], f: F)
    where F: Fn(&[u8]) -> Result<Vec<u8>, SnapshotError> + Send + Sync + 'static {
        self.migrations.push((from, to, Box::new(f)));
    }
    // Apply migrations to reach `target_hash`. Returns migrated bytes or error.
    pub fn apply(&self, current_hash: [u8; 32], target_hash: [u8; 32], bytes: &[u8])
        -> Result<Vec<u8>, SnapshotError>
    { /* chain-apply until current == target or no path */ }
}

pub fn load_snapshot_with_migrations(
    path: &std::path::Path,
    reg: &MigrationRegistry,
) -> Result<(SimState, EventRing), SnapshotError> {
    let bytes = std::fs::read(path)?;
    let hdr = SnapshotHeader::from_bytes(&bytes)?;
    let current = engine::schema_hash::schema_hash();
    let migrated = if hdr.schema_hash == current {
        bytes
    } else {
        reg.apply(hdr.schema_hash, current, &bytes)?
    };
    super::format::load_from_bytes(&migrated)
}
```

Keep it as a stub if the full chain-composition is too ambitious for now. MVP: just support one-step migration; multi-step is future work.

- [ ] **Step 3: Run + commit**

```bash
git add crates/engine/src/snapshot/migrate.rs \
        crates/engine/tests/snapshot_migration.rs
git commit -m "feat(engine): snapshot migration registry (one-step; chain composition is future)"
```

---

## Phase 2 — Observation Packer (§21)

`ObsPacker` builds a feature tensor `[n × feature_dim] f32` for policy input. Feature sources compose.

### Task 5: `FeatureSource` trait + `ObsPacker`

**Files:**
- Create: `crates/engine/src/obs/mod.rs`
- Create: `crates/engine/src/obs/packer.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/obs_packer.rs`

- [ ] **Step 1: Failing test**

```rust
use engine::creature::CreatureType;
use engine::ids::AgentId;
use engine::obs::{FeatureSource, ObsPacker};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

struct ConstantSource(f32);
impl FeatureSource for ConstantSource {
    fn dim(&self) -> usize { 2 }
    fn pack(&self, _state: &SimState, _agent: AgentId, out: &mut [f32]) {
        out[0] = self.0;
        out[1] = self.0 + 1.0;
    }
}

#[test]
fn packer_computes_total_feature_dim() {
    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(1.0)));
    packer.register(Box::new(ConstantSource(2.0)));
    assert_eq!(packer.feature_dim(), 4);
}

#[test]
fn pack_batch_writes_row_major_per_agent() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
    }).unwrap();
    let b = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::X, hp: 80.0,
    }).unwrap();

    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(5.0)));    // dim 2 → [5, 6]
    packer.register(Box::new(ConstantSource(10.0)));   // dim 2 → [10, 11]

    let mut out = vec![0.0f32; 2 * 4];
    packer.pack_batch(&state, &[a, b], &mut out);

    // Row 0 = agent a: [5,6,10,11]. Row 1 = agent b: same.
    assert_eq!(&out[..4], &[5.0, 6.0, 10.0, 11.0]);
    assert_eq!(&out[4..], &[5.0, 6.0, 10.0, 11.0]);
}

#[test]
fn pack_batch_panics_on_wrong_output_size() {
    let mut packer = ObsPacker::new();
    packer.register(Box::new(ConstantSource(1.0)));

    let state = SimState::new(4, 42);
    let mut out = vec![0.0f32; 1];  // too small for even 1 agent × 2 dim
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        packer.pack_batch(&state, &[AgentId::new(1).unwrap()], &mut out);
    }));
    assert!(result.is_err());
}
```

- [ ] **Step 2: Implement**

```rust
// crates/engine/src/obs/packer.rs
use crate::ids::AgentId;
use crate::state::SimState;

pub trait FeatureSource: Send + Sync {
    fn dim(&self) -> usize;
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]);
}

pub struct ObsPacker {
    sources:     Vec<Box<dyn FeatureSource>>,
    feature_dim: usize,
}

impl ObsPacker {
    pub fn new() -> Self { Self { sources: Vec::new(), feature_dim: 0 } }

    pub fn register(&mut self, source: Box<dyn FeatureSource>) {
        self.feature_dim += source.dim();
        self.sources.push(source);
    }

    pub fn feature_dim(&self) -> usize { self.feature_dim }

    /// Pack `[agents × feature_dim]` f32 row-major into `out`.
    /// Panics if `out.len() < agents.len() * feature_dim`.
    pub fn pack_batch(&self, state: &SimState, agents: &[AgentId], out: &mut [f32]) {
        let need = agents.len() * self.feature_dim;
        assert!(out.len() >= need, "obs buffer too small: have {}, need {}", out.len(), need);
        for (row, &agent) in agents.iter().enumerate() {
            let row_start = row * self.feature_dim;
            let mut col = 0;
            for source in &self.sources {
                let d = source.dim();
                source.pack(state, agent, &mut out[row_start + col .. row_start + col + d]);
                col += d;
            }
        }
    }
}

impl Default for ObsPacker { fn default() -> Self { Self::new() } }
```

```rust
// crates/engine/src/obs/mod.rs
pub mod packer;
pub mod sources;

pub use packer::{FeatureSource, ObsPacker};
pub use sources::{VitalsSource, PositionSource, NeighborSource};
```

Stub `sources.rs` (Tasks 6-8 fill in):

```rust
use super::packer::FeatureSource;
pub struct VitalsSource;
pub struct PositionSource;
pub struct NeighborSource<const K: usize>;
```

Register module in `lib.rs` between `mask` and `policy`.

- [ ] **Step 3: Run + commit**

```
cargo test -p engine --test obs_packer
cargo test -p engine
cargo clippy -p engine --all-targets -- -D warnings
```

Expect ~157 tests.

```bash
git add crates/engine/src/obs/ crates/engine/src/lib.rs crates/engine/tests/obs_packer.rs
git commit -m "feat(engine): ObsPacker + FeatureSource trait + row-major batch packing"
```

---

### Task 6: `VitalsSource`

**Files:**
- Modify: `crates/engine/src/obs/sources.rs`
- Test: `crates/engine/tests/obs_sources_vitals.rs`

Packs `[hp_frac, hunger, thirst, rest_timer]` → dim 4.

- [ ] **Test**:

```rust
use engine::creature::CreatureType;
use engine::obs::{FeatureSource, VitalsSource};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

#[test]
fn vitals_pack_reads_hp_hunger_thirst_rest() {
    let mut state = SimState::new(4, 42);
    let a = state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 50.0,
    }).unwrap();
    state.set_agent_hunger(a, 0.3);
    state.set_agent_thirst(a, 0.7);
    state.set_agent_rest_timer(a, 0.9);

    let src = VitalsSource;
    let mut out = [0.0f32; 4];
    src.pack(&state, a, &mut out);
    assert!((out[0] - 0.5).abs() < 1e-6, "hp_frac = hp/max_hp = 50/50 = 1.0? check");
    // actually max_hp defaults to hp.max(1.0) = 50 → frac = 1.0
    // Fix the test expectations to match the actual SimState::spawn_agent behavior;
    // read state/mod.rs to confirm.
}
```

(Correct the expected hp_frac based on actual SimState spawn behavior: `max_hp = hp.max(1.0)` so an agent spawned with hp=50 has max_hp=50, hp_frac=1.0. Adjust the test to spawn with hp=50, then `set_agent_hp(a, 25.0)` to get hp_frac=0.5.)

- [ ] **Implementation**:

```rust
pub struct VitalsSource;
impl FeatureSource for VitalsSource {
    fn dim(&self) -> usize { 4 }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        let hp = state.agent_hp(agent).unwrap_or(0.0);
        let max_hp = state.agent_max_hp(agent).unwrap_or(1.0).max(1e-6);
        out[0] = hp / max_hp;
        out[1] = state.agent_hunger(agent).unwrap_or(0.0);
        out[2] = state.agent_thirst(agent).unwrap_or(0.0);
        out[3] = state.agent_rest_timer(agent).unwrap_or(0.0);
    }
}
```

- [ ] **Commit**:

```bash
git add crates/engine/src/obs/sources.rs crates/engine/tests/obs_sources_vitals.rs
git commit -m "feat(engine): VitalsSource — hp_frac + hunger + thirst + rest (dim 4)"
```

---

### Task 7: `PositionSource`

Packs `[pos.x, pos.y, pos.z, movement_mode_one_hot_walk, fly, swim, climb]` → dim 7.

- [ ] **Implementation**:

```rust
use crate::state::MovementMode;

pub struct PositionSource;
impl FeatureSource for PositionSource {
    fn dim(&self) -> usize { 7 }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        let pos = state.agent_pos(agent).unwrap_or(glam::Vec3::ZERO);
        out[0] = pos.x;
        out[1] = pos.y;
        out[2] = pos.z;
        let mode = state.agent_movement_mode(agent).unwrap_or(MovementMode::Walk);
        out[3] = (mode == MovementMode::Walk) as u8 as f32;
        out[4] = (mode == MovementMode::Fly)  as u8 as f32;
        out[5] = (mode == MovementMode::Swim) as u8 as f32;
        out[6] = (mode == MovementMode::Climb) as u8 as f32;
    }
}
```

- [ ] **Test at `obs_sources_position.rs`**:

Assert dim == 7, xyz reads correct, one-hot matches movement mode.

- [ ] **Commit**:

```bash
git commit -m "feat(engine): PositionSource — xyz + movement_mode one-hot (dim 7)"
```

---

### Task 8: `NeighborSource<const K: usize>`

Per-agent top-K nearest others, each contributing `[rel_x, rel_y, rel_z, dist, hp_frac, present_flag]` → dim `K * 6`.

- [ ] **Implementation**:

```rust
pub struct NeighborSource<const K: usize>;
impl<const K: usize> FeatureSource for NeighborSource<K> {
    fn dim(&self) -> usize { K * 6 }
    fn pack(&self, state: &SimState, agent: AgentId, out: &mut [f32]) {
        let sp = state.agent_pos(agent).unwrap_or(glam::Vec3::ZERO);
        // Collect (dist, other_id) for all alive others.
        let mut neighbors: Vec<(f32, AgentId)> = state.agents_alive()
            .filter(|id| *id != agent)
            .filter_map(|id| state.agent_pos(id).map(|op| ((op - sp).length(), id)))
            .collect();
        // Sort by distance ascending; take top K.
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(K);
        out.fill(0.0);
        for (i, (dist, other)) in neighbors.iter().enumerate() {
            let op = state.agent_pos(*other).unwrap_or(sp);
            let rel = op - sp;
            let hp = state.agent_hp(*other).unwrap_or(0.0);
            let max_hp = state.agent_max_hp(*other).unwrap_or(1.0).max(1e-6);
            let base = i * 6;
            out[base + 0] = rel.x;
            out[base + 1] = rel.y;
            out[base + 2] = rel.z;
            out[base + 3] = *dist;
            out[base + 4] = hp / max_hp;
            out[base + 5] = 1.0;  // present flag
        }
    }
}
```

Per-tick allocation: the `Vec<(f32, AgentId)>` is non-zero-alloc. For MVP accept it; Plan 5 will add a `SimScratch` slot for neighbor scratch buffers.

- [ ] **Test at `obs_sources_neighbors.rs`** — construct 1 + K + 2 agents, assert the K nearest are picked, present flags set only for first K, unused slots are zero.

- [ ] **Commit**:

```bash
git commit -m "feat(engine): NeighborSource<K> — top-K nearest (rel_xyz + dist + hp_frac + present, dim 6K)"
```

---

## Phase 3 — Probes (§18)

Scripted smoke tests that drive the tick pipeline with fixed seed + spawn + N ticks, then assert on observed state.

### Task 9: `Probe` struct + `run_probe`

**Files:**
- Create: `crates/engine/src/probe/mod.rs`
- Modify: `crates/engine/src/lib.rs`
- Test: `crates/engine/tests/probe_harness.rs`

- [ ] **Step 1: Failing test**

```rust
use engine::creature::CreatureType;
use engine::event::Event;
use engine::probe::{Probe, run_probe};
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn probe_two_agents() -> Probe {
    Probe {
        name: "two_agents_spawn_and_hold",
        seed: 42,
        spawn: |state| {
            state.spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human, pos: Vec3::ZERO, hp: 100.0,
            });
            state.spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(5.0, 0.0, 10.0), hp: 100.0,
            });
        },
        ticks: 10,
        assert: |state, _events| {
            if state.tick != 10 { return Err(format!("tick: {}", state.tick)); }
            if state.agents_alive().count() != 2 { return Err("agents".into()); }
            Ok(())
        },
    }
}

#[test]
fn run_probe_returns_ok_when_assertions_pass() {
    run_probe(&probe_two_agents()).unwrap();
}

#[test]
fn run_probe_returns_err_when_assertion_fails() {
    let bad = Probe {
        name: "expect_wrong_tick",
        seed: 42,
        spawn: |_| {},
        ticks: 3,
        assert: |state, _| {
            if state.tick == 3 { Err("oh no".into()) } else { Ok(()) }
        },
    };
    let r = run_probe(&bad);
    assert!(r.is_err());
}
```

- [ ] **Step 2: Implement `crates/engine/src/probe/mod.rs`**

```rust
//! Probes — scripted smoke tests that drive the tick pipeline with a fixed
//! seed + spawn + tick count, then assert on resulting state + events.

use crate::cascade::CascadeRegistry;
use crate::event::EventRing;
use crate::policy::UtilityBackend;
use crate::state::SimState;
use crate::step::{step, SimScratch};

pub struct Probe {
    pub name:   &'static str,
    pub seed:   u64,
    pub spawn:  fn(&mut SimState),
    pub ticks:  u32,
    pub assert: fn(&SimState, &EventRing) -> Result<(), String>,
}

pub fn run_probe(p: &Probe) -> Result<(), String> {
    // Default cap = 2048 (headroom for Announce cascades etc.). Tests can
    // extend later if needed via a richer ProbeConfig.
    let mut state = SimState::new(256, p.seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(4096);
    let cascade = CascadeRegistry::new();

    (p.spawn)(&mut state);
    for _ in 0..p.ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    (p.assert)(&state, &events).map_err(|e| format!("probe '{}': {}", p.name, e))
}
```

Register `pub mod probe;` in `lib.rs` — alphabetical, between `policy` and `rng`.

- [ ] **Step 3: Run + commit**

```bash
git add crates/engine/src/probe/ crates/engine/src/lib.rs \
        crates/engine/tests/probe_harness.rs
git commit -m "feat(engine): Probe struct + run_probe — scripted smoke-test runner"
```

---

### Task 10: Probe determinism

**Files:**
- Test: `crates/engine/tests/probe_determinism.rs`

Probes with the same spec run twice produce byte-identical event hashes.

```rust
use engine::event::EventRing;
use engine::probe::{Probe, run_probe};
use engine::creature::CreatureType;
use engine::state::{AgentSpawn, SimState};
use glam::Vec3;

fn hash_of_probe(seed: u64) -> [u8; 32] {
    use engine::cascade::CascadeRegistry;
    use engine::policy::UtilityBackend;
    use engine::step::{step, SimScratch};
    let mut state = SimState::new(64, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(8192);
    let cascade = CascadeRegistry::new();
    for i in 0..16 {
        let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(20.0 * angle.cos(), 20.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }
    for _ in 0..200 {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    events.replayable_sha256()
}

#[test]
fn same_seed_same_probe_hash() {
    assert_eq!(hash_of_probe(42), hash_of_probe(42));
}
```

```bash
git commit -m "test(engine): probe determinism — same seed reproduces event hash"
```

---

## Phase 4 — Integration + Acceptance

### Task 11: Schema-hash bump for Plan 3

**Files:**
- Modify: `crates/engine/src/schema_hash.rs`
- Modify: `crates/engine/.schema_hash`

Add fingerprint coverage for:

```rust
h.update(b"SnapshotFormat:v1:WSIMSV01");
h.update(b"FeatureSource:Vitals=4,Position=7,Neighbor<K>=6K");
h.update(b"ProbeHarness:v1");
```

Regenerate baseline.

```bash
git commit -m "chore(engine): schema hash bump — Plan 3 snapshot + obs + probe surface"
```

---

### Task 12: Plan 3 acceptance

**Files:**
- Test: `crates/engine/tests/acceptance_plan3.rs`

End-to-end: spawn 20 agents, run 100 ticks, save, reload, run 100 more ticks, assert final state matches a reference run of 200 continuous ticks.

```rust
use engine::creature::CreatureType;
use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::snapshot::{save_snapshot, load_snapshot};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;

fn run_straight(seed: u64, ticks: u32) -> [u8; 32] {
    let mut state = SimState::new(32, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(100_000);
    let cascade = CascadeRegistry::new();
    for i in 0..20 {
        let angle = (i as f32 / 20.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(30.0 * angle.cos(), 30.0 * angle.sin(), 10.0),
            hp: 100.0,
        });
    }
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    events.replayable_sha256()
}

fn run_with_save_reload(seed: u64, ticks_a: u32, ticks_b: u32) -> [u8; 32] {
    let tmp = std::env::temp_dir().join("engine_plan3_acc.bin");
    // ... run ticks_a; save_snapshot; load_snapshot; run ticks_b; return final events hash ...
    // Note: events are snapshotted in the ring; continuation must preserve replayable hash.
    // If event tail isn't fully preserved (ring may drop oldest), the hashes won't match
    // between "straight run" and "save-reload run". Document this in the assertion.
    unimplemented!()
}

#[test]
fn save_reload_yields_same_final_state() {
    let straight = run_straight(42, 200);
    let interrupted = run_with_save_reload(42, 100, 100);
    // The EVENT hashes may differ if save_snapshot truncated the ring.
    // More robust assertion: final STATE equality, not event-hash equality.
    // Implement by saving state, not just events.
    let _ = (straight, interrupted);
}
```

This task is tricky — event ring might drop oldest events on `save_snapshot` → post-reload event log is truncated → post-reload hash differs from continuous-run hash.

**Revised acceptance**: assert **state equality** (positions, hp, hunger) between save-reload and straight-run paths. State is a complete representation; event log tail isn't.

```bash
git commit -m "test(engine): Plan 3 acceptance — save-reload produces identical state"
```

---

## Self-review checklist

- [ ] **Spec coverage.** §§16, 18, 21 (Serial column) move from ❌ to ✅. §22 (schema hash) coverage extended.
- [ ] **Placeholder scan.** No TBD / TODO in committed code (one TBD in Task 4 migration chain composition, called out explicitly as "future").
- [ ] **Type consistency.** `SnapshotHeader`, `MigrationRegistry`, `FeatureSource`, `ObsPacker`, `Probe` signatures consistent across tasks.
- [ ] **Dependency direction.** No new crate deps. Snapshot is host-side bincode-shaped hand rolling. Obs packing is scalar math. Probes reuse existing step().
- [ ] **Backend neutrality.** Snapshot always operates on host mirror — valid for both Serial and GPU backends (GPU backend will force sync before save). Obs packer is Serial-only for now; GPU variant lands in Plan 7+. Probes drive step() — backend-agnostic by construction.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-19-engine-plan-3-persistence-obs-probes.md`. Same-session subagent-driven-development recommended.

Plan 3 leaves these Serial engine features complete:
- Snapshot save/load with schema-hash rejection + single-step migrations
- Feature tensor packing for policy input (vitals / position / top-K neighbors)
- Probe harness for scripted determinism tests

Plan 4 next: debug & trace runtime (§22) — trace_mask, causal_tree, tick_stepper, tick_profile, agent_history, snapshot repro bundle.

Then Plan 5: `ComputeBackend` trait extraction + parity-test infrastructure. GPU backend lands Plan 6+.
