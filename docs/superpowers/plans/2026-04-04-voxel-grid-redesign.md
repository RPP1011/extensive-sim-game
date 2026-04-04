# VoxelGrid Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 2D settlement-tied CityGrid with the existing 3D VoxelWorld (extended with building/zone metadata), add a baked NavGrid for pathfinding, add destructible terrain, and rename LocalGrid to FidelityZone.

**Architecture:** The codebase already has a chunk-based `VoxelWorld` in `src/world_sim/voxel.rs` with 16^3 chunks, 21 `VoxelMaterial` variants, terrain generation, mining, and SDF integration. Rather than introducing a new octree-based grid system, we extend `VoxelWorld` to absorb CityGrid's building/zone tracking. CityGrid (2D, settlement-tied) is then deleted. NavGrid is a new baked 2D walkable surface extracted from VoxelWorld. LocalGrid is renamed to FidelityZone (struct unchanged).

**Tech Stack:** Rust, serde (Serialize/Deserialize)

**Key Discovery:** The spec designed an octree `VoxelGrid` not knowing about the existing `VoxelWorld`. This plan adapts the spec's concepts (materials, destructible terrain, NavGrid, FidelityZone) to extend the existing chunk-based system instead. The existing `Voxel` struct (4 bytes: material, light, damage, flags) needs expansion to carry building_id and zone. The existing `VoxelMaterial` enum needs the spec's additional materials. The octree is not needed — chunks already provide sparse storage with lazy generation.

---

### Task 1: Extend VoxelMaterial with spec materials

**Files:**
- Modify: `src/world_sim/voxel.rs:86-120` (VoxelMaterial enum)
- Modify: `src/world_sim/voxel.rs:126-176` (is_solid, hardness, mine_yield methods)
- Test: `src/world_sim/voxel.rs` (inline tests module)

The spec defines 26 materials. The existing enum has 21. We need to add: Basalt, Sandstone, Marble, Bone, Brick, CutStone, Concrete, Ceramic, Steel, Bronze, Obsidian. Some spec materials already exist under different names (Stone=Stone, Iron=Iron, etc.). We also add `MaterialProperties` as a flat struct with a lookup function.

- [ ] **Step 1: Write failing test for new materials and properties**

Add to the `#[cfg(test)] mod tests` block at the bottom of `src/world_sim/voxel.rs`:

```rust
#[test]
fn material_properties_all_variants() {
    // Every VoxelMaterial variant must have properties defined
    let all = [
        VoxelMaterial::Air, VoxelMaterial::Stone, VoxelMaterial::Granite,
        VoxelMaterial::Basalt, VoxelMaterial::Sandstone, VoxelMaterial::Marble,
        VoxelMaterial::Dirt, VoxelMaterial::Clay, VoxelMaterial::Sand,
        VoxelMaterial::Gravel, VoxelMaterial::Ice, VoxelMaterial::Snow,
        VoxelMaterial::Water, VoxelMaterial::Lava,
        VoxelMaterial::WoodLog, VoxelMaterial::WoodPlanks, VoxelMaterial::Thatch,
        VoxelMaterial::Bone,
        VoxelMaterial::Brick, VoxelMaterial::CutStone, VoxelMaterial::Concrete,
        VoxelMaterial::Glass, VoxelMaterial::Ceramic,
        VoxelMaterial::Iron, VoxelMaterial::Steel, VoxelMaterial::Bronze,
        VoxelMaterial::CopperOre, VoxelMaterial::GoldOre,
        VoxelMaterial::Obsidian, VoxelMaterial::Crystal,
        VoxelMaterial::IronOre, VoxelMaterial::Coal,
        VoxelMaterial::Grass, VoxelMaterial::Farmland, VoxelMaterial::Crop,
        VoxelMaterial::StoneBlock, VoxelMaterial::StoneBrick,
    ];
    for mat in &all {
        let props = mat.properties();
        assert!(props.hp_multiplier > 0.0 || *mat == VoxelMaterial::Air);
        // All solid materials must have positive weight
        if mat.is_solid() {
            assert!(props.weight > 0.0, "{:?} should have positive weight", mat);
        }
    }
}

#[test]
fn material_properties_values() {
    let steel = VoxelMaterial::Steel.properties();
    assert!(steel.hp_multiplier > VoxelMaterial::WoodLog.properties().hp_multiplier);
    assert!(steel.blast_resistance > VoxelMaterial::Glass.properties().blast_resistance);
    assert!(VoxelMaterial::Glass.properties().load_bearing == false);
    assert!(VoxelMaterial::Stone.properties().load_bearing == true);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test material_properties_all_variants -- --nocapture 2>&1 | head -20`
Expected: FAIL — `Basalt`, `Sandstone`, etc. not found in VoxelMaterial; `properties()` method doesn't exist.

- [ ] **Step 3: Add new VoxelMaterial variants**

In `src/world_sim/voxel.rs`, add new variants to the `VoxelMaterial` enum after the existing ones. Insert these before the closing brace:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum VoxelMaterial {
    Air = 0,
    // Natural terrain
    Dirt,
    Stone,
    Granite,
    Basalt,
    Sandstone,
    Marble,
    Sand,
    Clay,
    Gravel,
    Grass,
    // Fluids
    Water,
    Lava,
    Ice,
    Snow,
    // Ores
    IronOre,
    CopperOre,
    GoldOre,
    Coal,
    Crystal,
    // Organic
    WoodLog,
    WoodPlanks,
    Thatch,
    Bone,
    // Worked
    StoneBlock,
    StoneBrick,
    Brick,
    CutStone,
    Concrete,
    Glass,
    Ceramic,
    // Metal
    Iron,
    Steel,
    Bronze,
    // Exotic
    Obsidian,
    // Agricultural
    Farmland,
    Crop,
}
```

Update `is_solid()` to include new solid materials (all new ones except Air/Water/Lava are solid — same pattern as existing). Update `hardness()` with values for new variants:

```rust
pub fn hardness(self) -> u32 {
    match self {
        VoxelMaterial::Air | VoxelMaterial::Water | VoxelMaterial::Lava => 0,
        VoxelMaterial::Dirt | VoxelMaterial::Grass | VoxelMaterial::Sand
        | VoxelMaterial::Farmland | VoxelMaterial::Crop | VoxelMaterial::Snow => 5,
        VoxelMaterial::Clay | VoxelMaterial::Gravel | VoxelMaterial::Thatch => 8,
        VoxelMaterial::Bone => 10,
        VoxelMaterial::WoodLog | VoxelMaterial::WoodPlanks => 15,
        VoxelMaterial::Sandstone => 20,
        VoxelMaterial::Coal => 20,
        VoxelMaterial::Brick | VoxelMaterial::Ceramic => 25,
        VoxelMaterial::Iron | VoxelMaterial::Glass | VoxelMaterial::Ice => 25,
        VoxelMaterial::Stone | VoxelMaterial::StoneBrick | VoxelMaterial::StoneBlock
        | VoxelMaterial::CutStone | VoxelMaterial::Concrete => 30,
        VoxelMaterial::IronOre | VoxelMaterial::CopperOre => 35,
        VoxelMaterial::Marble | VoxelMaterial::Basalt => 35,
        VoxelMaterial::Bronze => 40,
        VoxelMaterial::GoldOre | VoxelMaterial::Crystal => 40,
        VoxelMaterial::Steel => 50,
        VoxelMaterial::Obsidian => 60,
        VoxelMaterial::Granite => u32::MAX, // bedrock
    }
}
```

- [ ] **Step 4: Add MaterialProperties struct and properties() method**

Add after the `mine_yield` method on `VoxelMaterial`:

```rust
/// Physical properties of a material. Flat struct — add fields as needed.
#[derive(Debug, Clone, Copy)]
pub struct MaterialProperties {
    pub hp_multiplier: f32,
    pub fire_resistance: f32,
    pub load_bearing: bool,
    pub weight: f32,
    pub rubble_move_cost: f32,
    pub construction_cost: f32,
    pub blast_resistance: f32,
}

impl VoxelMaterial {
    pub fn properties(self) -> MaterialProperties {
        match self {
            // Natural
            VoxelMaterial::Air => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 0.0, load_bearing: false, weight: 0.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Stone => MaterialProperties { hp_multiplier: 100.0, fire_resistance: 1.0, load_bearing: true, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 5.0, blast_resistance: 0.6 },
            VoxelMaterial::Granite => MaterialProperties { hp_multiplier: 200.0, fire_resistance: 1.0, load_bearing: true, weight: 2.7, rubble_move_cost: 4.0, construction_cost: 10.0, blast_resistance: 0.9 },
            VoxelMaterial::Basalt => MaterialProperties { hp_multiplier: 150.0, fire_resistance: 1.0, load_bearing: true, weight: 2.8, rubble_move_cost: 3.5, construction_cost: 8.0, blast_resistance: 0.8 },
            VoxelMaterial::Sandstone => MaterialProperties { hp_multiplier: 60.0, fire_resistance: 0.9, load_bearing: true, weight: 2.2, rubble_move_cost: 2.0, construction_cost: 3.0, blast_resistance: 0.3 },
            VoxelMaterial::Marble => MaterialProperties { hp_multiplier: 80.0, fire_resistance: 1.0, load_bearing: true, weight: 2.6, rubble_move_cost: 3.0, construction_cost: 12.0, blast_resistance: 0.5 },
            VoxelMaterial::Dirt => MaterialProperties { hp_multiplier: 20.0, fire_resistance: 0.8, load_bearing: true, weight: 1.5, rubble_move_cost: 1.5, construction_cost: 1.0, blast_resistance: 0.1 },
            VoxelMaterial::Clay => MaterialProperties { hp_multiplier: 25.0, fire_resistance: 0.9, load_bearing: true, weight: 1.8, rubble_move_cost: 2.0, construction_cost: 2.0, blast_resistance: 0.15 },
            VoxelMaterial::Sand => MaterialProperties { hp_multiplier: 10.0, fire_resistance: 1.0, load_bearing: false, weight: 1.6, rubble_move_cost: 2.5, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Gravel => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 1.0, load_bearing: false, weight: 1.7, rubble_move_cost: 2.0, construction_cost: 1.0, blast_resistance: 0.1 },
            VoxelMaterial::Ice => MaterialProperties { hp_multiplier: 30.0, fire_resistance: 0.0, load_bearing: true, weight: 0.9, rubble_move_cost: 1.0, construction_cost: 0.0, blast_resistance: 0.2 },
            VoxelMaterial::Snow => MaterialProperties { hp_multiplier: 5.0, fire_resistance: 0.0, load_bearing: false, weight: 0.3, rubble_move_cost: 1.5, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Grass => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.3, load_bearing: true, weight: 1.4, rubble_move_cost: 1.0, construction_cost: 0.0, blast_resistance: 0.05 },
            // Fluids
            VoxelMaterial::Water => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 1.0, load_bearing: false, weight: 1.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            VoxelMaterial::Lava => MaterialProperties { hp_multiplier: 0.0, fire_resistance: 1.0, load_bearing: false, weight: 3.0, rubble_move_cost: 0.0, construction_cost: 0.0, blast_resistance: 0.0 },
            // Ores
            VoxelMaterial::IronOre => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 1.0, load_bearing: true, weight: 3.5, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.7 },
            VoxelMaterial::CopperOre => MaterialProperties { hp_multiplier: 100.0, fire_resistance: 1.0, load_bearing: true, weight: 3.2, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.6 },
            VoxelMaterial::GoldOre => MaterialProperties { hp_multiplier: 80.0, fire_resistance: 1.0, load_bearing: true, weight: 4.0, rubble_move_cost: 3.0, construction_cost: 0.0, blast_resistance: 0.5 },
            VoxelMaterial::Coal => MaterialProperties { hp_multiplier: 40.0, fire_resistance: 0.2, load_bearing: true, weight: 1.4, rubble_move_cost: 2.0, construction_cost: 0.0, blast_resistance: 0.2 },
            VoxelMaterial::Crystal => MaterialProperties { hp_multiplier: 50.0, fire_resistance: 0.8, load_bearing: false, weight: 2.3, rubble_move_cost: 2.5, construction_cost: 15.0, blast_resistance: 0.3 },
            // Organic
            VoxelMaterial::WoodLog => MaterialProperties { hp_multiplier: 40.0, fire_resistance: 0.2, load_bearing: true, weight: 0.7, rubble_move_cost: 2.0, construction_cost: 3.0, blast_resistance: 0.2 },
            VoxelMaterial::WoodPlanks => MaterialProperties { hp_multiplier: 35.0, fire_resistance: 0.2, load_bearing: true, weight: 0.5, rubble_move_cost: 1.5, construction_cost: 2.0, blast_resistance: 0.15 },
            VoxelMaterial::Thatch => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.1, load_bearing: false, weight: 0.2, rubble_move_cost: 1.0, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Bone => MaterialProperties { hp_multiplier: 30.0, fire_resistance: 0.5, load_bearing: true, weight: 1.0, rubble_move_cost: 2.0, construction_cost: 2.0, blast_resistance: 0.2 },
            // Worked
            VoxelMaterial::StoneBlock => MaterialProperties { hp_multiplier: 110.0, fire_resistance: 1.0, load_bearing: true, weight: 2.6, rubble_move_cost: 3.0, construction_cost: 7.0, blast_resistance: 0.7 },
            VoxelMaterial::StoneBrick => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 1.0, load_bearing: true, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 8.0, blast_resistance: 0.75 },
            VoxelMaterial::Brick => MaterialProperties { hp_multiplier: 90.0, fire_resistance: 1.0, load_bearing: true, weight: 2.0, rubble_move_cost: 2.5, construction_cost: 5.0, blast_resistance: 0.5 },
            VoxelMaterial::CutStone => MaterialProperties { hp_multiplier: 130.0, fire_resistance: 1.0, load_bearing: true, weight: 2.7, rubble_move_cost: 3.5, construction_cost: 10.0, blast_resistance: 0.8 },
            VoxelMaterial::Concrete => MaterialProperties { hp_multiplier: 140.0, fire_resistance: 1.0, load_bearing: true, weight: 2.4, rubble_move_cost: 4.0, construction_cost: 6.0, blast_resistance: 0.85 },
            VoxelMaterial::Glass => MaterialProperties { hp_multiplier: 10.0, fire_resistance: 0.7, load_bearing: false, weight: 2.5, rubble_move_cost: 3.0, construction_cost: 8.0, blast_resistance: 0.05 },
            VoxelMaterial::Ceramic => MaterialProperties { hp_multiplier: 50.0, fire_resistance: 1.0, load_bearing: false, weight: 2.0, rubble_move_cost: 2.5, construction_cost: 6.0, blast_resistance: 0.3 },
            // Metal
            VoxelMaterial::Iron => MaterialProperties { hp_multiplier: 150.0, fire_resistance: 0.8, load_bearing: true, weight: 7.8, rubble_move_cost: 4.0, construction_cost: 10.0, blast_resistance: 0.8 },
            VoxelMaterial::Steel => MaterialProperties { hp_multiplier: 200.0, fire_resistance: 0.9, load_bearing: true, weight: 7.9, rubble_move_cost: 4.5, construction_cost: 15.0, blast_resistance: 0.95 },
            VoxelMaterial::Bronze => MaterialProperties { hp_multiplier: 120.0, fire_resistance: 0.85, load_bearing: true, weight: 8.5, rubble_move_cost: 4.0, construction_cost: 12.0, blast_resistance: 0.7 },
            // Exotic
            VoxelMaterial::Obsidian => MaterialProperties { hp_multiplier: 70.0, fire_resistance: 1.0, load_bearing: true, weight: 2.4, rubble_move_cost: 3.5, construction_cost: 20.0, blast_resistance: 0.4 },
            // Agricultural
            VoxelMaterial::Farmland => MaterialProperties { hp_multiplier: 15.0, fire_resistance: 0.5, load_bearing: true, weight: 1.3, rubble_move_cost: 1.0, construction_cost: 1.0, blast_resistance: 0.05 },
            VoxelMaterial::Crop => MaterialProperties { hp_multiplier: 5.0, fire_resistance: 0.1, load_bearing: false, weight: 0.1, rubble_move_cost: 0.5, construction_cost: 0.0, blast_resistance: 0.0 },
        }
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test material_properties -- --nocapture`
Expected: PASS for both `material_properties_all_variants` and `material_properties_values`.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/voxel.rs
git commit -m "feat: extend VoxelMaterial with spec materials and MaterialProperties"
```

---

### Task 2: Extend Voxel struct with building metadata and integrity

**Files:**
- Modify: `src/world_sim/voxel.rs:182-218` (Voxel struct)
- Test: `src/world_sim/voxel.rs` (inline tests)

The existing `Voxel` is 4 bytes (material, light, damage, flags). We need to add `building_id: Option<u32>` and `zone: VoxelZone` for building tracking, plus `integrity: f32` for destructible terrain. This increases voxel size but is necessary for CityGrid replacement. We also add a `VoxelZone` enum.

- [ ] **Step 1: Write failing test for voxel building metadata**

```rust
#[test]
fn voxel_building_metadata() {
    let mut v = Voxel::new(VoxelMaterial::StoneBrick);
    assert_eq!(v.zone, VoxelZone::None);
    assert_eq!(v.building_id, None);
    assert_eq!(v.integrity, 1.0);

    v.building_id = Some(42);
    v.zone = VoxelZone::Residential;
    v.integrity = 0.75;

    assert_eq!(v.building_id, Some(42));
    assert_eq!(v.zone, VoxelZone::Residential);
    assert!((v.integrity - 0.75).abs() < f32::EPSILON);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test voxel_building_metadata -- --nocapture 2>&1 | head -20`
Expected: FAIL — `VoxelZone` not found, `building_id`/`zone`/`integrity` fields don't exist.

- [ ] **Step 3: Add VoxelZone enum**

Add before the `Voxel` struct in `src/world_sim/voxel.rs`:

```rust
/// Functional zone designation for building voxels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum VoxelZone {
    #[default]
    None = 0,
    Residential = 1,
    Commercial = 2,
    Industrial = 3,
    Military = 4,
    Agricultural = 5,
    Sacred = 6,
    Underground = 7,
}
```

- [ ] **Step 4: Extend Voxel struct**

Replace the existing `Voxel` struct and its `impl` blocks:

```rust
/// Per-voxel data. Carries material, structural integrity, and building metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Voxel {
    pub material: VoxelMaterial,
    /// Light level 0-15.
    pub light: u8,
    /// Mining damage accumulated (0-255). Breaks when >= hardness.
    pub damage: u8,
    /// Packed flags:
    /// - bits 0-3: water level (0-15, for fluid voxels)
    /// - bits 4-5: flow direction (0=none, 1=N, 2=E, 3=S... encoded)
    /// - bit 6: is_source (spring/sea boundary)
    /// - bit 7: is_support (load-bearing)
    pub flags: u8,
    /// Structural integrity 0.0 (destroyed) to 1.0 (full health).
    pub integrity: f32,
    /// Building entity ID if this voxel is part of a building.
    pub building_id: Option<u32>,
    /// Functional zone designation.
    pub zone: VoxelZone,
}

impl Default for Voxel {
    fn default() -> Self {
        Self {
            material: VoxelMaterial::Air,
            light: 0,
            damage: 0,
            flags: 0,
            integrity: 1.0,
            building_id: None,
            zone: VoxelZone::None,
        }
    }
}

impl Voxel {
    pub fn new(material: VoxelMaterial) -> Self {
        Self {
            material,
            light: 0,
            damage: 0,
            flags: 0,
            integrity: 1.0,
            building_id: None,
            zone: VoxelZone::None,
        }
    }

    pub fn water_level(self) -> u8 { self.flags & 0x0F }
    pub fn set_water_level(&mut self, level: u8) {
        self.flags = (self.flags & 0xF0) | (level & 0x0F);
    }
    pub fn is_source(self) -> bool { self.flags & 0x40 != 0 }
    pub fn set_source(&mut self, v: bool) {
        if v { self.flags |= 0x40; } else { self.flags &= !0x40; }
    }

    /// Effective HP = integrity * material hp_multiplier.
    pub fn effective_hp(&self) -> f32 {
        self.integrity * self.material.properties().hp_multiplier
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test voxel_building_metadata -- --nocapture`
Expected: PASS.

Run: `cargo test -p bevy_game 2>&1 | tail -5`
Expected: All existing voxel tests still pass (they use `Voxel::new()` and `Voxel::default()` which remain compatible).

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/voxel.rs
git commit -m "feat: extend Voxel with integrity, building_id, zone for CityGrid replacement"
```

---

### Task 3: Add destructible terrain to VoxelWorld

**Files:**
- Modify: `src/world_sim/voxel.rs` (VoxelWorld impl)
- Test: `src/world_sim/voxel.rs` (inline tests)

Per the spec: damage reduces integrity, when integrity hits 0 the voxel becomes rubble (load-bearing) or air (non-load-bearing), and cascading collapse checks voxels above.

- [ ] **Step 1: Write failing test for terrain damage**

```rust
#[test]
fn voxel_world_damage_destroys() {
    let mut world = VoxelWorld::default();
    let cp = ChunkPos::new(0, 0, 0);
    world.generate_chunk(cp, 42);

    // Find a solid voxel that isn't bedrock
    let vx = 8;
    let vy = 8;
    let vz = 10; // stone layer
    let mat = world.get_voxel(vx, vy, vz).material;
    assert!(mat.is_solid(), "expected solid at z=10");

    let hp = mat.properties().hp_multiplier;
    // Damage it to destruction
    let destroyed = world.damage_voxel(vx, vy, vz, hp + 1.0);
    assert!(destroyed);

    let after = world.get_voxel(vx, vy, vz);
    // Load-bearing materials become rubble (integrity 0), non-load-bearing become Air
    if mat.properties().load_bearing {
        assert_eq!(after.integrity, 0.0);
        assert_eq!(after.zone, VoxelZone::None);
    } else {
        assert_eq!(after.material, VoxelMaterial::Air);
    }
}

#[test]
fn voxel_world_damage_partial() {
    let mut world = VoxelWorld::default();
    let cp = ChunkPos::new(0, 0, 0);
    world.generate_chunk(cp, 42);

    let vx = 8;
    let vy = 8;
    let vz = 10;
    let mat = world.get_voxel(vx, vy, vz).material;
    let hp = mat.properties().hp_multiplier;

    // Partial damage
    let destroyed = world.damage_voxel(vx, vy, vz, hp * 0.3);
    assert!(!destroyed);

    let after = world.get_voxel(vx, vy, vz);
    assert!(after.integrity > 0.0 && after.integrity < 1.0);
}

#[test]
fn cascading_collapse() {
    let mut world = VoxelWorld::default();
    // Build a column: stone at z=0,1,2
    for z in 0..3 {
        world.set_voxel(0, 0, z, Voxel::new(VoxelMaterial::Stone));
    }
    // Destroy the bottom support (z=0)
    let hp = VoxelMaterial::Stone.properties().hp_multiplier;
    world.damage_voxel(0, 0, 0, hp + 1.0);

    // z=1 should collapse (no support below, no horizontal neighbors)
    let v1 = world.get_voxel(0, 0, 1);
    assert_eq!(v1.integrity, 0.0, "z=1 should collapse");

    // z=2 should also collapse
    let v2 = world.get_voxel(0, 0, 2);
    assert_eq!(v2.integrity, 0.0, "z=2 should cascade");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test voxel_world_damage -- --nocapture 2>&1 | head -20`
Expected: FAIL — `damage_voxel` method doesn't exist on VoxelWorld.

- [ ] **Step 3: Implement damage_voxel and cascading collapse on VoxelWorld**

Add to the `impl VoxelWorld` block:

```rust
/// Apply structural damage to a voxel. Returns true if the voxel was destroyed.
/// Triggers cascading collapse for unsupported voxels above.
pub fn damage_voxel(&mut self, vx: i32, vy: i32, vz: i32, damage: f32) -> bool {
    let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
    let chunk = match self.chunks.get_mut(&cp) {
        Some(c) => c,
        None => return false,
    };
    let voxel = &mut chunk.voxels[idx];
    if !voxel.material.is_solid() { return false; }

    let props = voxel.material.properties();
    let effective_hp = voxel.integrity * props.hp_multiplier;
    let new_hp = effective_hp - damage;

    if new_hp <= 0.0 {
        // Destroyed
        if props.load_bearing {
            // Becomes rubble: same material, integrity 0, zone cleared
            voxel.integrity = 0.0;
            voxel.zone = VoxelZone::None;
            voxel.building_id = None;
        } else {
            // Non-load-bearing → air
            *voxel = Voxel::default();
        }
        chunk.dirty = true;

        // Cascading collapse: check voxels above
        self.cascade_collapse(vx, vy, vz + 1);
        true
    } else {
        voxel.integrity = new_hp / props.hp_multiplier;
        chunk.dirty = true;
        false
    }
}

/// Check structural support for voxel at (vx, vy, vz) and collapse if unsupported.
/// Recurses upward.
fn cascade_collapse(&mut self, vx: i32, vy: i32, vz: i32) {
    let voxel = self.get_voxel(vx, vy, vz);
    if !voxel.material.is_solid() || voxel.integrity == 0.0 { return; }

    if self.is_supported(vx, vy, vz) { return; }

    // Collapse this voxel
    let (cp, idx) = voxel_to_chunk_local(vx, vy, vz);
    if let Some(chunk) = self.chunks.get_mut(&cp) {
        let v = &mut chunk.voxels[idx];
        if v.material.properties().load_bearing {
            v.integrity = 0.0;
            v.zone = VoxelZone::None;
            v.building_id = None;
        } else {
            *v = Voxel::default();
        }
        chunk.dirty = true;
    }

    // Check above
    self.cascade_collapse(vx, vy, vz + 1);
}

/// Check if a voxel position is structurally supported.
/// Supported if: below is solid with integrity > 0, OR 2+ horizontal neighbors are solid.
/// Z=0 voxels are always supported (bedrock).
fn is_supported(&self, vx: i32, vy: i32, vz: i32) -> bool {
    if vz <= 0 { return true; } // bedrock

    // Check below
    let below = self.get_voxel(vx, vy, vz - 1);
    if below.material.is_solid() && below.integrity > 0.0 && below.material.properties().load_bearing {
        return true;
    }

    // Check 4 horizontal neighbors
    let mut solid_neighbors = 0u8;
    for &(dx, dy) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
        let n = self.get_voxel(vx + dx, vy + dy, vz);
        if n.material.is_solid() && n.integrity > 0.0 {
            solid_neighbors += 1;
        }
    }
    solid_neighbors >= 2
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test voxel_world_damage -- --nocapture && cargo test cascading_collapse -- --nocapture`
Expected: All three tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/voxel.rs
git commit -m "feat: add destructible terrain with cascading collapse to VoxelWorld"
```

---

### Task 4: Create NavGrid module

**Files:**
- Create: `src/world_sim/nav_grid.rs`
- Modify: `src/world_sim/mod.rs:9` (add `pub mod nav_grid;`)
- Test: inline in `src/world_sim/nav_grid.rs`

NavGrid is a baked 2D walkable surface graph derived from VoxelWorld. Each (x, y) column has a walkable flag, surface z, and move cost. Pathfinding (A* and flow fields) operates on NavGrid.

- [ ] **Step 1: Write the NavGrid module with tests first**

Create `src/world_sim/nav_grid.rs`:

```rust
//! NavGrid — baked 2D walkable surface derived from VoxelWorld.
//!
//! Each (x, y) position stores whether it's walkable, the surface z level,
//! and movement cost. Pathfinding (A* and flow fields) operates on NavGrid,
//! not on VoxelWorld directly.
//!
//! Rebaked when VoxelWorld undergoes structural changes.

use serde::{Deserialize, Serialize};

use super::voxel::VoxelWorld;

/// A baked 2D walkable surface from a region of VoxelWorld.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavGrid {
    /// Origin in voxel-space (min corner).
    pub origin_vx: i32,
    pub origin_vy: i32,
    pub width: u32,
    pub height: u32,
    pub nodes: Vec<NavNode>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NavNode {
    pub walkable: bool,
    /// Z of the walkable surface (top of highest solid voxel).
    pub surface_z: i32,
    /// Movement cost (material-based, 0.0 for non-walkable).
    pub move_cost: f32,
}

impl Default for NavNode {
    fn default() -> Self {
        Self { walkable: false, surface_z: 0, move_cost: 0.0 }
    }
}

impl NavGrid {
    /// Bake a NavGrid from a rectangular region of VoxelWorld.
    /// Scans each (x, y) column from top (max_z) down to find the surface.
    pub fn bake(world: &VoxelWorld, origin_vx: i32, origin_vy: i32, width: u32, height: u32, max_z: i32) -> Self {
        let mut nodes = vec![NavNode::default(); (width * height) as usize];

        for dy in 0..height {
            for dx in 0..width {
                let vx = origin_vx + dx as i32;
                let vy = origin_vy + dy as i32;
                let idx = (dy * width + dx) as usize;

                // Find surface: highest solid voxel in column
                let mut surface_z = -1i32;
                for vz in (0..=max_z).rev() {
                    let v = world.get_voxel(vx, vy, vz);
                    if v.material.is_solid() && v.integrity > 0.0 {
                        surface_z = vz;
                        break;
                    }
                }

                if surface_z < 0 {
                    // No solid surface in this column
                    nodes[idx] = NavNode { walkable: false, surface_z: 0, move_cost: 0.0 };
                    continue;
                }

                // Check that the cell above the surface is air (walkable space)
                let above = world.get_voxel(vx, vy, surface_z + 1);
                let walkable = !above.material.is_solid();

                let surface_mat = world.get_voxel(vx, vy, surface_z);
                let props = surface_mat.material.properties();
                let move_cost = if walkable {
                    if surface_mat.integrity == 0.0 {
                        // Rubble
                        1.0 + props.rubble_move_cost
                    } else {
                        1.0 // normal surface
                    }
                } else {
                    0.0
                };

                nodes[idx] = NavNode { walkable, surface_z, move_cost };
            }
        }

        Self { origin_vx, origin_vy, width, height, nodes }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    #[inline]
    pub fn in_bounds(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    pub fn is_walkable(&self, x: u32, y: u32) -> bool {
        if !self.in_bounds(x, y) { return false; }
        self.nodes[self.idx(x, y)].walkable
    }

    pub fn surface_z_at(&self, x: u32, y: u32) -> i32 {
        if !self.in_bounds(x, y) { return 0; }
        self.nodes[self.idx(x, y)].surface_z
    }

    pub fn move_cost(&self, x: u32, y: u32) -> f32 {
        if !self.in_bounds(x, y) { return f32::MAX; }
        let node = &self.nodes[self.idx(x, y)];
        if !node.walkable { return f32::MAX; }
        node.move_cost
    }

    /// A* pathfinding. Returns path of (x, y) positions from start to goal,
    /// or None if no path exists. 8-connected. Max 1000 iterations.
    pub fn find_path(&self, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<(u32, u32)>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        if !self.in_bounds(start.0, start.1) || !self.in_bounds(goal.0, goal.1) {
            return None;
        }
        if start == goal { return Some(vec![goal]); }

        let n = (self.width * self.height) as usize;
        let mut g_score = vec![f32::MAX; n];
        let mut came_from = vec![u32::MAX; n];
        let mut open: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();

        let start_idx = self.idx(start.0, start.1);
        let goal_idx = self.idx(goal.0, goal.1);
        g_score[start_idx] = 0.0;

        let heuristic = |idx: usize| -> f32 {
            let x = (idx % self.width as usize) as f32;
            let y = (idx / self.width as usize) as f32;
            (x - goal.0 as f32).abs() + (y - goal.1 as f32).abs()
        };

        let f0 = (heuristic(start_idx) * 100.0) as u32;
        open.push(Reverse((f0, start_idx as u32)));

        let mut iterations = 0u32;
        const MAX_ITER: u32 = 1000;

        while let Some(Reverse((_, current_u32))) = open.pop() {
            let current = current_u32 as usize;
            if current == goal_idx {
                let mut path = Vec::new();
                let mut c = current;
                while c != start_idx {
                    path.push(((c % self.width as usize) as u32, (c / self.width as usize) as u32));
                    c = came_from[c] as usize;
                    if c == u32::MAX as usize { return None; }
                }
                path.reverse();
                return Some(path);
            }

            iterations += 1;
            if iterations >= MAX_ITER { return None; }

            let cx = (current % self.width as usize) as i32;
            let cy = (current / self.width as usize) as i32;
            let current_g = g_score[current];

            for &(dx, dy) in &[
                (-1i32, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ] {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                let nx_u = nx as u32;
                let ny_u = ny as u32;
                let ni = self.idx(nx_u, ny_u);

                // Goal is always reachable
                if ni != goal_idx && !self.is_walkable(nx_u, ny_u) { continue; }

                // Diagonal corner-cutting prevention
                if dx != 0 && dy != 0 {
                    if !self.is_walkable(cx as u32, ny_u) || !self.is_walkable(nx_u, cy as u32) {
                        continue;
                    }
                }

                let step_cost = if dx != 0 && dy != 0 {
                    self.move_cost(nx_u, ny_u) * 1.414
                } else {
                    self.move_cost(nx_u, ny_u)
                };
                if step_cost >= f32::MAX { continue; }

                let tentative_g = current_g + step_cost;
                if tentative_g < g_score[ni] {
                    g_score[ni] = tentative_g;
                    came_from[ni] = current as u32;
                    let f = (tentative_g + heuristic(ni)) * 100.0;
                    open.push(Reverse((f as u32, ni as u32)));
                }
            }
        }
        None
    }

    /// BFS flow field toward a target. Returns direction map where
    /// `flow[idx] = next_idx` (u32::MAX if unreachable). 4-connected.
    pub fn compute_flow_field(&self, target: (u32, u32)) -> Vec<u32> {
        let n = (self.width * self.height) as usize;
        let mut flow = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::new();

        let target_idx = self.idx(target.0, target.1);
        flow[target_idx] = target_idx as u32;
        visited[target_idx] = true;
        queue.push_back(target_idx);

        while let Some(current) = queue.pop_front() {
            let cx = (current % self.width as usize) as i32;
            let cy = (current / self.width as usize) as i32;

            for &(dx, dy) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 { continue; }
                let ni = self.idx(nx as u32, ny as u32);
                if visited[ni] { continue; }
                if !self.is_walkable(nx as u32, ny as u32) { continue; }

                visited[ni] = true;
                flow[ni] = current as u32;
                queue.push_back(ni);
            }
        }
        flow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::voxel::{Voxel, VoxelMaterial};

    fn make_flat_world() -> VoxelWorld {
        let mut world = VoxelWorld::default();
        // Create a flat surface at z=5 (stone) with air above
        for y in 0..10 {
            for x in 0..10 {
                for z in 0..=5 {
                    world.set_voxel(x, y, z, Voxel::new(VoxelMaterial::Stone));
                }
            }
        }
        world
    }

    #[test]
    fn bake_flat_surface() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        assert_eq!(nav.width, 10);
        assert_eq!(nav.height, 10);

        // All cells should be walkable with surface at z=5
        for y in 0..10 {
            for x in 0..10 {
                assert!(nav.is_walkable(x, y), "({}, {}) should be walkable", x, y);
                assert_eq!(nav.surface_z_at(x, y), 5);
                assert!(nav.move_cost(x, y) > 0.0);
            }
        }
    }

    #[test]
    fn find_path_simple() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        let path = nav.find_path((0, 0), (5, 5));
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.last().unwrap(), (5, 5));
    }

    #[test]
    fn find_path_blocked() {
        let mut world = make_flat_world();
        // Place a wall across y=5 (solid blocks above the surface)
        for x in 0..10 {
            world.set_voxel(x, 5, 6, Voxel::new(VoxelMaterial::StoneBlock));
        }
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        // y=5 should now have surface_z=6 and be walkable on top of the wall
        // But a path from (0,0) to (0,9) should still exist (go over the wall or around)
        let path = nav.find_path((0, 0), (0, 9));
        // The wall is only 1 high, so the surface shifts up. Path should exist.
        assert!(path.is_some());
    }

    #[test]
    fn flow_field_basic() {
        let world = make_flat_world();
        let nav = NavGrid::bake(&world, 0, 0, 10, 10, 20);

        let flow = nav.compute_flow_field((5, 5));
        let target_idx = nav.idx(5, 5);

        // Target points to itself
        assert_eq!(flow[target_idx], target_idx as u32);

        // All walkable cells should be reachable (flow != MAX)
        for y in 0..10u32 {
            for x in 0..10u32 {
                assert_ne!(flow[nav.idx(x, y)], u32::MAX, "({}, {}) should be reachable", x, y);
            }
        }
    }
}
```

- [ ] **Step 2: Add module declaration**

In `src/world_sim/mod.rs`, add after the `pub mod voxel;` line:

```rust
pub mod nav_grid;
```

- [ ] **Step 3: Run tests**

Run: `cargo test nav_grid -- --nocapture`
Expected: All 4 NavGrid tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/nav_grid.rs src/world_sim/mod.rs
git commit -m "feat: add NavGrid — baked 2D walkable surface with A* and flow fields"
```

---

### Task 5: Rename LocalGrid to FidelityZone

**Files:**
- Modify: `src/world_sim/state.rs:3750-3779` (struct rename)
- Modify: `src/world_sim/state.rs:379` (field rename)
- Modify: `src/world_sim/state.rs:759,821` (accessor methods)
- Modify: `src/world_sim/state.rs:459` (WorldState::new initializer)
- Modify: `src/world_sim/mod.rs:36` (re-export)
- Modify: `src/world_sim/tick.rs:293,355,373,394` (usage)
- Modify: `src/world_sim/runtime.rs:2102` (usage)
- Modify: `src/world_sim/compute_high.rs:168` (usage)
- Modify: `src/world_sim/systems/battles.rs:54` (usage)

Pure rename — no logic changes. The struct stays the same, just `LocalGrid` → `FidelityZone` and `grids` → `fidelity_zones`.

- [ ] **Step 1: Rename the struct and field in state.rs**

In `src/world_sim/state.rs`:

1. Line 3750: Change comment from `// LocalGrid` to `// FidelityZone`
2. Line 3754: Rename `pub struct LocalGrid` → `pub struct FidelityZone`
3. Line 3763: Rename `impl LocalGrid` → `impl FidelityZone`
4. Line 379: Rename `pub grids: Vec<LocalGrid>` → `pub fidelity_zones: Vec<FidelityZone>`
5. Line 459: Rename `grids: Vec::new()` → `fidelity_zones: Vec::new()`
6. Line 759: Change `grid_mut` to `fidelity_zone_mut`, update return type from `Option<&mut LocalGrid>` to `Option<&mut FidelityZone>`, update `self.grids.iter_mut()` to `self.fidelity_zones.iter_mut()`
7. Line 821: Change `grid` to `fidelity_zone`, update return type and body similarly

- [ ] **Step 2: Update re-export in mod.rs**

In `src/world_sim/mod.rs` line 36: change `LocalGrid` to `FidelityZone`.

- [ ] **Step 3: Update all consumer files**

Each file needs `LocalGrid` → `FidelityZone` and `state.grids` → `state.fidelity_zones` and `grid_mut`/`grid` → `fidelity_zone_mut`/`fidelity_zone`:

**`src/world_sim/tick.rs`:**
- Line 293: Change parameter type `grid: &super::state::LocalGrid` → `grid: &super::state::FidelityZone`
- Lines 355, 373, 394: Change `state.grids.push(LocalGrid {` → `state.fidelity_zones.push(FidelityZone {`
- Import: `LocalGrid` → `FidelityZone`

**`src/world_sim/runtime.rs`:**
- Line 2102: Change `grid: &super::state::LocalGrid` → `grid: &super::state::FidelityZone`
- Any `state.grids` → `state.fidelity_zones`

**`src/world_sim/compute_high.rs`:**
- Line 168: Change `s.grids.push(LocalGrid {` → `s.fidelity_zones.push(FidelityZone {`

**`src/world_sim/systems/battles.rs`:**
- Line 54: Change `grid: &crate::world_sim::state::LocalGrid` → `grid: &crate::world_sim::state::FidelityZone`

- [ ] **Step 4: Build to verify**

Run: `cargo build 2>&1 | tail -10`
Expected: Build succeeds with no errors.

- [ ] **Step 5: Run all tests**

Run: `cargo test 2>&1 | tail -5`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/world_sim/state.rs src/world_sim/mod.rs src/world_sim/tick.rs src/world_sim/runtime.rs src/world_sim/compute_high.rs src/world_sim/systems/battles.rs
git commit -m "refactor: rename LocalGrid to FidelityZone"
```

---

### Task 6: Add NavGrid and FidelityZone storage to WorldState

**Files:**
- Modify: `src/world_sim/state.rs:375-465` (WorldState fields + new())
- Modify: `src/world_sim/mod.rs` (re-export NavGrid)
- Test: compile check

Add `nav_grids: Vec<NavGrid>` to WorldState alongside the existing `voxel_world`. This gives the building AI a baked walkable surface to query without touching VoxelWorld directly.

- [ ] **Step 1: Add nav_grids field to WorldState**

In `src/world_sim/state.rs`, after the `voxel_world` field (around line 406), add:

```rust
    /// Baked 2D walkable surfaces derived from voxel_world. One per settlement area.
    pub nav_grids: Vec<super::nav_grid::NavGrid>,
```

In `WorldState::new()`, add `nav_grids: Vec::new(),` to the initializer.

- [ ] **Step 2: Add re-export in mod.rs**

In `src/world_sim/mod.rs`, add to the `pub use` block or as standalone:

```rust
pub use nav_grid::NavGrid;
```

- [ ] **Step 3: Build to verify**

Run: `cargo build 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/state.rs src/world_sim/mod.rs
git commit -m "feat: add nav_grids storage to WorldState"
```

---

### Task 7: Migrate buildings.rs from CityGrid to VoxelWorld

**Files:**
- Modify: `src/world_sim/systems/buildings.rs:1-360`
- Test: `cargo build` + `cargo test`

This is the largest migration. `process_npc_builds()` currently reads/writes `CityGrid` cells. We migrate it to use `VoxelWorld` for building placement and `NavGrid` for finding buildable positions. The `grow_cities()` function is already a stub (line 104-111), so it just needs import cleanup.

Key changes:
- `CityGrid::frontier` → scan `NavGrid` for walkable cells with no building_id above
- `CityGrid::cell_mut().state = CellState::Building` → `VoxelWorld::set_voxel()` with building material + building_id
- `CityGrid::grid_to_world(col, row, settlement_pos)` → `voxel::voxel_to_world(vx, vy, vz)`
- `building_type_to_zone(bt) -> ZoneType` → `building_type_to_voxel_zone(bt) -> VoxelZone`
- Remove `city_grid_idx` lookups

- [ ] **Step 1: Update imports**

Replace the import block at top of `src/world_sim/systems/buildings.rs`:

```rust
use crate::world_sim::voxel::{VoxelMaterial, VoxelZone, world_to_voxel, voxel_to_world};
use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, Entity, EntityKind, EconomicIntent, ChronicleEntry, ChronicleCategory, tags, BuildingType, BuildingData, SettlementSpecialty, ActionTags, WorkState, MemoryEvent, MemEventType, GoalKind, entity_hash};
use crate::world_sim::NUM_COMMODITIES;
```

Remove the `use crate::world_sim::city_grid::...` line entirely.

- [ ] **Step 2: Replace building_type_to_zone with VoxelZone version**

Replace the `building_type_to_zone` function:

```rust
fn building_type_to_voxel_zone(bt: BuildingType) -> VoxelZone {
    match bt {
        BuildingType::House | BuildingType::Longhouse | BuildingType::Manor => VoxelZone::Residential,
        BuildingType::Market | BuildingType::Warehouse | BuildingType::TradePost | BuildingType::Inn => VoxelZone::Commercial,
        BuildingType::Farm | BuildingType::Mine | BuildingType::Sawmill | BuildingType::Forge
        | BuildingType::Workshop | BuildingType::Apothecary => VoxelZone::Industrial,
        BuildingType::Barracks | BuildingType::Watchtower | BuildingType::Wall | BuildingType::Gate => VoxelZone::Military,
        BuildingType::Temple | BuildingType::Shrine => VoxelZone::Sacred,
        _ => VoxelZone::None,
    }
}
```

- [ ] **Step 3: Rewrite process_npc_builds to use VoxelWorld**

Replace the body of `process_npc_builds()`. The new version finds buildable positions by scanning the settlement area in VoxelWorld for surface cells without buildings:

```rust
pub fn process_npc_builds(state: &mut WorldState) {
    use crate::world_sim::commodity;
    use super::super::interior_gen::footprint_size;

    let mut build_requests: Vec<(usize, u32, BuildingType)> = Vec::new();

    for (i, entity) in state.entities.iter().enumerate() {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        let sid = match npc.home_settlement_id { Some(s) => s, None => continue };

        let build_type = npc.goal_stack.goals.iter().find_map(|g| {
            match g.kind {
                GoalKind::Build { .. } => Some(BuildingType::House),
                _ => None,
            }
        });
        let building_type = match build_type { Some(bt) => bt, None => continue };

        let inv = match &entity.inventory { Some(inv) => inv, None => continue };
        let (wood_cost, iron_cost) = building_type.build_cost();
        if inv.commodities[commodity::WOOD] < wood_cost { continue; }
        if inv.commodities[commodity::IRON] < iron_cost { continue; }

        build_requests.push((i, sid, building_type));
    }

    if build_requests.is_empty() { return; }

    let tick = state.tick;
    let mut new_entities: Vec<Entity> = Vec::new();

    for (entity_idx, settlement_id, building_type) in build_requests {
        let settlement_pos = match state.settlement(settlement_id) {
            Some(s) => s.pos,
            None => continue,
        };

        let npc_pos = state.entities[entity_idx].pos;
        let (fp_w, fp_h) = footprint_size(building_type, 0);

        // Find buildable position near the NPC using VoxelWorld surface scan.
        // Search in a 60x60 voxel area centered on the settlement.
        let (cx, cy, _cz) = world_to_voxel(settlement_pos.0, settlement_pos.1, 0.0);
        let search_radius = 30i32;

        let mut best_pos: Option<(i32, i32, i32)> = None;
        let mut best_dist = f32::MAX;

        for dy in -search_radius..search_radius {
            for dx in -search_radius..search_radius {
                let vx = cx + dx;
                let vy = cy + dy;
                let vz = state.voxel_world.surface_height(vx, vy);

                // Check footprint fits (all cells must be surface-level, no existing buildings)
                let mut fits = true;
                for fy in 0..fp_h as i32 {
                    for fx in 0..fp_w as i32 {
                        let sv = state.voxel_world.get_voxel(vx + fx, vy + fy, vz);
                        let above = state.voxel_world.get_voxel(vx + fx, vy + fy, vz + 1);
                        if !sv.material.is_solid() || sv.building_id.is_some() || above.material.is_solid() {
                            fits = false;
                            break;
                        }
                        // Also check surface is roughly level (same z ±1)
                        let neighbor_z = state.voxel_world.surface_height(vx + fx, vy + fy);
                        if (neighbor_z - vz).abs() > 1 {
                            fits = false;
                            break;
                        }
                    }
                    if !fits { break; }
                }
                if !fits { continue; }

                let (wx, wy, _) = voxel_to_world(vx, vy, vz);
                let dist = (wx - npc_pos.0).powi(2) + (wy - npc_pos.1).powi(2);
                if dist < best_dist {
                    best_dist = dist;
                    best_pos = Some((vx, vy, vz));
                }
            }
        }

        let (bx, by, bz) = match best_pos { Some(p) => p, None => continue };

        // Deduct resources
        if let Some(inv) = &mut state.entities[entity_idx].inventory {
            let (wood_cost, iron_cost) = building_type.build_cost();
            inv.commodities[commodity::WOOD] -= wood_cost;
            inv.commodities[commodity::IRON] -= iron_cost;
        }

        // Spawn building entity
        state.sync_next_id();
        let new_id = state.next_entity_id();
        let (wx, wy, wz) = voxel_to_world(bx, by, bz + 1);
        let world_pos = (wx, wy);
        let zone = building_type_to_voxel_zone(building_type);

        let mut entity = Entity::new_building(new_id, world_pos);
        entity.building = Some(BuildingData {
            building_type,
            settlement_id: Some(settlement_id),
            grid_col: bx as u16,
            grid_row: by as u16,
            footprint_w: fp_w as u8,
            footprint_h: fp_h as u8,
            tier: 0,
            room_seed: entity_hash(new_id, tick, 0x800E) as u64,
            rooms: building_type.default_rooms(),
            residential_capacity: building_type.residential_capacity(),
            work_capacity: building_type.work_capacity(),
            resident_ids: Vec::new(),
            worker_ids: Vec::new(),
            construction_progress: 0.0,
            built_tick: tick,
            builder_id: Some(state.entities[entity_idx].id),
            temporary: false,
            ttl_ticks: None,
            name: generate_building_name(building_type, new_id),
            storage: [0.0; NUM_COMMODITIES],
            storage_capacity: building_type.storage_capacity(),
            owner_id: Some(state.entities[entity_idx].id),
            builder_modifiers: Vec::new(),
            owner_modifiers: Vec::new(),
            worker_class_ticks: Vec::new(),
            specialization_tag: None,
            specialization_strength: 0.0,
            specialization_name: String::new(),
            structural: None,
        });

        // Mark voxels as building
        let build_material = match building_type {
            BuildingType::Farm | BuildingType::Sawmill => VoxelMaterial::WoodLog,
            BuildingType::Mine | BuildingType::Forge => VoxelMaterial::StoneBlock,
            _ => VoxelMaterial::StoneBrick,
        };
        for fy in 0..fp_h as i32 {
            for fx in 0..fp_w as i32 {
                // Place building voxel one above surface
                let mut v = crate::world_sim::voxel::Voxel::new(build_material);
                v.building_id = Some(new_id);
                v.zone = zone;
                state.voxel_world.set_voxel(bx + fx, by + fy, bz + 1, v);
            }
        }

        new_entities.push(entity);

        // Pop Build goal
        if let Some(npc) = &mut state.entities[entity_idx].npc {
            npc.goal_stack.goals.retain(|g| !matches!(g.kind, GoalKind::Build { .. }));
        }

        // Chronicle
        let npc_name = state.entities[entity_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Economy,
            text: format!("{} began building a {:?}", npc_name, building_type),
            entity_ids: vec![state.entities[entity_idx].id, new_id],
        });
    }

    if !new_entities.is_empty() {
        for e in new_entities {
            state.entities.push(e);
        }
        state.rebuild_entity_cache();
    }
}
```

- [ ] **Step 4: Clean up grow_cities()**

The function is already a stub (returns early at line 111). Remove the dead code after `return;` and clean up imports. The function body becomes:

```rust
pub fn grow_cities(state: &mut WorldState) {
    if state.tick % GROWTH_TICK_INTERVAL != 0 {
        return;
    }
    // CityGrid growth disabled — building placement uses VoxelWorld directly.

    // Advance construction on incomplete buildings (builders do physical work).
    advance_construction(state);

    // Assign unhoused/unassigned NPCs to buildings.
    assign_npcs_to_buildings(state);

    // NPC-driven building placement.
    process_npc_builds(state);
}
```

- [ ] **Step 5: Build to verify**

Run: `cargo build 2>&1 | tail -10`
Expected: Build succeeds. There may be warnings about unused imports in other files — that's fine for now.

- [ ] **Step 6: Run tests**

Run: `cargo test 2>&1 | tail -10`
Expected: Tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/systems/buildings.rs
git commit -m "refactor: migrate buildings.rs from CityGrid to VoxelWorld"
```

---

### Task 8: Migrate building_ai modules from CityGrid

**Files:**
- Modify: `src/world_sim/building_ai/scoring.rs`
- Modify: `src/world_sim/building_ai/features.rs`
- Modify: `src/world_sim/building_ai/scenario_gen.rs`
- Modify: `src/world_sim/building_ai/mass_gen.rs`
- Modify: `src/world_sim/building_ai/validation/action.rs`
- Modify: `src/world_sim/building_ai/validation/world_state.rs`
- Modify: `src/world_sim/building_ai/validation/features.rs`
- Modify: `src/world_sim/building_ai/validation/probes.rs`
- Test: `cargo build && cargo test`

These files have the densest CityGrid usage. The migration pattern is the same: `CityGrid` lookups → `VoxelWorld` queries, `city_grid_idx` → direct voxel operations, `CellState/ZoneType` → `VoxelMaterial/VoxelZone`.

This is a large task. The implementation subagent should:
1. Read each file fully before modifying
2. Replace `use crate::world_sim::city_grid::*` imports with `use crate::world_sim::voxel::*`
3. Replace `state.city_grids[gi]` patterns with `state.voxel_world`
4. Replace `CityGrid::new(128, 128, ...)` with `VoxelWorld::ensure_loaded_around()` calls
5. Replace `cell.state == CellState::Building` with `voxel.building_id.is_some()`
6. Replace `grid_to_world(col, row, settlement_pos)` with `voxel_to_world(vx, vy, vz)`
7. Replace `ZoneType` references with `VoxelZone`
8. Replace `CellState` references with voxel material checks
9. Replace `settlement.city_grid_idx` with direct voxel_world access (settlements have `pos` which gives voxel coordinates)
10. Update all tests that create `CityGrid::new(...)` to instead populate `VoxelWorld` directly

The implementer should be model=opus for this task (multi-file integration with judgment calls).

- [ ] **Step 1: Migrate scoring.rs**

Replace `use crate::world_sim::city_grid::CellState` with `use crate::world_sim::voxel::{VoxelZone, world_to_voxel, voxel_to_world}`. Update the `apply_actions` function to write voxels instead of grid cells. Update zone string matching to use `VoxelZone`.

- [ ] **Step 2: Migrate features.rs**

Replace `CellState, CellTerrain, ZoneType` imports. Update `compute_spatial_features()` to query `voxel_world` instead of `city_grids[grid_idx]`. The connectivity analysis should scan voxels for building_id presence instead of checking CellState.

- [ ] **Step 3: Migrate scenario_gen.rs**

This is the most critical file. Replace `CityGrid::new(128, 128, ...)` with `VoxelWorld::ensure_loaded_around()`. Replace all `state.city_grids[gi]` access patterns. Replace grid cell stamping with voxel placement. Remove `settlement.city_grid_idx` usage.

- [ ] **Step 4: Migrate mass_gen.rs**

Replace `CityGrid, CellState, CellTerrain, InfluenceMap` imports. Update `place_buildings_batch()`, `is_buildable()`, `generate_scenario_state()` to use VoxelWorld. Replace `CityGrid::new` call in `generate_scenario_state` with voxel world initialization.

- [ ] **Step 5: Migrate validation modules**

Update `validation/action.rs`, `validation/world_state.rs`, `validation/features.rs`, `validation/probes.rs` — replace all CityGrid references. Update test fixtures that create `CityGrid::new(...)` to instead populate VoxelWorld.

- [ ] **Step 6: Build and test**

Run: `cargo build 2>&1 | tail -10`
Run: `cargo test 2>&1 | tail -10`
Expected: Both pass.

- [ ] **Step 7: Commit**

```bash
git add src/world_sim/building_ai/
git commit -m "refactor: migrate building_ai modules from CityGrid to VoxelWorld"
```

---

### Task 9: Remove CityGrid and clean up WorldState

**Files:**
- Delete: `src/world_sim/city_grid.rs`
- Modify: `src/world_sim/mod.rs` (remove `pub mod city_grid;`)
- Modify: `src/world_sim/state.rs` (remove `city_grids`, `influence_maps` fields, `city_grid_idx` from SettlementState)
- Test: `cargo build && cargo test`

Final cleanup: remove the old CityGrid module entirely and all references to it from WorldState.

- [ ] **Step 1: Remove module declaration**

In `src/world_sim/mod.rs`, delete the line `pub mod city_grid;`.

- [ ] **Step 2: Remove WorldState fields**

In `src/world_sim/state.rs`:
- Remove `pub city_grids: Vec<super::city_grid::CityGrid>,` and its comment
- Remove `pub influence_maps: Vec<super::city_grid::InfluenceMap>,` and its comment
- Remove `city_grids: Vec::new(),` from `WorldState::new()`
- Remove `influence_maps: Vec::new(),` from `WorldState::new()`

- [ ] **Step 3: Remove city_grid_idx from SettlementState**

In `src/world_sim/state.rs`:
- Remove `pub city_grid_idx: Option<usize>,` from `SettlementState`
- Remove `city_grid_idx: None,` from SettlementState initializer

- [ ] **Step 4: Delete the file**

Delete `src/world_sim/city_grid.rs`.

- [ ] **Step 5: Fix any remaining compile errors**

Run: `cargo build 2>&1`

If any files still reference `city_grid`, `CityGrid`, `CellState`, `CellTerrain`, `ZoneType`, `InfluenceMap`, `city_grids`, `influence_maps`, or `city_grid_idx`, fix them. These should have been caught in Tasks 7-8, but there may be stray references.

Common patterns:
- `city_grid::ZoneType` → `voxel::VoxelZone`
- `city_grid::CellState` → check `voxel.material` and `voxel.building_id`
- `state.city_grids` → `state.voxel_world`
- `settlement.city_grid_idx` → remove (use settlement.pos to derive voxel coords)

- [ ] **Step 6: Run full test suite**

Run: `cargo test 2>&1 | tail -10`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove CityGrid — VoxelWorld is now the sole spatial grid system"
```

---

### Task 10: Update apply.rs grid transitions for FidelityZone

**Files:**
- Modify: `src/world_sim/apply.rs` (function names)
- Modify: `src/world_sim/delta.rs` (if `grid_id` fields need renaming)
- Test: `cargo build && cargo test`

The `apply_grid_transitions` and `apply_fidelity` functions in apply.rs reference `grids` (now `fidelity_zones`). The delta variants `EntityEntersGrid`/`EntityLeavesGrid`/`EscalateFidelity` use `grid_id` which still makes sense semantically for fidelity zones.

- [ ] **Step 1: Read apply.rs grid-related functions**

Read the `apply_grid_transitions` and `apply_fidelity` functions to find `state.grids` references.

- [ ] **Step 2: Update references**

Change `state.grids` → `state.fidelity_zones` and `state.grid_mut(id)` → `state.fidelity_zone_mut(id)` in:
- `apply_grid_transitions()`
- `apply_fidelity()`

The delta enum variants (`EntityEntersGrid`, `EntityLeavesGrid`, `EscalateFidelity`) keep their names — they refer to fidelity zone IDs and the rename is cosmetic. Changing them would touch too many systems for no functional benefit.

- [ ] **Step 3: Build and test**

Run: `cargo build && cargo test 2>&1 | tail -10`
Expected: Both pass.

- [ ] **Step 4: Commit**

```bash
git add src/world_sim/apply.rs
git commit -m "refactor: update apply.rs grid references to fidelity_zones"
```
