//! `mesh_renderer` — host-side glTF mesh loading + (eventually) Vulkan
//! mesh-pass rendering for agent character models.
//!
//! # Status: Phase 1 — loading only
//!
//! This module currently loads a `.glb` from disk and parses its first
//! mesh into a CPU-side [`MeshCpu`] (positions + indices + optional
//! per-vertex colors). The Vulkan mesh-rendering pass is the next step;
//! for now this lets us validate the asset pipeline + leave a clean
//! seam to wire skinning + animation later (instance-buffer layout
//! already includes a placeholder field for a future bone-matrix
//! offset).

use anyhow::{anyhow, Context, Result};
use glam::{Vec3, Vec4};
use std::path::Path;

/// CPU-side static mesh loaded from a glTF file. Vertices are
/// position + optional color; indices are u32. Triangle-list topology.
///
/// Future: skinning weights/joints + per-mesh bone-binding metadata.
/// The current shape is the minimum-viable for "render N instances
/// of one mesh"; we add skinning fields when actual animations land.
pub struct MeshCpu {
    pub positions: Vec<Vec3>,
    /// Optional per-vertex color (sRGB linear). Defaults to white when
    /// the glTF doesn't ship colors — the renderer's per-instance tint
    /// is the primary recoloring channel anyway.
    pub colors: Vec<Vec4>,
    pub indices: Vec<u32>,
    /// Source file path for debug / error messages.
    pub source: String,
}

impl MeshCpu {
    /// Load the first mesh of the first primitive in `path` (`.glb`
    /// or `.gltf`). Multi-mesh / multi-primitive files use only the
    /// first; Kenney's mini-characters pack ships one mesh per file.
    pub fn load_glb(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path_str = path.display().to_string();
        let (doc, buffers, _images) = gltf::import(path)
            .with_context(|| format!("gltf::import failed for {path_str}"))?;

        let mesh = doc
            .meshes()
            .next()
            .ok_or_else(|| anyhow!("{path_str}: glTF has no meshes"))?;
        let prim = mesh
            .primitives()
            .next()
            .ok_or_else(|| anyhow!("{path_str}: mesh has no primitives"))?;

        let reader = prim.reader(|b| Some(&buffers[b.index()]));
        let positions: Vec<Vec3> = reader
            .read_positions()
            .ok_or_else(|| anyhow!("{path_str}: primitive has no POSITION attribute"))?
            .map(|[x, y, z]| Vec3::new(x, y, z))
            .collect();
        // glTF accessor flavors: u8 / u16 / u32 / no-index. `read_indices()`
        // returns an enum; `into_u32()` normalizes.
        let indices: Vec<u32> = reader
            .read_indices()
            .map(|i| i.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let colors: Vec<Vec4> = match reader.read_colors(0) {
            Some(c) => c.into_rgba_f32()
                .map(|[r, g, b, a]| Vec4::new(r, g, b, a))
                .collect(),
            None => vec![Vec4::ONE; positions.len()],
        };

        Ok(Self { positions, colors, indices, source: path_str })
    }

    pub fn vertex_count(&self) -> usize { self.positions.len() }
    pub fn triangle_count(&self) -> usize { self.indices.len() / 3 }
}

/// Per-instance shading + transform record. Lives in a host-uploaded
/// storage buffer the vertex shader reads via the instance index.
///
/// **Animation-readiness:** `bone_matrix_offset` is a forward-compatible
/// placeholder. Today it's always 0; once skinning lands, each instance
/// allocates a contiguous run of bone matrices in a separate buffer and
/// stores its base offset here. The vertex shader picks the right
/// matrices by reading `bones[bone_matrix_offset + joint_idx]`. Adding
/// this field now means we don't rewrite the shader uniform layout
/// later — the field is just unused for static meshes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
pub struct InstanceData {
    /// World transform (column-major mat4) baked from agent position.
    pub model: [[f32; 4]; 4],
    /// RGB tint applied to the mesh's base color. A = unused (padding).
    pub tint: [f32; 4],
    /// Offset into the global bone-matrix array. **Unused today**
    /// (no skinning); reserved for the skinning pass.
    pub bone_matrix_offset: u32,
    /// Number of joints this instance reads. **Unused today**.
    pub bone_matrix_count: u32,
    pub _pad: [u32; 2],
}

impl InstanceData {
    pub fn from_position(pos: Vec3, scale: f32, tint: [f32; 3]) -> Self {
        let s = scale;
        // Column-major mat4 = translation + uniform scale.
        let model = [
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, s, 0.0],
            [pos.x, pos.y, pos.z, 1.0],
        ];
        Self {
            model,
            tint: [tint[0], tint[1], tint[2], 1.0],
            bone_matrix_offset: 0,
            bone_matrix_count: 0,
            _pad: [0, 0],
        }
    }
}
