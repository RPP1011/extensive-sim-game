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
/// position + normals + color + optional skinning data; indices are
/// u32. Triangle-list topology.
///
/// Skinning is loaded but not yet consumed by the renderer — the
/// vertex shader doesn't bind joint/weight attributes today. Having
/// the data on disk + in MeshCpu lets us wire a skinning pipeline
/// later (Phase 3) without re-touching the asset loader. The 4-joint
/// limit matches the glTF spec's JOINTS_0/WEIGHTS_0 layout.
pub struct MeshCpu {
    pub positions: Vec<Vec3>,
    /// Per-vertex normals from the glTF NORMAL_0 attribute. Defaults to
    /// (0, 1, 0) (up) when missing so lighting falls back to a flat-top
    /// look instead of dividing by zero. Used by the fragment shader
    /// for diffuse lighting.
    pub normals: Vec<Vec3>,
    /// Optional per-vertex color (sRGB linear). Defaults to white when
    /// the glTF doesn't ship colors — the renderer's per-instance tint
    /// is the primary recoloring channel anyway.
    pub colors: Vec<Vec4>,
    /// Per-vertex joint indices (JOINTS_0). Each entry binds up to 4
    /// bones; weights[i] holds the corresponding influence. Empty Vec
    /// when the glTF has no skin — most non-character meshes skip this.
    pub joints: Vec<[u16; 4]>,
    /// Per-vertex skinning weights (WEIGHTS_0). Sums to ~1.0 per vertex
    /// when normalised. Empty when no skin.
    pub weights: Vec<[f32; 4]>,
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
        let normals: Vec<Vec3> = match reader.read_normals() {
            Some(it) => it.map(|[x, y, z]| Vec3::new(x, y, z)).collect(),
            None => vec![Vec3::Y; positions.len()],
        };
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

        // Skinning: JOINTS_0 + WEIGHTS_0 if the mesh has a skin. Kenney
        // mini-characters do; static props won't. Loaded but not yet
        // consumed by the shader — the vertex input is still
        // position + normal only. Wire later when adding animation.
        let joints: Vec<[u16; 4]> = match reader.read_joints(0) {
            Some(j) => j.into_u16().collect(),
            None => Vec::new(),
        };
        let weights: Vec<[f32; 4]> = match reader.read_weights(0) {
            Some(w) => w.into_f32().collect(),
            None => Vec::new(),
        };

        Ok(Self { positions, normals, colors, joints, weights, indices, source: path_str })
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
        Self::from_pos_facing(pos, scale, [0.0, 0.0, 1.0], tint)
    }

    /// Position + uniform scale + Y-axis-only orientation. `facing` is a
    /// 3D direction in renderer space (Y-up); only the XZ components
    /// matter — Y is silently dropped so models stay upright.
    ///
    /// Convention: the model's local +Z axis is taken as "forward", so
    /// after rotation, local +Z lines up with `facing` in world space.
    /// Local +Y stays world +Y (up).
    pub fn from_pos_facing(pos: Vec3, scale: f32, facing: [f32; 3], tint: [f32; 3]) -> Self {
        let fx = facing[0];
        let fz = facing[2];
        let len = (fx * fx + fz * fz).sqrt();
        let (fx, fz) = if len > 1e-4 { (fx / len, fz / len) } else { (0.0, 1.0) };
        let sc = scale;
        // Column-major: col0 = local +X → (fz, 0, -fx) * sc, col1 = +Y → up,
        // col2 = local +Z → (fx, 0, fz) * sc, col3 = translation.
        let model = [
            [fz * sc, 0.0, -fx * sc, 0.0],
            [0.0, sc, 0.0, 0.0],
            [fx * sc, 0.0, fz * sc, 0.0],
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

// ---------------------------------------------------------------------
// Vulkan mesh-rendering pass.
// ---------------------------------------------------------------------
//
// Plugs into voxel_engine's `present_blit_with_overlay` callback. Voxel
// pass writes to `light_output_image`, swapchain blit copies it to the
// swapchain image (already in COLOR_ATTACHMENT_OPTIMAL when our overlay
// fires), our overlay records a single render pass that loads existing
// contents and draws ALL agent meshes on top, then transitions the
// image to PRESENT_SRC_KHR for present.
//
// Supports multiple meshes per render pass (one per creature type +
// per hero role). Caller passes draws sorted by mesh slot; the instance
// buffer is filled contiguously per slot, with one
// `cmd_draw_indexed(first_instance=slot_offset)` per mesh.
//
// Allocates buffers directly via ash::vk (raw) because voxel_engine's
// `VulkanAllocator` hardcodes STORAGE_BUFFER usage which can't be used
// for vertex/index bindings.

use ash::vk;

use voxel_engine::vulkan::instance::VulkanContext;

const VERT_SPV: &[u8] = include_bytes!("../shaders/mesh.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../shaders/mesh.frag.spv");

/// One uploaded mesh — vertex + index buffers, no per-mesh state beyond
/// that. Multiple `MeshSlot`s live inside one [`MeshRendererGpu`] and
/// share the pipeline + render pass + instance buffer.
struct MeshSlot {
    vertex_buf: vk::Buffer,
    vertex_mem: vk::DeviceMemory,
    index_buf: vk::Buffer,
    index_mem: vk::DeviceMemory,
    index_count: u32,
    /// Source file path for debug — read by `add_mesh`'s log line.
    #[allow(dead_code)]
    source: String,
}

/// Local pipeline holder — voxel_engine's `GraphicsPipeline` keeps the
/// shader modules private so we can't construct one with our own
/// custom vertex input (2 attributes: position + normal). Same fields
/// + manual destroy.
struct MeshPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    vert_module: vk::ShaderModule,
    frag_module: vk::ShaderModule,
}

impl MeshPipeline {
    fn destroy(&self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.vert_module, None);
            device.destroy_shader_module(self.frag_module, None);
        }
    }
}

pub struct MeshRendererGpu {
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: MeshPipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    meshes: Vec<MeshSlot>,
    instance_buf: vk::Buffer,
    instance_mem: vk::DeviceMemory,
    instance_capacity: u32,
    instance_mapped: *mut u8,
    /// Depth attachment for mesh-to-mesh occlusion. Cleared per frame
    /// (LOAD_OP_CLEAR) — voxel pass doesn't share depth, so values in
    /// here only constrain mesh-vs-mesh ordering.
    depth_image: vk::Image,
    depth_view: vk::ImageView,
    depth_mem: vk::DeviceMemory,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct PushConstants {
    view_proj: [[f32; 4]; 4],
}

/// One per-frame draw — `mesh_slot` is the index returned by
/// [`MeshRendererGpu::add_mesh`]; `instances` holds the agents that
/// should render with that mesh. Caller is responsible for bucketing.
pub struct MeshDraw<'a> {
    pub mesh_slot: usize,
    pub instances: &'a [InstanceData],
}

impl MeshRendererGpu {
    pub fn new(
        ctx: &VulkanContext,
        swapchain_views: &[vk::ImageView],
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
        max_instances: u32,
    ) -> Result<Self> {
        let device = ctx.device();
        const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
        let (depth_image, depth_mem, depth_view) =
            alloc_depth_image(ctx, swapchain_extent, DEPTH_FORMAT)?;
        let render_pass =
            create_overlay_render_pass(device, swapchain_format, DEPTH_FORMAT)?;
        let framebuffers = create_framebuffers(
            device, render_pass, swapchain_views, depth_view, swapchain_extent,
        )?;

        let pipeline = build_mesh_pipeline(ctx, render_pass)?;

        let instance_bytes = (max_instances as u64) * std::mem::size_of::<InstanceData>() as u64;
        let (instance_buf, instance_mem) = alloc_host_visible_buffer(
            ctx,
            instance_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let instance_mapped = unsafe {
            device.map_memory(instance_mem, 0, instance_bytes, vk::MemoryMapFlags::empty())?
        } as *mut u8;

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        };
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .context("create_descriptor_pool")?;
        let set_layout = pipeline.descriptor_set_layout;
        let layouts = [set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("allocate_descriptor_sets")?[0];

        let buf_info = vk::DescriptorBufferInfo {
            buffer: instance_buf,
            offset: 0,
            range: instance_bytes,
        };
        let buf_infos = [buf_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_infos);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        Ok(Self {
            extent: swapchain_extent,
            render_pass,
            framebuffers,
            pipeline,
            descriptor_pool,
            descriptor_set,
            meshes: Vec::new(),
            instance_buf,
            instance_mem,
            instance_capacity: max_instances,
            instance_mapped,
            depth_image,
            depth_view,
            depth_mem,
        })
    }

    /// Upload a mesh and return its slot index. Heroes vs creature
    /// types each register one slot; per-frame draws specify slot
    /// indices in their [`MeshDraw`] entries.
    pub fn add_mesh(&mut self, ctx: &VulkanContext, mesh: &MeshCpu) -> Result<usize> {
        // Interleaved position + normal: 24 bytes per vertex. Matches
        // the pipeline's vertex input layout (binding 0 stride 24,
        // attribute 0 = pos at offset 0, attribute 1 = normal at
        // offset 12). Both vec3 of f32.
        let n = mesh.positions.len();
        let mut interleaved: Vec<u8> = Vec::with_capacity(n * 24);
        for i in 0..n {
            let p = mesh.positions[i];
            let nrm = if i < mesh.normals.len() {
                mesh.normals[i]
            } else {
                glam::Vec3::Y
            };
            interleaved.extend_from_slice(&p.x.to_le_bytes());
            interleaved.extend_from_slice(&p.y.to_le_bytes());
            interleaved.extend_from_slice(&p.z.to_le_bytes());
            interleaved.extend_from_slice(&nrm.x.to_le_bytes());
            interleaved.extend_from_slice(&nrm.y.to_le_bytes());
            interleaved.extend_from_slice(&nrm.z.to_le_bytes());
        }
        let (vertex_buf, vertex_mem) = alloc_host_visible_buffer(
            ctx,
            interleaved.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        write_buffer(ctx, vertex_mem, 0, &interleaved)?;

        let index_bytes: Vec<u8> = mesh
            .indices
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let (index_buf, index_mem) = alloc_host_visible_buffer(
            ctx,
            index_bytes.len() as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        write_buffer(ctx, index_mem, 0, &index_bytes)?;

        let slot = self.meshes.len();
        self.meshes.push(MeshSlot {
            vertex_buf,
            vertex_mem,
            index_buf,
            index_mem,
            index_count: mesh.indices.len() as u32,
            source: mesh.source.clone(),
        });
        Ok(slot)
    }

    /// Record the overlay pass: load voxel contents, draw each
    /// `MeshDraw`, transition to PRESENT_SRC_KHR. Instances are
    /// flattened into the persistent-mapped instance buffer in slot
    /// order, with first_instance per draw advancing as we go.
    pub fn record_overlay(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        image_index: usize,
        view_proj: glam::Mat4,
        draws: &[MeshDraw<'_>],
    ) -> Result<()> {
        let device = ctx.device();

        // Pack instances into the mapped buffer; remember each draw's
        // first_instance offset.
        let stride = std::mem::size_of::<InstanceData>();
        let mut first_instances: Vec<u32> = Vec::with_capacity(draws.len());
        let mut counts: Vec<u32> = Vec::with_capacity(draws.len());
        let mut cursor: u32 = 0;
        for d in draws {
            let n = (d.instances.len() as u32).min(self.instance_capacity - cursor);
            first_instances.push(cursor);
            counts.push(n);
            if n > 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        d.instances.as_ptr() as *const u8,
                        self.instance_mapped.add((cursor as usize) * stride),
                        (n as usize) * stride,
                    );
                }
            }
            cursor += n;
        }

        // Color attachment is LOAD_OP_LOAD (no clear); depth is
        // LOAD_OP_CLEAR with depth=1.0 (far plane). Vulkan still expects
        // a clear value at each attachment's index when LOAD is LOAD,
        // but ignores it; we still have to pass two entries.
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            let pc = PushConstants { view_proj: view_proj.to_cols_array_2d() };
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&pc),
            );
            for (di, d) in draws.iter().enumerate() {
                let n = counts[di];
                if n == 0 { continue; }
                let slot = match self.meshes.get(d.mesh_slot) {
                    Some(s) => s,
                    None => continue,
                };
                device.cmd_bind_vertex_buffers(cmd, 0, &[slot.vertex_buf], &[0]);
                device.cmd_bind_index_buffer(cmd, slot.index_buf, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed(cmd, slot.index_count, n, 0, 0, first_instances[di]);
            }
            device.cmd_end_render_pass(cmd);
        }
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.device_wait_idle().ok();
            for m in self.meshes.drain(..) {
                device.destroy_buffer(m.index_buf, None);
                device.free_memory(m.index_mem, None);
                device.destroy_buffer(m.vertex_buf, None);
                device.free_memory(m.vertex_mem, None);
            }
            device.unmap_memory(self.instance_mem);
            device.destroy_buffer(self.instance_buf, None);
            device.free_memory(self.instance_mem, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for fb in &self.framebuffers {
                device.destroy_framebuffer(*fb, None);
            }
            self.framebuffers.clear();
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_mem, None);
            device.destroy_render_pass(self.render_pass, None);
        }
        self.pipeline.destroy(ctx);
    }
}

fn create_overlay_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_att = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    // Depth attachment: cleared on entry, discarded on exit. Image lives
    // for the lifetime of MeshRendererGpu; first frame's UNDEFINED →
    // CLEAR transition happens implicitly via the LOAD_OP_CLEAR.
    let depth_att = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let depth_ref = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);
    let atts = [color_att, depth_att];
    let subpasses = [subpass];
    let rp_ci = vk::RenderPassCreateInfo::default()
        .attachments(&atts)
        .subpasses(&subpasses);
    unsafe { device.create_render_pass(&rp_ci, None) }.context("create_render_pass")
}

fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_views: &[vk::ImageView],
    depth_view: vk::ImageView,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    swapchain_views
        .iter()
        .map(|view| {
            let attachments = [*view, depth_view];
            let fb_ci = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);
            unsafe { device.create_framebuffer(&fb_ci, None) }.context("create_framebuffer")
        })
        .collect()
}

fn build_mesh_pipeline(
    ctx: &VulkanContext,
    render_pass: vk::RenderPass,
) -> Result<MeshPipeline> {
    let device = ctx.device();

    // Shader modules.
    let make_module = |spv: &[u8]| -> Result<vk::ShaderModule> {
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let ci = vk::ShaderModuleCreateInfo::default().code(&words);
        unsafe { device.create_shader_module(&ci, None) }
            .context("create_shader_module")
    };
    let vert_module = make_module(VERT_SPV)?;
    let frag_module = make_module(FRAG_SPV)?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // 1 binding, 2 attributes (position + normal interleaved at stride 24).
    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: 24,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 12,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);
    let blend_atts = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blending =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_atts);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let dsl_bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)];
    let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings);
    let descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }
            .context("create_descriptor_set_layout")?;

    let push_range = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: std::mem::size_of::<PushConstants>() as u32,
    }];
    let set_layouts = [descriptor_set_layout];
    let layout_ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_range);
    let layout = unsafe { device.create_pipeline_layout(&layout_ci, None) }
        .context("create_pipeline_layout")?;

    let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blending)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    // voxel_engine's VulkanContext has no pipeline-cache accessor — every
    // pipeline-creation call site inside voxel_engine itself (graphics and
    // compute alike) passes vk::PipelineCache::null() directly; match that
    // convention here instead of relying on a method that doesn't exist.
    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
    }
    .map_err(|(_, e)| e)
    .context("create_graphics_pipelines (mesh)")?[0];

    Ok(MeshPipeline {
        pipeline,
        layout,
        descriptor_set_layout,
        vert_module,
        frag_module,
    })
}

fn alloc_depth_image(
    ctx: &VulkanContext,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let device = ctx.device();
    let img_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&img_ci, None) }.context("create_image (depth)")?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let mem_props =
        unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mem_idx = find_memory_type(
        &mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_idx);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .context("allocate_memory (depth)")?;
    unsafe { device.bind_image_memory(image, memory, 0) }.context("bind_image_memory (depth)")?;
    let view_ci = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let view = unsafe { device.create_image_view(&view_ci, None) }
        .context("create_image_view (depth)")?;
    Ok((image, memory, view))
}

fn alloc_host_visible_buffer(
    ctx: &VulkanContext,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let device = ctx.device();
    let buf_ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buf_ci, None) }.context("create_buffer")?;
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_props = unsafe { ctx.instance().get_physical_device_memory_properties(ctx.physical_device()) };
    let mem_idx = find_memory_type(
        &mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_idx);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.context("allocate_memory")?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }.context("bind_buffer_memory")?;
    Ok((buffer, memory))
}

fn write_buffer(
    ctx: &VulkanContext,
    memory: vk::DeviceMemory,
    offset: u64,
    data: &[u8],
) -> Result<()> {
    let device = ctx.device();
    let ptr = unsafe { device.map_memory(memory, offset, data.len() as u64, vk::MemoryMapFlags::empty()) }
        .context("map_memory")? as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        device.unmap_memory(memory);
    }
    Ok(())
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(flags)
        {
            return Ok(i);
        }
    }
    anyhow::bail!("no memory type with flags {flags:?} matching type_bits {type_bits:#x}")
}
