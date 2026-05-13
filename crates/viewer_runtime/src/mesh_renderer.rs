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

// ---------------------------------------------------------------------
// Vulkan mesh-rendering pass.
// ---------------------------------------------------------------------
//
// Plugs into voxel_engine's `present_blit_with_overlay` callback. Voxel
// pass writes to `light_output_image`, swapchain blit copies it to the
// swapchain image (already in COLOR_ATTACHMENT_OPTIMAL when our overlay
// fires), our overlay records a single render pass that loads existing
// contents and draws meshes on top, then transitions the image to
// PRESENT_SRC_KHR for present.
//
// Allocates buffers directly via ash::vk (raw) because voxel_engine's
// `VulkanAllocator` hardcodes STORAGE_BUFFER usage which can't be used
// for vertex/index bindings.

use ash::vk;
use voxel_engine::vulkan::graphics_pipeline::{GraphicsPipeline, GraphicsPipelineBuilder};
use voxel_engine::vulkan::instance::VulkanContext;

const VERT_SPV: &[u8] = include_bytes!("../shaders/mesh.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../shaders/mesh.frag.spv");

pub struct MeshRendererGpu {
    extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: GraphicsPipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    vertex_buf: vk::Buffer,
    vertex_mem: vk::DeviceMemory,
    index_buf: vk::Buffer,
    index_mem: vk::DeviceMemory,
    index_count: u32,
    instance_buf: vk::Buffer,
    instance_mem: vk::DeviceMemory,
    instance_capacity: u32,
    instance_mapped: *mut u8,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
struct PushConstants {
    view_proj: [[f32; 4]; 4],
}

impl MeshRendererGpu {
    pub fn new(
        ctx: &VulkanContext,
        swapchain_views: &[vk::ImageView],
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
        mesh: &MeshCpu,
        max_instances: u32,
    ) -> Result<Self> {
        let device = ctx.device();
        let render_pass = create_overlay_render_pass(device, swapchain_format)?;
        let framebuffers = create_framebuffers(device, render_pass, swapchain_views, swapchain_extent)?;

        let pipeline = GraphicsPipelineBuilder::new(ctx)
            .vertex_shader(VERT_SPV)
            .fragment_shader(FRAG_SPV)
            .render_pass(render_pass)
            .push_constant_size(
                std::mem::size_of::<PushConstants>() as u32,
                vk::ShaderStageFlags::VERTEX,
            )
            .descriptor(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )
            .depth_write(false)
            .cull_mode(vk::CullModeFlags::NONE)
            .no_depth_test()
            .build()?;

        // Vertex buffer (positions only, host-visible). Kenney's
        // mini-characters are <1k vertices each; host-visible is fine
        // for one-time upload at init.
        let positions_bytes: Vec<u8> = mesh
            .positions
            .iter()
            .flat_map(|p| {
                let mut out = [0u8; 12];
                out[..4].copy_from_slice(&p.x.to_le_bytes());
                out[4..8].copy_from_slice(&p.y.to_le_bytes());
                out[8..12].copy_from_slice(&p.z.to_le_bytes());
                out
            })
            .collect();
        let (vertex_buf, vertex_mem) = alloc_host_visible_buffer(
            ctx,
            positions_bytes.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        write_buffer(ctx, vertex_mem, 0, &positions_bytes)?;

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

        let instance_bytes = (max_instances as u64) * std::mem::size_of::<InstanceData>() as u64;
        let (instance_buf, instance_mem) = alloc_host_visible_buffer(
            ctx,
            instance_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let instance_mapped = unsafe {
            device.map_memory(instance_mem, 0, instance_bytes, vk::MemoryMapFlags::empty())?
        } as *mut u8;

        // Descriptor pool + set.
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
        let set_layout = pipeline
            .descriptor_set_layout
            .expect("mesh pipeline has descriptor set");
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
            vertex_buf,
            vertex_mem,
            index_buf,
            index_mem,
            index_count: mesh.indices.len() as u32,
            instance_buf,
            instance_mem,
            instance_capacity: max_instances,
            instance_mapped,
        })
    }

    /// Record the mesh draw into `cmd` (already in a recording state,
    /// caller's responsibility per voxel_engine's
    /// `present_blit_with_overlay` contract). Image is in
    /// COLOR_ATTACHMENT_OPTIMAL on entry; render pass transitions to
    /// PRESENT_SRC_KHR on exit.
    pub fn record_overlay(
        &mut self,
        ctx: &VulkanContext,
        cmd: vk::CommandBuffer,
        image_index: usize,
        view_proj: glam::Mat4,
        instances: &[InstanceData],
    ) -> Result<()> {
        let device = ctx.device();
        let n = (instances.len() as u32).min(self.instance_capacity);
        if n > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    instances.as_ptr() as *const u8,
                    self.instance_mapped,
                    (n as usize) * std::mem::size_of::<InstanceData>(),
                );
            }
        }

        let clear_values = []; // load-op = Load; no clears
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
            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buf], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buf, 0, vk::IndexType::UINT32);
            if n > 0 {
                device.cmd_draw_indexed(cmd, self.index_count, n, 0, 0, 0);
            }
            device.cmd_end_render_pass(cmd);
        }
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &VulkanContext) {
        let device = ctx.device();
        unsafe {
            device.device_wait_idle().ok();
            device.unmap_memory(self.instance_mem);
            device.destroy_buffer(self.instance_buf, None);
            device.free_memory(self.instance_mem, None);
            device.destroy_buffer(self.index_buf, None);
            device.free_memory(self.index_mem, None);
            device.destroy_buffer(self.vertex_buf, None);
            device.free_memory(self.vertex_mem, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for fb in &self.framebuffers {
                device.destroy_framebuffer(*fb, None);
            }
            self.framebuffers.clear();
            device.destroy_render_pass(self.render_pass, None);
        }
        self.pipeline.destroy(ctx);
    }
}

fn create_overlay_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_att = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        // LOAD_OP_LOAD preserves voxel_engine's output beneath us.
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // Caller (present_blit_with_overlay) guarantees this layout on entry.
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        // Render pass transitions to PRESENT_SRC_KHR — caller's contract.
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let color_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    let atts = [color_att];
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
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    swapchain_views
        .iter()
        .map(|view| {
            let attachments = [*view];
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
