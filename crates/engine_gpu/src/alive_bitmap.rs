//! Per-tick alive bitmap — packed `array<u32>` with one bit per agent slot.
//!
//! # Motivation
//!
//! Every `agents.alive(x)` predicate in the DSL (masks, scoring rows,
//! physics rule guards) previously lowered to a field read through the
//! 64-byte `AgentSlot` cacheline (physics) or the per-field SoA buffers
//! (mask, scoring). At N=100k with ~30 `alive()` calls per agent per
//! tick that is ≥3M cacheline reads per tick JUST to test a single
//! flag.
//!
//! # Design
//!
//! One `array<u32>` sized to `ceil(agent_cap / 32)` is packed ONCE at
//! the top of each tick by [`AlivePackKernel`]. Every subsequent kernel
//! on the hot path binds the bitmap at slot 22 and reads
//!
//! ```text
//!   (alive_bitmap[x >> 5u] >> (x & 31u)) & 1u != 0u
//! ```
//!
//! instead of a full `agents[x].alive` load. At N=100k the bitmap is
//! 12.5 KB — L1-resident on the 4090 — so every read is a cache hit.
//!
//! # Correctness
//!
//! The pack kernel runs at the TOP of each tick. The bitmap therefore
//! reflects alive-as-of-tick-start. Every kernel that runs inside the
//! tick (mask / scoring / physics cascade) sees a consistent snapshot.
//! This matches the pre-bitmap semantics: the SoA `alive` flag is only
//! mutated by `apply_actions` inside the tick, so mid-tick reads from
//! the SoA field already saw stale values — the bitmap's staleness is
//! identical.
//!
//! If a kernel ever needs mid-tick fresh alive state (no known call
//! site today) it can still read `agents[x].alive` directly via the
//! pre-bitmap lowering path.
//!
//! # Binding slot
//!
//! Slot 22 on the resident physics BGL, the mask BGL, and the scoring
//! BGL. 17 is gold, 18/19 is standing, 20/21 is memory — 22 is the
//! next contiguous slot.
//!
//! # Write semantics
//!
//! One thread per u32 word of the bitmap. Each thread reads the 32
//! `agents[i].alive` fields covered by its word, packs them into the
//! word, and writes the word non-atomically. Each thread owns its
//! output word — no atomics, no races.

#![cfg(feature = "gpu")]

/// Binding slot on every kernel BGL that reads `alive_bitmap`.
/// `agent_cap / 32` u32 words in the bitmap — sized in
/// [`alive_bitmap_bytes`].
pub const ALIVE_BITMAP_BINDING: u32 = 22;

/// Word-count (u32s) in the bitmap for a given `agent_cap`. One bit
/// per slot, packed 32 slots per word. `ceil(agent_cap / 32)`.
#[inline]
pub fn alive_bitmap_words(agent_cap: u32) -> u32 {
    (agent_cap + 31) / 32
}

/// Byte size of the bitmap storage buffer for a given `agent_cap`.
/// Clamped to at least 4 B so the wgpu allocator never sees a
/// zero-sized buffer (agent_cap == 0 is rare but legal in tests).
#[inline]
pub fn alive_bitmap_bytes(agent_cap: u32) -> u64 {
    (alive_bitmap_words(agent_cap) as u64 * 4).max(4)
}

/// Create an empty bitmap storage buffer sized for `agent_cap` slots.
/// Zero-initialised by wgpu — safe as a pre-pack default (bitmap is
/// overwritten at the top of every tick before any reader runs).
pub fn create_alive_bitmap_buffer(device: &wgpu::Device, agent_cap: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine_gpu::alive_bitmap"),
        size: alive_bitmap_bytes(agent_cap),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Errors the pack kernel can surface at init time.
#[derive(Debug)]
pub enum AliveBitmapError {
    /// WGSL compile / shader module creation failed.
    ShaderCompile(String),
}

impl std::fmt::Display for AliveBitmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AliveBitmapError::ShaderCompile(s) => write!(f, "alive_bitmap shader compile: {s}"),
        }
    }
}

impl std::error::Error for AliveBitmapError {}

/// WGSL source for the alive-bitmap pack kernel.
///
/// One thread per output u32 word. Each thread scans 32 agent slots,
/// OR's their `alive` bits into a local accumulator, and writes the
/// word non-atomically. `agents[i].alive` is a `u32` at offset 16 B
/// inside the 64-byte `AgentSlot` layout; WGSL only needs the `alive`
/// field of the struct declared here.
const ALIVE_PACK_WGSL: &str = r#"
struct AgentSlot {
    hp: f32,
    max_hp: f32,
    shield_hp: f32,
    attack_damage: f32,
    alive: u32,
    creature_type: u32,
    engaged_with: u32,
    stun_expires_at: u32,
    slow_expires_at: u32,
    slow_factor_q8: u32,
    cooldown_next_ready: u32,
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    _pad0: u32,
    _pad1: u32,
};

struct PackCfg {
    agent_cap: u32,
    num_words: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       agents: array<AgentSlot>;
@group(0) @binding(1) var<storage, read_write> alive_bitmap: array<u32>;
@group(0) @binding(2) var<uniform>             cfg: PackCfg;

@compute @workgroup_size(64)
fn cs_pack_alive_bitmap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    if (word_idx >= cfg.num_words) { return; }

    let base_slot = word_idx * 32u;
    var word: u32 = 0u;
    // Scan 32 slots into a single u32. Each thread owns its word,
    // non-atomic write at the end.
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        let slot = base_slot + i;
        if (slot >= cfg.agent_cap) { break; }
        if (agents[slot].alive != 0u) {
            word = word | (1u << i);
        }
    }
    alive_bitmap[word_idx] = word;
}
"#;

/// WGSL snippet that every consumer kernel pastes into its shader
/// prefix so it can read the bitmap. Declares the binding + a helper
/// function `alive_bit(slot)` that returns `bool`.
///
/// Takes the binding slot as a parameter so the caller can align it
/// with its BGL numbering (all production consumers bind at
/// [`ALIVE_BITMAP_BINDING`] = 22).
pub fn alive_bitmap_wgsl_decl(binding: u32) -> String {
    format!(
        "@group(0) @binding({binding}) var<storage, read> alive_bitmap: array<u32>;\n\
         fn alive_bit(slot: u32) -> bool {{\n\
         \x20   return (alive_bitmap[slot >> 5u] >> (slot & 31u)) & 1u != 0u;\n\
         }}\n"
    )
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct PackCfg {
    agent_cap: u32,
    num_words: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Per-tick pack kernel. Owns its pipeline + BGL; callers lazy-build
/// one instance via [`AlivePackKernel::new`] and dispatch each tick
/// via [`AlivePackKernel::encode_pack`].
pub struct AlivePackKernel {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cfg_buf: wgpu::Buffer,
    last_cfg: Option<PackCfg>,
    /// Cached bind group keyed by (agents_buf, alive_bitmap_buf)
    /// identity. Stable across a batch so this amortises to one BG
    /// build per batch.
    cached_bg: Option<(wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup)>,
}

impl AlivePackKernel {
    pub fn new(device: &wgpu::Device) -> Result<Self, AliveBitmapError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine_gpu::alive_bitmap::wgsl"),
            source: wgpu::ShaderSource::Wgsl(ALIVE_PACK_WGSL.into()),
        });
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            return Err(AliveBitmapError::ShaderCompile(format!(
                "{err}\n--- WGSL source ---\n{ALIVE_PACK_WGSL}"
            )));
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine_gpu::alive_bitmap::bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine_gpu::alive_bitmap::pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("engine_gpu::alive_bitmap::cp"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_pack_alive_bitmap"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cfg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine_gpu::alive_bitmap::cfg"),
            size: std::mem::size_of::<PackCfg>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bgl,
            cfg_buf,
            last_cfg: None,
            cached_bg: None,
        })
    }

    /// Encode one pack dispatch. Builds (or re-uses a cached) bind
    /// group, (re-)uploads the cfg uniform only when `agent_cap`
    /// changes, and records a `(ceil(num_words / 64), 1, 1)`
    /// dispatch. Non-atomic writes — each thread owns its output
    /// word, no races.
    pub fn encode_pack(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        agents_buf: &wgpu::Buffer,
        alive_bitmap_buf: &wgpu::Buffer,
        agent_cap: u32,
    ) {
        let num_words = alive_bitmap_words(agent_cap);
        let cfg = PackCfg {
            agent_cap,
            num_words,
            _pad0: 0,
            _pad1: 0,
        };
        if self.last_cfg != Some(cfg) {
            queue.write_buffer(&self.cfg_buf, 0, bytemuck::bytes_of(&cfg));
            self.last_cfg = Some(cfg);
        }

        // Invalidate cached BG if either bound buffer identity changed.
        // Buffer equality in wgpu is by handle identity; comparing Buffer
        // values with `==` works via the underlying Arc<BufferInner>.
        let cache_valid = match &self.cached_bg {
            Some((a, b, _)) => a == agents_buf && b == alive_bitmap_buf,
            None => false,
        };
        if !cache_valid {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine_gpu::alive_bitmap::bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: agents_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: alive_bitmap_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.cfg_buf.as_entire_binding(),
                    },
                ],
            });
            self.cached_bg = Some((agents_buf.clone(), alive_bitmap_buf.clone(), bg));
        }
        let bg = &self.cached_bg.as_ref().expect("bg just populated").2;

        if num_words == 0 {
            return;
        }

        let wg_size = 64u32;
        let workgroups = (num_words + wg_size - 1) / wg_size;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine_gpu::alive_bitmap::cpass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_rounds_up() {
        assert_eq!(alive_bitmap_words(0), 0);
        assert_eq!(alive_bitmap_words(1), 1);
        assert_eq!(alive_bitmap_words(31), 1);
        assert_eq!(alive_bitmap_words(32), 1);
        assert_eq!(alive_bitmap_words(33), 2);
        assert_eq!(alive_bitmap_words(100_000), 3125);
    }

    #[test]
    fn bytes_clamps_to_min() {
        assert_eq!(alive_bitmap_bytes(0), 4);
        assert_eq!(alive_bitmap_bytes(32), 4);
        assert_eq!(alive_bitmap_bytes(33), 8);
        assert_eq!(alive_bitmap_bytes(100_000), 3125 * 4);
    }

    #[test]
    fn wgsl_decl_at_binding_22() {
        let src = alive_bitmap_wgsl_decl(22);
        assert!(src.contains("@binding(22)"));
        assert!(src.contains("alive_bitmap: array<u32>"));
        assert!(src.contains("fn alive_bit("));
    }
}
