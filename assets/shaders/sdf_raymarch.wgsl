// SDF Raymarching Compute Shader
//
// Raymarches through a 3D SDF volume to render smooth voxel terrain.
// Each pixel casts a ray from the camera, steps through the SDF using
// sphere tracing, and shades the hit point using gradient normals.

struct CameraUniform {
    view_proj_inv: mat4x4<f32>,  // inverse view-projection matrix
    camera_pos: vec4<f32>,        // world-space camera position
    resolution: vec2<f32>,        // output texture resolution
    time: f32,                    // for animation
    _padding: f32,
};

struct ChunkInfo {
    world_offset: vec4<f32>,      // world-space position of chunk (0,0,0) corner
    chunk_count: vec4<i32>,       // number of chunks in x,y,z (w=total)
};

@group(0) @binding(0)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> camera: CameraUniform;

@group(0) @binding(2)
var<uniform> chunk_info: ChunkInfo;

// SDF data as a 3D texture (multiple chunks packed into one volume).
@group(0) @binding(3)
var sdf_volume: texture_3d<f32>;

@group(0) @binding(4)
var sdf_sampler: sampler;

// Material IDs as a 3D texture (same layout as SDF).
@group(0) @binding(5)
var material_volume: texture_3d<u32>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_STEPS: i32 = 128;
const MAX_DIST: f32 = 256.0;
const SURFACE_THRESHOLD: f32 = 0.1;
const CHUNK_SIZE: f32 = 16.0;
const NORMAL_EPSILON: f32 = 0.5;

// Material colors (indexed by VoxelMaterial enum)
const MAT_AIR: u32 = 0u;
const MAT_DIRT: u32 = 1u;
const MAT_STONE: u32 = 2u;
const MAT_GRANITE: u32 = 3u;
const MAT_SAND: u32 = 4u;
const MAT_CLAY: u32 = 5u;
const MAT_GRAVEL: u32 = 6u;
const MAT_GRASS: u32 = 7u;
const MAT_WATER: u32 = 8u;
const MAT_IRON_ORE: u32 = 12u;
const MAT_COPPER_ORE: u32 = 13u;
const MAT_GOLD_ORE: u32 = 14u;
const MAT_COAL: u32 = 15u;
const MAT_CRYSTAL: u32 = 16u;
const MAT_WOOD_LOG: u32 = 17u;
const MAT_WOOD_PLANKS: u32 = 18u;
const MAT_STONE_BLOCK: u32 = 19u;
const MAT_FARMLAND: u32 = 24u;
const MAT_CROP: u32 = 25u;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn material_color(mat_id: u32) -> vec3<f32> {
    switch mat_id {
        case 1u: { return vec3<f32>(0.45, 0.32, 0.18); }  // Dirt
        case 2u: { return vec3<f32>(0.5, 0.5, 0.5); }     // Stone
        case 3u: { return vec3<f32>(0.35, 0.33, 0.32); }   // Granite
        case 4u: { return vec3<f32>(0.85, 0.78, 0.55); }   // Sand
        case 5u: { return vec3<f32>(0.6, 0.45, 0.3); }     // Clay
        case 6u: { return vec3<f32>(0.55, 0.52, 0.48); }   // Gravel
        case 7u: { return vec3<f32>(0.25, 0.55, 0.18); }   // Grass
        case 8u: { return vec3<f32>(0.15, 0.35, 0.65); }   // Water
        case 9u: { return vec3<f32>(0.9, 0.3, 0.05); }     // Lava
        case 10u: { return vec3<f32>(0.7, 0.85, 0.95); }   // Ice
        case 11u: { return vec3<f32>(0.9, 0.92, 0.95); }   // Snow
        case 12u: { return vec3<f32>(0.55, 0.4, 0.3); }    // IronOre
        case 13u: { return vec3<f32>(0.6, 0.45, 0.25); }   // CopperOre
        case 14u: { return vec3<f32>(0.8, 0.7, 0.2); }     // GoldOre
        case 15u: { return vec3<f32>(0.15, 0.15, 0.15); }  // Coal
        case 16u: { return vec3<f32>(0.6, 0.3, 0.8); }     // Crystal
        case 17u: { return vec3<f32>(0.5, 0.35, 0.2); }    // WoodLog
        case 18u: { return vec3<f32>(0.65, 0.5, 0.3); }    // WoodPlanks
        case 19u: { return vec3<f32>(0.6, 0.58, 0.55); }   // StoneBlock
        case 24u: { return vec3<f32>(0.35, 0.25, 0.12); }  // Farmland
        case 25u: { return vec3<f32>(0.2, 0.6, 0.1); }     // Crop
        default: { return vec3<f32>(0.8, 0.2, 0.8); }      // Unknown = magenta
    }
}

/// Sample SDF distance at a world position using trilinear interpolation.
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    // Convert world position to SDF texture coordinates.
    let local = world_pos - chunk_info.world_offset.xyz;
    let tex_size = vec3<f32>(textureDimensions(sdf_volume));
    let uv = local / tex_size;

    // Out-of-bounds check.
    if any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0)) {
        return MAX_DIST;
    }

    return textureSampleLevel(sdf_volume, sdf_sampler, uv, 0.0).r;
}

/// Sample material ID at a world position (nearest-neighbor).
fn sample_material(world_pos: vec3<f32>) -> u32 {
    let local = world_pos - chunk_info.world_offset.xyz;
    let voxel_pos = vec3<i32>(floor(local));
    let tex_size = vec3<i32>(textureDimensions(material_volume));

    if any(voxel_pos < vec3<i32>(0)) || any(voxel_pos >= tex_size) {
        return 0u; // Air
    }

    return textureLoad(material_volume, voxel_pos, 0).r;
}

/// Compute SDF gradient normal via central differences.
fn sdf_normal(pos: vec3<f32>) -> vec3<f32> {
    let e = NORMAL_EPSILON;
    let n = vec3<f32>(
        sample_sdf(pos + vec3<f32>(e, 0.0, 0.0)) - sample_sdf(pos - vec3<f32>(e, 0.0, 0.0)),
        sample_sdf(pos + vec3<f32>(0.0, e, 0.0)) - sample_sdf(pos - vec3<f32>(0.0, e, 0.0)),
        sample_sdf(pos + vec3<f32>(0.0, 0.0, e)) - sample_sdf(pos - vec3<f32>(0.0, 0.0, e)),
    );
    return normalize(n);
}

/// Generate a ray from screen pixel coordinates.
fn get_ray(pixel: vec2<f32>) -> vec3<f32> {
    // NDC: pixel → [-1, 1]
    let ndc = vec2<f32>(
        (pixel.x / camera.resolution.x) * 2.0 - 1.0,
        1.0 - (pixel.y / camera.resolution.y) * 2.0, // flip Y
    );

    // Unproject near and far plane points.
    let near = camera.view_proj_inv * vec4<f32>(ndc, -1.0, 1.0);
    let far = camera.view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let near_world = near.xyz / near.w;
    let far_world = far.xyz / far.w;

    return normalize(far_world - near_world);
}

// ---------------------------------------------------------------------------
// Main compute shader
// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<f32>(global_id.xy);
    let dims = camera.resolution;

    // Skip out-of-bounds pixels.
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let ray_origin = camera.camera_pos.xyz;
    let ray_dir = get_ray(pixel);

    // Sphere tracing through the SDF.
    var t: f32 = 0.0;
    var hit = false;
    var hit_pos = vec3<f32>(0.0);

    for (var i: i32 = 0; i < MAX_STEPS; i = i + 1) {
        let pos = ray_origin + ray_dir * t;
        let d = sample_sdf(pos);

        if d < SURFACE_THRESHOLD {
            hit = true;
            hit_pos = pos;
            break;
        }

        // Advance by distance (sphere tracing).
        // Clamp minimum step to avoid getting stuck.
        t = t + max(d, 0.1);

        if t > MAX_DIST {
            break;
        }
    }

    var color: vec4<f32>;

    if hit {
        // Surface shading.
        let normal = sdf_normal(hit_pos);
        let mat_id = sample_material(hit_pos);
        let base_color = material_color(mat_id);

        // Simple directional light (sun from above-right).
        let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.4));
        let diffuse = max(dot(normal, light_dir), 0.0);
        let ambient = 0.2;

        // Fog based on distance.
        let fog_start = 64.0;
        let fog_end = 200.0;
        let fog_factor = clamp((t - fog_start) / (fog_end - fog_start), 0.0, 1.0);
        let sky_color = vec3<f32>(0.55, 0.7, 0.9);

        let lit_color = base_color * (ambient + diffuse * 0.8);
        let final_color = mix(lit_color, sky_color, fog_factor);

        color = vec4<f32>(final_color, 1.0);
    } else {
        // Sky gradient.
        let sky_t = ray_dir.y * 0.5 + 0.5;
        let sky_bottom = vec3<f32>(0.55, 0.7, 0.9);
        let sky_top = vec3<f32>(0.3, 0.5, 0.85);
        color = vec4<f32>(mix(sky_bottom, sky_top, sky_t), 1.0);
    }

    textureStore(output_texture, vec2<i32>(global_id.xy), color);
}
