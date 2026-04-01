#import bevy_pbr::forward_io::VertexOutput

struct VolumeInfo {
    size: vec3<f32>,
    _pad0: f32,
    origin: vec3<f32>,
    _pad1: f32,
};

@group(2) @binding(0) var sdf_texture: texture_3d<f32>;
@group(2) @binding(1) var sdf_sampler: sampler;
@group(2) @binding(2) var material_texture: texture_3d<u32>;
@group(2) @binding(3) var<uniform> volume: VolumeInfo;

const MAX_STEPS: i32 = 128;
const MAX_DIST: f32 = 300.0;
const HIT_DIST: f32 = 0.3;

fn sample_sdf(p: vec3<f32>) -> f32 {
    let uv = (p - volume.origin) / volume.size;
    if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) {
        return MAX_DIST;
    }
    return textureSampleLevel(sdf_texture, sdf_sampler, uv, 0.0).r;
}

fn sample_material(p: vec3<f32>) -> u32 {
    let local = p - volume.origin;
    let ip = vec3<i32>(floor(local));
    let vs = vec3<i32>(volume.size);
    if (any(ip < vec3<i32>(0)) || any(ip >= vs)) { return 0u; }
    return textureLoad(material_texture, ip, 0).r;
}

fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.5;
    return normalize(vec3<f32>(
        sample_sdf(p + vec3<f32>(e, 0.0, 0.0)) - sample_sdf(p - vec3<f32>(e, 0.0, 0.0)),
        sample_sdf(p + vec3<f32>(0.0, e, 0.0)) - sample_sdf(p - vec3<f32>(0.0, e, 0.0)),
        sample_sdf(p + vec3<f32>(0.0, 0.0, e)) - sample_sdf(p - vec3<f32>(0.0, 0.0, e)),
    ));
}

fn mat_color(m: u32) -> vec3<f32> {
    switch m {
        case 1u: { return vec3<f32>(0.45, 0.32, 0.18); }   // Dirt
        case 2u: { return vec3<f32>(0.50, 0.50, 0.50); }   // Stone
        case 3u: { return vec3<f32>(0.35, 0.33, 0.32); }   // Granite
        case 4u: { return vec3<f32>(0.85, 0.78, 0.55); }   // Sand
        case 5u: { return vec3<f32>(0.60, 0.45, 0.30); }   // Clay
        case 6u: { return vec3<f32>(0.55, 0.52, 0.48); }   // Gravel
        case 7u: { return vec3<f32>(0.25, 0.55, 0.18); }   // Grass
        case 8u: { return vec3<f32>(0.15, 0.35, 0.65); }   // Water
        case 12u: { return vec3<f32>(0.55, 0.40, 0.30); }  // IronOre
        case 14u: { return vec3<f32>(0.80, 0.70, 0.20); }  // GoldOre
        case 15u: { return vec3<f32>(0.15, 0.15, 0.15); }  // Coal
        case 16u: { return vec3<f32>(0.60, 0.30, 0.80); }  // Crystal
        case 17u: { return vec3<f32>(0.50, 0.35, 0.20); }  // WoodLog
        case 18u: { return vec3<f32>(0.65, 0.50, 0.30); }  // WoodPlanks
        case 24u: { return vec3<f32>(0.35, 0.25, 0.12); }  // Farmland
        case 25u: { return vec3<f32>(0.20, 0.60, 0.10); }  // Crop
        default: { return vec3<f32>(0.60, 0.55, 0.50); }
    }
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Ray from camera through this fragment's world position
    let ray_origin = in.world_position.xyz;
    // Ray direction: from camera toward this fragment
    // We use the fragment's world normal as a proxy for the view direction
    // since this is on a large box surrounding the volume
    let ray_dir = normalize(in.world_position.xyz - volume.origin - volume.size * 0.5);

    var t: f32 = 0.0;
    var hit = false;
    var hit_pos = vec3<f32>(0.0);

    // Start raymarching from the fragment's world position
    for (var i: i32 = 0; i < MAX_STEPS; i++) {
        let pos = ray_origin - ray_dir * t; // march inward from surface of box
        let d = sample_sdf(pos);

        if (d < HIT_DIST && d < 0.0) {
            // Inside solid — back up and refine
            hit = true;
            hit_pos = pos;
            break;
        }

        if (abs(d) < HIT_DIST) {
            hit = true;
            hit_pos = pos;
            break;
        }

        t += max(abs(d), 0.5);
        if (t > MAX_DIST) { break; }
    }

    if (!hit) {
        discard;
    }

    let n = sdf_normal(hit_pos);
    let mat_id = sample_material(hit_pos);
    let base = mat_color(mat_id);

    // Lighting
    let light = normalize(vec3<f32>(0.3, 0.8, 0.4));
    let diff = max(dot(n, light), 0.0);
    let amb = 0.25;
    let lit = base * (amb + diff * 0.75);

    // Distance fog
    let fog = clamp(t / 200.0, 0.0, 0.6);
    let sky = vec3<f32>(0.55, 0.70, 0.90);
    let final_color = mix(lit, sky, fog);

    return vec4<f32>(final_color, 1.0);
}
