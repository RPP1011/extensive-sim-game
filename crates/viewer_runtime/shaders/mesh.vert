#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Per-instance shading + transform. Layout MUST match
// `viewer_runtime::mesh_renderer::InstanceData`.
struct InstanceData {
    mat4 model;
    vec4 tint;
    uint bone_matrix_offset;
    uint bone_matrix_count;
    uvec2 _pad;
};

layout(set = 0, binding = 0) readonly buffer Instances {
    InstanceData instances[];
} instance_buf;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

layout(location = 0) out vec4 v_color;
// World-space normal — fragment shader uses this for diffuse lighting.
// Y-axis-only rotation in InstanceData::model means the normal can be
// rotated identically (no inverse-transpose needed for uniform scale +
// rotation; no shear is introduced).
layout(location = 1) out vec3 v_world_normal;

void main() {
    InstanceData inst = instance_buf.instances[gl_InstanceIndex];
    vec4 world_pos = inst.model * vec4(in_position, 1.0);
    gl_Position = pc.view_proj * world_pos;
    v_color = inst.tint;
    // model is rotation*scale*translation; normal needs the rotation
    // part. mat3(model) handles this when scale is uniform — true for
    // all our agents.
    v_world_normal = normalize(mat3(inst.model) * in_normal);
}
