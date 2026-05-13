#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

// Per-instance data. Layout MUST match
// `viewer_runtime::mesh_renderer::InstanceData` (mat4 + vec4 + 4×u32).
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

void main() {
    InstanceData inst = instance_buf.instances[gl_InstanceIndex];
    vec4 world_pos = inst.model * vec4(in_position, 1.0);
    gl_Position = pc.view_proj * world_pos;
    v_color = in_color * inst.tint;
}
