#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_world_normal;
layout(location = 0) out vec4 out_color;

void main() {
    // DIAG: solid bright magenta to test whether the mesh pass is
    // rasterizing at all. If meshes show as magenta blobs, lighting is
    // the bug. If they STILL don't show, problem is upstream (vertex
    // transform, pipeline state, render pass binding, etc).
    out_color = vec4(1.0, 0.0, 1.0, 1.0);
}
