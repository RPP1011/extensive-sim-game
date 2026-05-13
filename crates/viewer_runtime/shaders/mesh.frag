#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in float v_local_y;
layout(location = 0) out vec4 out_color;

void main() {
    // Fake vertical lighting — Kenney's chars are ~1.7 units tall in
    // model space with feet near y=0, head near y=1.7. Map that to a
    // brightness gradient (top brighter) so silhouettes read as 3D
    // shapes instead of flat colored cutouts. Cheap stand-in for
    // proper diffuse lighting until voxel_engine's pipeline builder
    // grows multi-attribute vertex input (then we can pass per-vertex
    // normals + light direction).
    float ambient = 0.45;
    float top = clamp(v_local_y * 0.4, 0.0, 0.55);
    float bright = ambient + top;
    out_color = vec4(v_color.rgb * bright, v_color.a);
}
