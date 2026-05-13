#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec3 v_world_normal;
layout(location = 0) out vec4 out_color;

void main() {
    // Sun-like directional light from above-and-side. Lambert diffuse
    // with non-zero ambient so back-facing surfaces don't go pitch
    // black at the camera angle. Light direction tuned for the
    // 3/4 isometric camera (eye is behind +Z + above +Y).
    vec3 light_dir = normalize(vec3(0.3, -1.0, -0.4));
    vec3 normal = normalize(v_world_normal);
    float diffuse = max(0.0, dot(normal, -light_dir));
    float ambient = 0.35;
    float bright = ambient + 0.65 * diffuse;
    out_color = vec4(v_color.rgb * bright, v_color.a);
}
