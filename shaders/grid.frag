#version 450

// Procedural grid.
//
// Line positions come from the world coordinate rather than from uploaded
// geometry, and coverage is divided by fwidth() — the screen-space derivative —
// so a line stays about one pixel wide however far away it is. Uploaded line
// geometry cannot do this: at distance the lines fall between samples and alias
// into crawling noise.

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 cameraPos; // xyz = camera world position, w = half-extent of the quad
} pc;

layout(location = 0) in vec2 inWorldXZ;
layout(location = 1) flat in int inIsAxis;

layout(location = 0) out vec4 outColor;

// Antialiased coverage in [0,1] for a grid of the given spacing.
float coverage(vec2 p, float spacing) {
    vec2 c = p / spacing;
    vec2 g = abs(fract(c - 0.5) - 0.5) / fwidth(c);
    return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
    // Fade out well inside the quad edge so the boundary is never visible.
    float dist = distance(inWorldXZ, pc.cameraPos.xz);
    float fade = 1.0 - smoothstep(pc.cameraPos.w * 0.10, pc.cameraPos.w * 0.55, dist);

    // The vertical axis is separate geometry, not part of the ground plane, so
    // it skips the grid maths. Green completes X red / Y green / Z blue.
    if (inIsAxis == 1) {
        if (fade < 0.002) {
            discard;
        }
        outColor = vec4(0.45, 0.90, 0.45, fade);
        return;
    }

    float minor = coverage(inWorldXZ, 1.0);
    float major = coverage(inWorldXZ, 10.0);

    vec3 color = mix(vec3(0.32), vec3(0.55), major);
    float alpha = max(minor * 0.45, major * 0.85);

    // Axes lying in the ground plane: X red, Z blue.
    vec2 axis = abs(inWorldXZ) / fwidth(inWorldXZ);
    if (axis.y < 1.0) {
        color = vec3(0.90, 0.35, 0.35);
        alpha = 1.0;
    }
    if (axis.x < 1.0) {
        color = vec3(0.35, 0.55, 0.95);
        alpha = 1.0;
    }

    alpha *= fade;
    if (alpha < 0.002) {
        discard;
    }
    outColor = vec4(color, alpha);
}
