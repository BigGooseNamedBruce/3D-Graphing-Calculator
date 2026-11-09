#version 450

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 cameraPos; // xyz = camera world position, w = grid half-extent
    vec4 surface;   // xy = patch centre XZ, z = patch half-width
} pc;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

// Must match the render pass clear colour, or the patch border shows up as a
// seam instead of dissolving into the background.
const vec3 kBackground = vec3(0.04, 0.05, 0.08);

// Blue -> green -> red ramp keyed on surface height.
vec3 heightColor(float h) {
    float t   = clamp(h / 12.0, 0.0, 1.0);
    vec3  cold = vec3(0.10, 0.25, 0.85);
    vec3  mid  = vec3(0.10, 0.80, 0.35);
    vec3  hot  = vec3(0.90, 0.25, 0.10);
    return t < 0.5 ? mix(cold, mid, t * 2.0)
                   : mix(mid, hot, (t - 0.5) * 2.0);
}

void main() {
    vec3  N       = normalize(fragNormal);
    vec3  L       = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse = max(dot(N, L), 0.0);
    float ambient = 0.25;

    vec3 base = heightColor(fragWorldPos.y);
    vec3 color = base * (ambient + 0.85 * diffuse);

    // Aerial perspective. The far field is the coarsest part of the mesh and
    // the least trustworthy, so it is worth burying: fog thickens with distance
    // from the eye, which reads as depth and hides the patch boundary from most
    // viewpoints. Squared falloff leaves the near field completely clear
    // instead of laying a haze over everything.
    float eye = distance(fragWorldPos, pc.cameraPos.xyz) / (pc.surface.z * 0.55);
    float fade = 1.0 - exp(-eye * eye);

    // Backstop for the one case fog misses: sighting along the plane from just
    // inside the rim, where the boundary is close enough to stay clear. The
    // patch is finite and follows the camera, so its border would otherwise be
    // a hard square sliding along with you. Chebyshev distance matches the
    // square patch, so the dissolve is even along all four edges.
    vec2 d = abs(fragWorldPos.xz - pc.surface.xy);
    float edge = max(d.x, d.y) / pc.surface.z;
    fade = max(fade, smoothstep(0.80, 1.0, edge));

    outColor = vec4(mix(color, kBackground, fade), 1.0);
}
