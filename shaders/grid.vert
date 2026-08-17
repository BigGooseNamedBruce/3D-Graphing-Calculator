#version 450

// Two pieces in one bufferless draw, selected by gl_VertexIndex:
//
//   0..5   ground plane — a quad centred on the camera so it never runs out.
//          Its lines are drawn procedurally in the fragment shader, so no line
//          geometry is uploaded and nothing regenerates when you move.
//   6..11  the vertical axis (Z when graphing, +Y in world space) — a ribbon at
//          the origin that rotates to face the camera. A flat line would
//          disappear edge-on, and Vulkan cannot draw a thick line without the
//          wideLines device feature.

layout(push_constant) uniform Push {
    mat4 viewProj;
    vec4 cameraPos; // xyz = camera world position, w = half-extent of the quad
} pc;

layout(location = 0) out vec2 outWorldXZ;
layout(location = 1) flat out int outIsAxis;

// Ribbon half-width as a fraction of viewing distance, which keeps it roughly
// constant on screen instead of shrinking away as you fly off.
const float kAxisWidth = 0.0012;

void main() {
    const vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
        vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0));

    if (gl_VertexIndex < 6) {
        vec2 xz = pc.cameraPos.xz + corners[gl_VertexIndex] * pc.cameraPos.w;
        outWorldXZ = xz;
        outIsAxis = 0;
        gl_Position = pc.viewProj * vec4(xz.x, 0.0, xz.y, 1.0);
        return;
    }

    vec2 c = corners[gl_VertexIndex - 6];
    float y = c.y * pc.cameraPos.w * 0.25;

    // Billboard about world up: offset sideways, perpendicular to the direction
    // from the axis out to the camera.
    float horiz = max(length(pc.cameraPos.xz), 1.0);
    vec2 toCamera = pc.cameraPos.xz / horiz;
    vec2 side = vec2(-toCamera.y, toCamera.x);

    // One width for the whole ribbon, measured at its closest approach to the
    // camera. Scaling per vertex made it a trapezoid: the corners sit hundreds
    // of units above and below, so their width interpolated across the middle —
    // the part actually in view — and came out enormously too thick.
    vec2 offset = side * (c.x * horiz * kAxisWidth);
    vec3 world = vec3(offset.x, y, offset.y);

    outWorldXZ = world.xz;
    outIsAxis = 1;
    gl_Position = pc.viewProj * vec4(world, 1.0);
}
