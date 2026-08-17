#version 450

layout(push_constant) uniform Push {
    mat4 viewProj;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;

void main() {
    // Vertices are already in world space (model matrix is identity), so the
    // incoming normal is the world-space normal and needs no transform.
    gl_Position  = pc.viewProj * vec4(inPos, 1.0);
    fragNormal   = inNormal;
    fragWorldPos = inPos;
}
