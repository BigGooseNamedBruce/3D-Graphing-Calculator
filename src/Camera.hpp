#pragma once

// =============================================================================
// Free-fly camera: a world position plus a yaw/pitch facing, and the view
// matrix built from the two.
//
// Replaces an earlier orbit camera, which could only circle a fixed point and
// so could never get inside or underneath a surface — awkward when the whole
// point of the app is inspecting a graph from any angle.
//
// Aiming and movement are deliberately decoupled. Pitch steers the view but not
// the walk direction, so pressing W while looking at your feet moves you across
// the ground rather than into it. Changing altitude is an explicit Q/E action.
//
// There is no input code here. This file only knows how to move and aim; which
// key or mouse motion triggers which movement is decided in Input.hpp, which
// calls rotate(), move() and zoom() below.
// =============================================================================

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

// Free-fly camera: mouse aims, WASD walks along the ground plane, Q/E change
// height. Replaces the previous orbit camera, which could only circle a fixed
// point and so could not get inside or underneath a surface.
class FlyCamera {
public:
    glm::vec3 position{0.0f, 12.0f, 22.0f};

    // Spherical facing. Defaults look down at the origin from the start
    // position, which is where the surface is generated.
    float yaw = glm::radians(-90.0f);
    float pitch = glm::radians(-28.0f);

    float moveSpeed = 10.0f; // world units / second
    float sprintMul = 3.0f;

    void rotate(float dYaw, float dPitch) {
        yaw += dYaw;
        pitch += dPitch;
        // Stop just short of vertical: at exactly +-90 the forward vector goes
        // parallel to world-up and lookAt's basis becomes degenerate.
        const float lim = glm::radians(89.0f);
        pitch = std::clamp(pitch, -lim, lim);
    }

    // Bound to the scroll wheel. On a fly camera there is nothing to zoom
    // toward, so scrolling trades range for precision instead.
    void zoom(float scrollDelta) {
        moveSpeed *= std::pow(1.15f, scrollDelta);
        moveSpeed = std::clamp(moveSpeed, 0.5f, 200.0f);
    }

    // Axis amounts in [-1, 1]: +x right, +y world-up, +z forward.
    //
    // Forward here is groundForward(), not forward(): looking down and pressing
    // W walks along the ground rather than diving into it, which is what makes
    // the controls predictable when you are studying a surface from above.
    // Vertical movement is deliberate, on Q/E, rather than a side effect of
    // where you happen to be aiming.
    void move(const glm::vec3& axes, float dt, bool sprint = false) {
        if (axes == glm::vec3(0.0f)) {
            return;
        }
        const float step = moveSpeed * (sprint ? sprintMul : 1.0f) * dt;
        glm::vec3 delta = right() * axes.x + worldUp() * axes.y + groundForward() * axes.z;
        if (glm::length(delta) > 1e-6f) {
            position += glm::normalize(delta) * step;
        }
    }

    // True facing, including pitch. This is where the camera looks, so view()
    // uses it — only movement is flattened.
    glm::vec3 forward() const {
        return glm::normalize(glm::vec3(std::cos(pitch) * std::cos(yaw),
                                        std::sin(pitch),
                                        std::cos(pitch) * std::sin(yaw)));
    }

    // Facing projected onto the ground plane. Already unit length, and unlike
    // flattening forward() it stays well defined when looking straight down.
    glm::vec3 groundForward() const {
        return glm::vec3(std::cos(yaw), 0.0f, std::sin(yaw));
    }

    // Derived from groundForward() so strafing is horizontal and the magnitude
    // before normalising is always 1. Built from forward() it would shrink to
    // cos(pitch), which is 0.017 at the 89-degree clamp.
    glm::vec3 right() const {
        return glm::normalize(glm::cross(groundForward(), worldUp()));
    }

    static glm::vec3 worldUp() { return glm::vec3(0.0f, 1.0f, 0.0f); }

    glm::mat4 view() const {
        return glm::lookAt(position, position + forward(), worldUp());
    }
};
