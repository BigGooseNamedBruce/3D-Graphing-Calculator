#pragma once

// =============================================================================
// Device input to camera motion. Kept in its own header so the mapping from
// keys and mouse to movement lives in one place instead of being spread through
// the renderer.
//
// The mapping:
//   W A S D                walk along the ground plane, steered by yaw only
//   E / Space, Q / Ctrl    up, down along world-up
//   Shift                  sprint multiplier
//   left-drag              yaw and pitch
//   scroll                 fly speed (a fly camera has nothing to zoom toward)
//   Esc                    close the window
//
// These are free functions rather than a class that installs its own GLFW
// callbacks. GLFW allows a single user pointer per window and the app already
// needs it for framebuffer resize, so the app keeps ownership of the callbacks
// and delegates their bodies here — no second dispatch layer.
//
// Every entry point checks with ImGui first. Without that, dragging inside the
// panel would also swing the camera, and typing "s" into the expression field
// would fly you backwards.
// =============================================================================

#include "Camera.hpp" // FlyCamera; also brings glm configured for radians

#include <GLFW/glfw3.h>

#include "imgui.h"

#include <algorithm>

namespace Input {

// Drag and frame-timing state. Owned by the caller so this header holds no
// globals.
struct State {
    bool dragging = false;
    double lastX = 0.0;
    double lastY = 0.0;
    double lastFrameTime = 0.0;
};

constexpr float kLookSensitivity = 0.004f;
constexpr float kMaxFrameStep = 0.1f; // a hitch must not teleport the camera

// ImGui gets first refusal on the pointer and the keyboard, so dragging or
// typing inside the panel never also drives the camera.
inline bool imguiWantsMouse() {
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
}

inline bool imguiWantsKeyboard() {
    return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
}

// Per-frame look and fly. Call once before drawing.
inline void update(GLFWwindow* window, FlyCamera& camera, State& state) {
    const double now = glfwGetTime();
    float dt = static_cast<float>(now - state.lastFrameTime);
    state.lastFrameTime = now;
    dt = std::clamp(dt, 0.0f, kMaxFrameStep);

    if (imguiWantsMouse()) {
        state.dragging = false; // drop the drag if the pointer moved onto the panel
    } else {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!state.dragging) {
                state.dragging = true;
                state.lastX = x;
                state.lastY = y;
            }
            const double dx = x - state.lastX;
            const double dy = y - state.lastY;
            state.lastX = x;
            state.lastY = y;
            camera.rotate(static_cast<float>(dx) * kLookSensitivity,
                          static_cast<float>(-dy) * kLookSensitivity);
        } else {
            state.dragging = false;
        }
    }

    if (imguiWantsKeyboard()) {
        return; // the text field has the keys
    }

    auto down = [window](int k) { return glfwGetKey(window, k) == GLFW_PRESS; };
    glm::vec3 axes{0.0f};
    if (down(GLFW_KEY_W)) {
        axes.z += 1.0f;
    }
    if (down(GLFW_KEY_S)) {
        axes.z -= 1.0f;
    }
    if (down(GLFW_KEY_D)) {
        axes.x += 1.0f;
    }
    if (down(GLFW_KEY_A)) {
        axes.x -= 1.0f;
    }
    if (down(GLFW_KEY_E) || down(GLFW_KEY_SPACE)) {
        axes.y += 1.0f;
    }
    if (down(GLFW_KEY_Q) || down(GLFW_KEY_LEFT_CONTROL)) {
        axes.y -= 1.0f;
    }
    camera.move(axes, dt, down(GLFW_KEY_LEFT_SHIFT));
}

// Bound to the scroll wheel. On a fly camera there is nothing to zoom toward,
// so scrolling trades range for precision instead.
inline void scroll(FlyCamera& camera, double yoff) {
    if (imguiWantsMouse()) {
        return;
    }
    camera.zoom(static_cast<float>(yoff));
}

// True when a key event should close the window. Escape must not close while
// the panel wants the keyboard, or leaving the text field would quit instead.
inline bool requestsClose(int key, int action) {
    if (action != GLFW_PRESS) {
        return false;
    }
    if (imguiWantsKeyboard()) {
        return false;
    }
    return key == GLFW_KEY_ESCAPE;
}

} // namespace Input
