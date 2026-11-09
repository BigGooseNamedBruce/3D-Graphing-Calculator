#pragma once

// =============================================================================
// The application: owns the window, the Vulkan device and swapchain, the render
// loop, and the ImGui panel that edits the surface equation.
//
// This is by far the largest file in the project because Vulkan setup is
// verbose — roughly 1100 of its lines are instance/device/swapchain/pipeline
// ceremony that every Vulkan program needs and none of which is specific to
// graphing. Banner comments mark the seams:
//
//   cfg              window size, frames in flight, validation toggle
//   PipelineBuilder  chained construction for the surface and grid pipelines
//   Window + input   GLFW window and callbacks, delegating into Input.hpp
//   Vulkan setup     instance -> device -> swapchain -> pipelines -> commands
//
// One frame, in order (see mainLoop):
//
//   1. processInput()           keys and mouse move the camera
//   2. buildUI()                ImGui panel; Apply recompiles the expression
//   3. consumeLatestSnapshot()  copy the newest finished mesh into the vertex
//                               buffer, which stays mapped for the lifetime of
//                               the process so this is a plain memcpy
//   4. drawFrame()              acquire, record, submit, present
//
// Mesh generation deliberately does NOT happen on that thread. A producer
// thread re-evaluates f(x, y, t) over a 160x160 grid and publishes finished
// snapshots into a ring buffer with a release store; the render thread picks up
// the newest one with an acquire load. A rebuild costs roughly 128k expression
// evaluations, so doing it inline would stall every frame.
//
// Elsewhere: expression parsing in Expression.hpp, mesh construction in
// GraphMesh.hpp, camera maths in Camera.hpp, and the mapping from keys and
// mouse to camera motion in Input.hpp.
// =============================================================================

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "Camera.hpp"
#include "Expression.hpp"
#include "GraphMesh.hpp"
#include "Input.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace cfg {
constexpr uint32_t WIDTH = 1280;
constexpr uint32_t HEIGHT = 720;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// The surface is a fixed-size patch that follows the camera, so DOMAIN_HALF is
// how far it reaches around you rather than a fixed region of the plane.
//
// Sample spacing is not uniform: LOD_FALLOFF is the ratio between the outermost
// and innermost cell, so the same vertex budget covers far more ground. At a
// reach of 300 and a falloff of 50 the near cells are about 0.15 units and the
// far ones about 7.5 — which is how reach went up twentyfold without costing a
// single extra evaluation.
//
// Vertex count is the only thing that costs CPU time: every rebuild is
// RESOLUTION^2 vertices, each box-filtered over its own cell, spread across all
// cores. Reach and falloff are free; resolution is not.
//
// MESH_RESOLUTION must be used everywhere a mesh is built: the vertex buffer is
// sized from the first mesh generated and every later snapshot is memcpy'd into
// it, so a mismatch would be a silent buffer overrun.
constexpr float DOMAIN_HALF = 300.0f;
constexpr float LOD_FALLOFF = 50.0f;
constexpr int MESH_RESOLUTION = 320;

// The patch centre snaps to whole multiples of the innermost cell, so near-field
// vertices keep landing on the same world positions as the camera moves. Not
// constexpr because the warp needs log and exp; it is a few instructions once a
// frame.
inline float snapStep() {
    return GraphMesh::centreCellSize(DOMAIN_HALF, LOD_FALLOFF, MESH_RESOLUTION);
}

// Near distance matters far more than far distance for depth precision, so it
// is not pushed lower than it needs to be. The surface crosses y = 0, which is
// exactly where the ground plane sits, so that is where z-fighting would show
// up first.
constexpr float NEAR_PLANE = 0.5f;
constexpr float FAR_PLANE = 5000.0f;

// Half-extent of the camera-centred ground quad. The grid fades out well
// inside this, so its edge is never visible.
constexpr float GRID_EXTENT = 2000.0f;

// Ceiling on multisampling. The far field is mostly triangles a pixel or two
// across, so their edges are where the remaining aliasing lives; 4x takes the
// stair-stepping off them, and 8x costs the bandwidth again for little more.
// The spec floor for both colour and depth sample counts includes 4x, so this
// is always what a device ends up with.
constexpr VkSampleCountFlagBits MAX_MSAA_SAMPLES = VK_SAMPLE_COUNT_4_BIT;

inline const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif
} // namespace cfg

struct PushConstants {
    glm::mat4 viewProj;
    glm::vec4 cameraPos; // xyz = camera world position, w = cfg::GRID_EXTENT
    glm::vec4 surface;   // xy = patch centre XZ, z = cfg::DOMAIN_HALF, w unused
};

class PipelineBuilder {
public:
    PipelineBuilder& setShaders(std::string vert, std::string frag) {
        vertPath_ = std::move(vert);
        fragPath_ = std::move(frag);
        return *this;
    }

    // Leaving this uncalled gives the pipeline no vertex input at all, for
    // shaders that build their geometry from gl_VertexIndex.
    PipelineBuilder& setVertexBinding(uint32_t stride) {
        binding_ = {0, stride, VK_VERTEX_INPUT_RATE_VERTEX};
        hasBinding_ = true;
        return *this;
    }

    PipelineBuilder& setAlphaBlend(bool enable) {
        alphaBlend_ = enable;
        return *this;
    }

    // Blended geometry should test depth but not write it, or it occludes what
    // is meant to be visible through it.
    PipelineBuilder& setDepthWrite(bool enable) {
        depthWrite_ = enable;
        return *this;
    }

    PipelineBuilder& addAttribute(uint32_t location, VkFormat format, uint32_t offset) {
        attrs_.push_back({location, 0, format, offset});
        return *this;
    }

    PipelineBuilder& setTopology(VkPrimitiveTopology topology) {
        topology_ = topology;
        return *this;
    }

    // Must match the render pass, which is multisampled.
    PipelineBuilder& setSampleCount(VkSampleCountFlagBits samples) {
        samples_ = samples;
        return *this;
    }

    VkPipeline build(VkDevice device, VkRenderPass renderPass, VkPipelineLayout layout) {
        VkShaderModule vert = loadShader(device, vertPath_);
        VkShaderModule frag = loadShader(device, fragPath_);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = hasBinding_ ? 1u : 0u;
        vertexInput.pVertexBindingDescriptions = hasBinding_ ? &binding_ : nullptr;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs_.size());
        vertexInput.pVertexAttributeDescriptions = attrs_.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = topology_;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.lineWidth = 1.0f;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = samples_;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = depthWrite_ ? VK_TRUE : VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttachment.blendEnable = alphaBlend_ ? VK_TRUE : VK_FALSE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blendAttachment;

        std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{};
        dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dynamic.pDynamicStates = dynStates.data();

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState = &viewportState;
        ci.pRasterizationState = &raster;
        ci.pMultisampleState = &multisample;
        ci.pDepthStencilState = &depthStencil;
        ci.pColorBlendState = &colorBlend;
        ci.pDynamicState = &dynamic;
        ci.layout = layout;
        ci.renderPass = renderPass;
        ci.subpass = 0;

        VkPipeline pipeline;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline");
        }

        vkDestroyShaderModule(device, frag, nullptr);
        vkDestroyShaderModule(device, vert, nullptr);
        return pipeline;
    }

private:
    std::string vertPath_, fragPath_;
    VkVertexInputBindingDescription binding_{};
    bool hasBinding_ = false;
    std::vector<VkVertexInputAttributeDescription> attrs_;
    VkPrimitiveTopology topology_ = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
    bool alphaBlend_ = false;
    bool depthWrite_ = true;

    static VkShaderModule loadShader(VkDevice device, const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open shader: " + path);
        }
        size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buf(size);
        file.seekg(0);
        file.read(buf.data(), static_cast<std::streamsize>(size));

        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = buf.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(buf.data());
        VkShaderModule module;
        if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module: " + path);
        }
        return module;
    }
};

class GraphingCalculator3D {
public:
    // An opaque callable could depend on anything, so it is assumed to animate
    // and the producer keeps rebuilding it. install() below knows better for
    // parsed expressions.
    void setFunction(std::function<float(float, float, float)> f) {
        install(std::move(f), true);
    }

    // Compile and install a source expression, e.g. "sin(3x)*cos(3y) + 0.2t".
    // Returns false and leaves the current surface untouched on a parse error.
    bool setExpression(const std::string& source, std::string* errorOut = nullptr) {
        expr::ParseResult r = expr::Parser::parse(source);
        if (!r.ok) {
            if (errorOut) {
                *errorOut = r.error + " (col " + std::to_string(r.position + 1) + ")";
            }
            return false;
        }
        installCompiled(std::move(r.expression));
        activeSource_ = source;
        std::snprintf(inputBuf_, sizeof(inputBuf_), "%s", source.c_str());
        return true;
    }

    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

    // Public so the static GLFW callbacks can reach them.
    bool framebufferResized = false;
    FlyCamera camera;

private:
    // ---------- types ----------
    struct QueueFamilyIndices {
        std::optional<uint32_t> graphics;
        std::optional<uint32_t> present;
        bool complete() const { return graphics.has_value() && present.has_value(); }
    };

    struct SwapChainSupport {
        VkSurfaceCapabilitiesKHR caps{};
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    // ---------- members ----------
    GLFWwindow* window_ = nullptr;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages_;
    VkFormat swapChainImageFormat_{};
    VkExtent2D swapChainExtent_{};
    std::vector<VkImageView> swapChainImageViews_;
    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;

    VkSampleCountFlagBits msaaSamples_ = VK_SAMPLE_COUNT_1_BIT; // set when the GPU is picked

    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;

    // Multisampled colour target, resolved into the swapchain image.
    VkImage colorImage_ = VK_NULL_HANDLE;
    VkDeviceMemory colorImageMemory_ = VK_NULL_HANDLE;
    VkImageView colorImageView_ = VK_NULL_HANDLE;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    VkBuffer vertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory_ = VK_NULL_HANDLE;
    VkBuffer indexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory_ = VK_NULL_HANDLE;
    uint32_t indexCount_ = 0;

    VkPipeline gridPipeline_ = VK_NULL_HANDLE;

    std::vector<VkSemaphore> imageAvailableSemaphores_; // per frame-in-flight
    std::vector<VkSemaphore> renderFinishedSemaphores_; // per swapchain image
    std::vector<VkFence> inFlightFences_; // per frame-in-flight
    uint32_t currentFrame_ = 0;

    std::function<float(float, float, float)> func_ =
        [](float x, float y, float t) { return x * x + y * y; };

    // Ring buffer: producer thread writes snapshots, render thread reads latest.
    static constexpr int RING_SIZE = 4; // must be power of 2
    struct Snapshot { std::vector<Vertex> vertices; };
    Snapshot ring_[RING_SIZE];
    std::atomic<uint32_t> writeIdx_{0}; // producer increments after each write
    uint32_t uploadedIdx_ = 0; // render thread: writeIdx_ as of the last upload
    void* vertexMapped_ = nullptr; // persistent CPU-side pointer into vertexBuffer_

    // Bumped whenever func_ is replaced, so the producer can tell a new surface
    // from one it has already built. Animated surfaces are rebuilt every pass
    // regardless; still ones only when this or the patch centre changes.
    std::atomic<uint32_t> funcGeneration_{0};
    std::atomic<bool> funcAnimated_{false};

    std::thread producerThread_;
    std::atomic<bool> producerRunning_{false};

    // Centre of the patch the producer should build next. Written by the render
    // thread and read by the producer, so it cannot just reach into
    // camera.position — that would be a data race on every frame.
    std::atomic<float> patchCenterX_{0.0f};
    std::atomic<float> patchCenterZ_{0.0f};

    Input::State input_;

    // Expression editing. ImGui owns the text field; Apply compiles the buffer
    // and, on success, swaps func_ for the producer thread.
    char inputBuf_[256] = "x*x + y*y";
    std::string activeSource_ = "x*x + y*y";
    std::string parseError_;

    VkDescriptorPool imguiPool_ = VK_NULL_HANDLE;

    // func_ is written from the input handler and read every frame by the
    // producer thread, so it needs guarding. The producer copies it once per
    // snapshot rather than locking per-vertex.
    std::mutex funcMutex_;

    // ======================================================================
    // Window + input
    // ======================================================================
    void initWindow() {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // no OpenGL context
        window_ = glfwCreateWindow(cfg::WIDTH, cfg::HEIGHT,
                                   "Vulkan 3D Graphing Calculator", nullptr, nullptr);
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
        glfwSetScrollCallback(window_, scrollCallback);
        glfwSetKeyCallback(window_, keyCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* w, int, int) {
        auto app = reinterpret_cast<GraphingCalculator3D*>(glfwGetWindowUserPointer(w));
        app->framebufferResized = true;
    }
    static void scrollCallback(GLFWwindow* w, double, double yoff) {
        auto app = reinterpret_cast<GraphingCalculator3D*>(glfwGetWindowUserPointer(w));
        Input::scroll(app->camera, yoff);
    }

    static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
        if (Input::requestsClose(key, action)) {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
    }

    // Compile what has been typed. On failure the previous surface keeps
    // rendering and the error shows in the panel, so a typo never blanks the view.
    void applyExpression() {
        expr::ParseResult r = expr::Parser::parse(inputBuf_);
        if (!r.ok) {
            parseError_ = r.error + " (col " + std::to_string(r.position + 1) + ")";
            return;
        }
        parseError_.clear();
        activeSource_ = inputBuf_;
        installCompiled(std::move(r.expression));
    }

    // Shared, immutable, and captured by value: the producer thread keeps a live
    // reference for as long as it is mid-snapshot, so replacing func_ here
    // cannot pull the expression out from under it.
    void installCompiled(expr::Expression parsed) {
        auto compiled = std::make_shared<const expr::Expression>(std::move(parsed));
        install([compiled](float x, float y, float t) { return compiled->eval(x, y, t); },
                compiled->usesTime());
    }

    // The generation counter is what tells the producer a new surface is
    // waiting; without it a swap to a different static expression would be
    // indistinguishable from no change at all and never get rebuilt. Bumped
    // last, so the producer that observes it also observes the new func_.
    void install(std::function<float(float, float, float)> f, bool animated) {
        {
            std::lock_guard<std::mutex> lock(funcMutex_);
            func_ = std::move(f);
        }
        funcAnimated_.store(animated, std::memory_order_relaxed);
        funcGeneration_.fetch_add(1, std::memory_order_release);
    }

    void processInput() {
        Input::update(window_, camera, input_);
    }

    // Publish where the next patch should be built. Snapping to a whole cell is
    // what keeps the surface still: unsnapped, every vertex resamples f at a
    // slightly different point each frame and the whole sheet crawls.
    void updatePatchCenter() {
        const float step = cfg::snapStep();
        patchCenterX_.store(std::floor(camera.position.x / step) * step,
                            std::memory_order_relaxed);
        patchCenterZ_.store(std::floor(camera.position.z / step) * step,
                            std::memory_order_relaxed);
    }

    // ======================================================================
    // Vulkan setup
    // ======================================================================
    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createGridPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        createMeshBuffers();
        createCommandBuffers();
        createSyncObjects();
        initImGui();
        startProducer();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            processInput();
            updatePatchCenter();
            buildUI();
            consumeLatestSnapshot();
            drawFrame();
        }
        vkDeviceWaitIdle(device_);
    }

    // ---------------------------------------------------------------- instance
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                  VkDebugUtilsMessageTypeFlagsEXT,
                  const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            std::cerr << "[validation] " << data->pMessage << '\n';
        }
        return VK_FALSE;
    }

    void populateDebugInfo(VkDebugUtilsMessengerCreateInfoEXT& info) {
        info = {};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = debugCallback;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwCount = 0;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&glfwCount);
        std::vector<const char*> ext(glfwExt, glfwExt + glfwCount);
        if (cfg::kEnableValidation) {
            ext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
#ifdef __APPLE__
        ext.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        ext.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif
        return ext;
    }

    bool checkValidationLayerSupport() {
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (const char* name : cfg::kValidationLayers) {
            bool found = false;
            for (const auto& props : available) {
                if (std::strcmp(name, props.layerName) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    void createInstance() {
        if (cfg::kEnableValidation && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested but unavailable");
        }

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "Vulkan Grapher";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName = "none";
        app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app;

        const auto extensions = getRequiredExtensions();
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
#ifdef __APPLE__
        // Required so MoltenVK (a non-conformant "portability" driver) is listed.
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VkDebugUtilsMessengerCreateInfoEXT dbg{};
        if (cfg::kEnableValidation) {
            ci.enabledLayerCount = static_cast<uint32_t>(cfg::kValidationLayers.size());
            ci.ppEnabledLayerNames = cfg::kValidationLayers.data();
            populateDebugInfo(dbg);
            ci.pNext = &dbg;
        }

        if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create Vulkan instance");
        }
    }

    void setupDebugMessenger() {
        if (!cfg::kEnableValidation) {
            return;
        }
        VkDebugUtilsMessengerCreateInfoEXT info{};
        populateDebugInfo(info);
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) {
            fn(instance_, &info, nullptr, &debugMessenger_);
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface");
        }
    }

    // ---------------------------------------------------------------- device
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
        QueueFamilyIndices indices;
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphics = i;
            }
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
            if (present) {
                indices.present = i;
            }
            if (indices.complete()) {
                break;
            }
        }
        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
        for (const auto& e : available) {
            if (std::strcmp(e.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                return true;
            }
        }
        return false;
    }

    SwapChainSupport querySwapChainSupport(VkPhysicalDevice dev) {
        SwapChainSupport d;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface_, &d.caps);

        uint32_t fmtCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &fmtCount, nullptr);
        d.formats.resize(fmtCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface_, &fmtCount, d.formats.data());

        uint32_t pmCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &pmCount, nullptr);
        d.presentModes.resize(pmCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface_, &pmCount, d.presentModes.data());
        return d;
    }

    bool isDeviceSuitable(VkPhysicalDevice dev) {
        if (!findQueueFamilies(dev).complete()) {
            return false;
        }
        if (!checkDeviceExtensionSupport(dev)) {
            return false;
        }
        auto sc = querySwapChainSupport(dev);
        return !sc.formats.empty() && !sc.presentModes.empty();
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        if (count == 0) {
            throw std::runtime_error("no Vulkan-capable GPU found");
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance_, &count, devices.data());

        // Prefer a discrete GPU, otherwise take the first suitable device.
        for (auto dev : devices) {
            if (!isDeviceSuitable(dev)) {
                continue;
            }
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(dev, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physicalDevice_ = dev;
                break;
            }
            if (physicalDevice_ == VK_NULL_HANDLE) {
                physicalDevice_ = dev;
            }
        }
        if (physicalDevice_ == VK_NULL_HANDLE) {
            throw std::runtime_error("no suitable GPU found");
        }
        msaaSamples_ = maxUsableSampleCount();
    }

    // Colour and depth are resolved out of the same subpass, so only counts
    // both attachments can be created with are usable.
    VkSampleCountFlagBits maxUsableSampleCount() {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        const VkSampleCountFlags counts = props.limits.framebufferColorSampleCounts &
                                          props.limits.framebufferDepthSampleCounts;
        for (VkSampleCountFlagBits bit : {VK_SAMPLE_COUNT_8_BIT, VK_SAMPLE_COUNT_4_BIT,
                                          VK_SAMPLE_COUNT_2_BIT}) {
            if (bit <= cfg::MAX_MSAA_SAMPLES && (counts & bit)) {
                return bit;
            }
        }
        return VK_SAMPLE_COUNT_1_BIT;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);

        std::set<uint32_t> uniqueFamilies = {*indices.graphics, *indices.present};
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        float priority = 1.0f;
        for (uint32_t fam : uniqueFamilies) {
            VkDeviceQueueCreateInfo qi{};
            qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = fam;
            qi.queueCount = 1;
            qi.pQueuePriorities = &priority;
            queueInfos.push_back(qi);
        }

        VkPhysicalDeviceFeatures features{}; // none required

        std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        // "VK_KHR_portability_subset" must be enabled if the device exposes it
        // (e.g. MoltenVK on macOS), per the Vulkan spec.
        {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> available(extCount);
            vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, available.data());
            for (const auto& e : available) {
                if (std::strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
                    deviceExtensions.push_back("VK_KHR_portability_subset");
                }
            }
        }

        VkDeviceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        ci.pQueueCreateInfos = queueInfos.data();
        ci.pEnabledFeatures = &features;
        ci.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        ci.ppEnabledExtensionNames = deviceExtensions.data();
        if (cfg::kEnableValidation) {
            ci.enabledLayerCount = static_cast<uint32_t>(cfg::kValidationLayers.size());
            ci.ppEnabledLayerNames = cfg::kValidationLayers.data();
        }

        if (vkCreateDevice(physicalDevice_, &ci, nullptr, &device_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device");
        }

        vkGetDeviceQueue(device_, *indices.graphics, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, *indices.present, 0, &presentQueue_);
    }

    // ---------------------------------------------------------------- swapchain
    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        }
        return formats[0];
    }

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
        for (auto m : modes) {
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
                return m;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR; // always available
    }

    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps) {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return caps.currentExtent;
        }
        int w, h;
        glfwGetFramebufferSize(window_, &w, &h);
        VkExtent2D extent{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
        extent.width = std::clamp(extent.width, caps.minImageExtent.width,
                                   caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, caps.minImageExtent.height,
                                   caps.maxImageExtent.height);
        return extent;
    }

    void createSwapChain() {
        SwapChainSupport support = querySwapChainSupport(physicalDevice_);
        VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
        VkExtent2D extent = chooseExtent(support.caps);

        uint32_t imageCount = support.caps.minImageCount + 1;
        if (support.caps.maxImageCount > 0 && imageCount > support.caps.maxImageCount) {
            imageCount = support.caps.maxImageCount;
        }

        VkSwapchainCreateInfoKHR ci{};
        ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface = surface_;
        ci.minImageCount = imageCount;
        ci.imageFormat = surfaceFormat.format;
        ci.imageColorSpace = surfaceFormat.colorSpace;
        ci.imageExtent = extent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        uint32_t famIndices[] = {*indices.graphics, *indices.present};
        if (indices.graphics != indices.present) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = famIndices;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        ci.preTransform = support.caps.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode = presentMode;
        ci.clipped = VK_TRUE;
        ci.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device_, &ci, nullptr, &swapChain_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
        swapChainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages_.data());
        swapChainImageFormat_ = surfaceFormat.format;
        swapChainExtent_ = extent;
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
        VkImageViewCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format;
        ci.subresourceRange.aspectMask = aspect;
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = 1;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = 1;
        VkImageView view;
        if (vkCreateImageView(device_, &ci, nullptr, &view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view");
        }
        return view;
    }

    void createImageViews() {
        swapChainImageViews_.resize(swapChainImages_.size());
        for (size_t i = 0; i < swapChainImages_.size(); ++i) {
            swapChainImageViews_[i] = createImageView(
                swapChainImages_[i], swapChainImageFormat_, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    // ---------------------------------------------------------------- depth
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                 VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat f : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, f, &props);
            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (props.linearTilingFeatures & features) == features)
                return f;
            if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                (props.optimalTilingFeatures & features) == features)
                return f;
        }
        throw std::runtime_error("failed to find supported depth format");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    void createImage(uint32_t w, uint32_t h, VkFormat format, VkSampleCountFlagBits samples,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags props,
                     VkImage& image, VkDeviceMemory& memory) {
        VkImageCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = format;
        ci.tiling = tiling;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = usage;
        ci.samples = samples;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(device_, &ci, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image");
        }

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, image, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
        if (vkAllocateMemory(device_, &ai, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory");
        }
        vkBindImageMemory(device_, image, memory, 0);
    }

    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent_.width, swapChainExtent_.height, depthFormat, msaaSamples_,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage_, depthImageMemory_);
        depthImageView_ = createImageView(depthImage_, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // The multisampled colour buffer everything is drawn into. It is resolved
    // down to the swapchain image at the end of the subpass and never read
    // afterwards, hence TRANSIENT: a tiler can keep it entirely on-chip.
    void createColorResources() {
        createImage(swapChainExtent_.width, swapChainExtent_.height, swapChainImageFormat_,
                    msaaSamples_, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage_, colorImageMemory_);
        colorImageView_ = createImageView(colorImage_, swapChainImageFormat_,
                                          VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // ---------------------------------------------------------------- render pass
    void createRenderPass() {
        // Rendering target: multisampled, and discarded once resolved, so it is
        // never stored and never presented.
        VkAttachmentDescription color{};
        color.format = swapChainImageFormat_;
        color.samples = msaaSamples_;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depth{};
        depth.format = findDepthFormat();
        depth.samples = msaaSamples_;
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        // The swapchain image, which only ever receives the resolve. Its prior
        // contents are irrelevant, so it is not loaded.
        VkAttachmentDescription resolve{};
        resolve.format = swapChainImageFormat_;
        resolve.samples = VK_SAMPLE_COUNT_1_BIT;
        resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        resolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference resolveRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;
        subpass.pResolveAttachments = &resolveRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = {color, depth, resolve};
        VkRenderPassCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments = attachments.data();
        ci.subpassCount = 1;
        ci.pSubpasses = &subpass;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;
        if (vkCreateRenderPass(device_, &ci, nullptr, &renderPass_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass");
        }
    }

    // ---------------------------------------------------------------- pipeline
    static std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file: " + path);
        }
        size_t size = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(size);
        file.seekg(0);
        file.read(buffer.data(), static_cast<std::streamsize>(size));
        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module;
        if (vkCreateShaderModule(device_, &ci, nullptr, &module) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module");
        }
        return module;
    }

    void createGraphicsPipeline() {
        VkPushConstantRange pushRange{};
        // The grid fragment shader reads cameraPos for its distance fade.
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;
        if (vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout");
        }

        graphicsPipeline_ = PipelineBuilder{}
            .setShaders(std::string(SHADER_DIR) + "/graph.vert.spv",
                        std::string(SHADER_DIR) + "/graph.frag.spv")
            .setVertexBinding(sizeof(Vertex))
            .addAttribute(0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos))
            .addAttribute(1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal))
            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .setSampleCount(msaaSamples_)
            .build(device_, renderPass_, pipelineLayout_);
    }

    // No vertex input: grid.vert builds a camera-centred quad from
    // gl_VertexIndex. Blended, and depth-write off so the transparent gaps
    // between lines do not occlude the surface drawn before it.
    void createGridPipeline() {
        gridPipeline_ = PipelineBuilder{}
            .setShaders(std::string(SHADER_DIR) + "/grid.vert.spv",
                        std::string(SHADER_DIR) + "/grid.frag.spv")
            .setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
            .setAlphaBlend(true)
            .setDepthWrite(false)
            .setSampleCount(msaaSamples_)
            .build(device_, renderPass_, pipelineLayout_);
    }

    void createFramebuffers() {
        swapChainFramebuffers_.resize(swapChainImageViews_.size());
        for (size_t i = 0; i < swapChainImageViews_.size(); ++i) {
            // Order matches createRenderPass: draw into the multisampled pair,
            // resolve into the swapchain image.
            std::array<VkImageView, 3> attachments = {colorImageView_, depthImageView_,
                                                      swapChainImageViews_[i]};
            VkFramebufferCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass = renderPass_;
            ci.attachmentCount = static_cast<uint32_t>(attachments.size());
            ci.pAttachments = attachments.data();
            ci.width = swapChainExtent_.width;
            ci.height = swapChainExtent_.height;
            ci.layers = 1;
            if (vkCreateFramebuffer(device_, &ci, nullptr, &swapChainFramebuffers_[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer");
            }
        }
    }

    // ---------------------------------------------------------------- commands/buffers
    void createCommandPool() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = *indices.graphics;
        if (vkCreateCommandPool(device_, &ci, nullptr, &commandPool_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool");
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) &&
                (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("failed to find suitable memory type");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer,
                      VkDeviceMemory& memory) {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &ci, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer");
        }

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, buffer, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
        if (vkAllocateMemory(device_, &ai, nullptr, &memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory");
        }
        vkBindBufferMemory(device_, buffer, memory, 0);
    }

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandPool = commandPool_;
        ai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device_, &ai, &cmd);

        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        VkBufferCopy region{0, 0, size};
        vkCmdCopyBuffer(cmd, src, dst, 1, &region);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        vkQueueSubmit(graphicsQueue_, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue_);
        vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    }

    template <typename T>
    void uploadToDeviceLocal(const std::vector<T>& data, VkBufferUsageFlags usage,
                             VkBuffer& buffer, VkDeviceMemory& memory) {
        VkDeviceSize size = sizeof(T) * data.size();

        VkBuffer staging;
        VkDeviceMemory stagingMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     staging, stagingMem);
        void* mapped;
        vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
        std::memcpy(mapped, data.data(), static_cast<size_t>(size));
        vkUnmapMemory(device_, stagingMem);

        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
        copyBuffer(staging, buffer, size);

        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, stagingMem, nullptr);
    }

    void createMeshBuffers() {
        Mesh mesh = GraphMesh::generate(
            [&](float x, float z) { return func_(x, z, 0.0f); },
            0.0f, 0.0f, cfg::DOMAIN_HALF, cfg::LOD_FALLOFF, cfg::MESH_RESOLUTION);
        indexCount_ = static_cast<uint32_t>(mesh.indices.size());

        // Vertex buffer: HOST_VISIBLE so the render thread can memcpy new frames
        // directly without a staging copy. Zero-allocation hot path update.
        VkDeviceSize vbSize = sizeof(Vertex) * mesh.vertices.size();
        createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     vertexBuffer_, vertexBufferMemory_);
        vkMapMemory(device_, vertexBufferMemory_, 0, vbSize, 0, &vertexMapped_);
        std::memcpy(vertexMapped_, mesh.vertices.data(), static_cast<size_t>(vbSize));

        // Prime all ring slots with t=0 so the producer never hands us uninitialized data.
        for (auto& slot : ring_) {
            slot.vertices = mesh.vertices;
        }

        uploadToDeviceLocal(mesh.indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                            indexBuffer_, indexBufferMemory_);
    }

    // ---------------------------------------------------------------- imgui
    void initImGui() {
        // ImGui needs one combined-image-sampler descriptor per texture it
        // binds; the default font atlas is the only one here.
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pi.maxSets = 8;
        pi.poolSizeCount = 1;
        pi.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device_, &pi, nullptr, &imguiPool_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create ImGui descriptor pool");
        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        // true = install GLFW callbacks. ImGui chains to the ones set in
        // initWindow() rather than replacing them.
        ImGui_ImplGlfw_InitForVulkan(window_, true);

        ImGui_ImplVulkan_InitInfo info{};
        info.Instance = instance_;
        info.PhysicalDevice = physicalDevice_;
        info.Device = device_;
        info.QueueFamily = findQueueFamilies(physicalDevice_).graphics.value();
        info.Queue = graphicsQueue_;
        info.DescriptorPool = imguiPool_;
        info.MinImageCount = static_cast<uint32_t>(swapChainImages_.size());
        info.ImageCount = static_cast<uint32_t>(swapChainImages_.size());
        // Render-pass state moved out of the top level into PipelineInfoMain;
        // CMake tracks ImGui master, so this needs the post-move backend.
        info.PipelineInfoMain.RenderPass = renderPass_;
        info.PipelineInfoMain.Subpass = 0;
        info.PipelineInfoMain.MSAASamples = msaaSamples_;
        ImGui_ImplVulkan_Init(&info); // font atlas uploads lazily since 1.90
    }

    void cleanupImGui() {
        if (imguiPool_ == VK_NULL_HANDLE) {
            return;
        }
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(device_, imguiPool_, nullptr);
        imguiPool_ = VK_NULL_HANDLE;
    }

    // Runs once per frame before drawFrame(). Builds the panel and finalizes
    // ImGui's draw lists, which recordCommandBuffer() then submits.
    void buildUI() {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_FirstUseEver);
        ImGui::Begin("Graph");

        ImGui::TextUnformatted("height = f(x, y, t)");
        ImGui::SetNextItemWidth(-90.0f);
        const bool submitted = ImGui::InputText("##expr", inputBuf_, sizeof(inputBuf_),
                                                ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        if (ImGui::Button("Apply", ImVec2(-1, 0)) || submitted) {
            applyExpression();
        }

        if (!parseError_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", parseError_.c_str());
        } else {
            ImGui::TextDisabled("t is seconds, so use it to animate the surface");
        }

        if (ImGui::CollapsingHeader("Examples")) {
            static const char* kPresets[] = {
                "sin(x - t) * cos(y - t)",
                "sin(3x)*cos(3y) + 0.3sin(t)",
                "1 / (1 + x^2 + y^2)",
                "sin(hypot(x, y)*3 - t*2)",
                "exp(-(x^2 + y^2)) * 3",
                "abs(x) - abs(y)",
            };
            for (const char* p : kPresets) {
                if (ImGui::Button(p, ImVec2(-1, 0))) {
                    std::snprintf(inputBuf_, sizeof(inputBuf_), "%s", p);
                    applyExpression();
                }
            }
        }

        ImGui::Separator();
        ImGui::Text("%.0f FPS", static_cast<double>(ImGui::GetIO().Framerate));
        ImGui::SliderFloat("Fly speed", &camera.moveSpeed, 0.5f, 60.0f, "%.1f");
        ImGui::TextDisabled("WASD move | Q/E up-down | Shift sprint | drag to look");

        ImGui::End();
        ImGui::Render();
    }

    void startProducer() {
        producerRunning_.store(true, std::memory_order_relaxed);
        producerThread_ = std::thread([this] {
            using clock = std::chrono::steady_clock;
            using ms = std::chrono::duration<float, std::milli>;
            constexpr float kTargetMs = 16.0f; // ~60 Hz

            // t is wall-clock elapsed rather than a per-iteration increment.
            // Iterations no longer take a fixed time — they are skipped
            // entirely for a still surface — so counting them would make
            // animation speed depend on how expensive the expression is.
            const auto startTime = clock::now();
            bool everBuilt = false;
            uint32_t builtGeneration = 0;
            float builtX = 0.0f, builtZ = 0.0f;

            while (producerRunning_.load(std::memory_order_relaxed)) {
                auto frameStart = clock::now();
                const float t =
                    std::chrono::duration<float>(frameStart - startTime).count();

                // Rebuild around wherever the camera is now. Vertex count never
                // changes, only the region sampled, so the buffer stays valid.
                const float cx = patchCenterX_.load(std::memory_order_relaxed);
                const float cz = patchCenterZ_.load(std::memory_order_relaxed);
                const uint32_t generation = funcGeneration_.load(std::memory_order_acquire);

                // A surface with no t in it, sampled over a patch that has not
                // moved, comes out identical to the one already uploaded. A
                // still graph is the common case, and this makes it free.
                const bool stale = !everBuilt || generation != builtGeneration ||
                                   cx != builtX || cz != builtZ;
                if (!stale && !funcAnimated_.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    continue;
                }
                everBuilt = true;
                builtGeneration = generation;
                builtX = cx;
                builtZ = cz;

                // Copy the callable once per snapshot rather than locking
                // per-vertex: the lock is held for microseconds, and the
                // 100k-vertex sample() below runs entirely unlocked. It also
                // has to, since sample() calls f from several threads at once
                // and they cannot all be waiting on one mutex.
                std::function<float(float, float, float)> f;
                {
                    std::lock_guard<std::mutex> lock(funcMutex_);
                    f = func_;
                }

                // Sampled straight into the slot: it is already the right size
                // from priming, so this reallocates nothing. Only the vertices
                // are rebuilt — the triangulation depends on the resolution
                // alone and was uploaded once.
                uint32_t slot = writeIdx_.load(std::memory_order_relaxed) & (RING_SIZE - 1);
                GraphMesh::sample(
                    [&](float x, float z) { return f(x, z, t); },
                    cx, cz, cfg::DOMAIN_HALF, cfg::LOD_FALLOFF,
                    cfg::MESH_RESOLUTION, ring_[slot].vertices);
                writeIdx_.fetch_add(1, std::memory_order_release);

                // Sleep for the remainder of the 16 ms budget.
                float elapsed = ms(clock::now() - frameStart).count();
                if (elapsed < kTargetMs) {
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(
                            static_cast<int>((kTargetMs - elapsed) * 1000.0f)));
                }
            }
        });
    }

    void stopProducer() {
        producerRunning_.store(false, std::memory_order_relaxed);
        if (producerThread_.joinable()) {
            producerThread_.join();
        }
    }

    // Called once per frame on the render thread — zero allocation, one memcpy.
    void consumeLatestSnapshot() {
        uint32_t w = writeIdx_.load(std::memory_order_acquire);
        if (w == 0 || w == uploadedIdx_) {
            // Nothing new: the buffer already holds this snapshot, and it is a
            // few megabytes to copy over itself every frame.
            return;
        }
        uploadedIdx_ = w;
        const Snapshot& snap = ring_[(w - 1) & (RING_SIZE - 1)];
        std::memcpy(vertexMapped_, snap.vertices.data(),
                    sizeof(Vertex) * snap.vertices.size());
    }

    void createCommandBuffers() {
        commandBuffers_.resize(cfg::MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
        if (vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers");
        }
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                             const PushConstants& pc) {
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin command buffer");
        }

        // One per attachment. The resolve target is not cleared — it is written
        // wholesale by the resolve — but the array is indexed by attachment, so
        // it still needs an entry.
        std::array<VkClearValue, 3> clears{};
        clears[0].color = {{0.04f, 0.05f, 0.08f, 1.0f}};
        clears[1].depthStencil = {1.0f, 0};
        clears[2].color = clears[0].color;

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = renderPass_;
        rp.framebuffer = swapChainFramebuffers_[imageIndex];
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = swapChainExtent_;
        rp.clearValueCount = static_cast<uint32_t>(clears.size());
        rp.pClearValues = clears.data();
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

        VkViewport viewport{};
        viewport.width = static_cast<float>(swapChainExtent_.width);
        viewport.height = static_cast<float>(swapChainExtent_.height);
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{{0, 0}, swapChainExtent_};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = {vertexBuffer_};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);

        vkCmdPushConstants(cmd, pipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(PushConstants), &pc);

        vkCmdDrawIndexed(cmd, indexCount_, 1, 0, 0, 0);

        // Ground plane and Y axis after the surface: both are blended, so they
        // have to be drawn over the opaque geometry they blend against. Twelve
        // vertices, no buffer — grid.vert builds the quad from the first six and
        // the axis ribbon from the rest.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gridPipeline_);
        vkCmdDraw(cmd, 12, 1, 0, 0);

        // Last in the pass, so the panel composites over the surface.
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

        vkCmdEndRenderPass(cmd);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer");
        }
    }

    // ---------------------------------------------------------------- sync
    void createSyncObjects() {
        imageAvailableSemaphores_.resize(cfg::MAX_FRAMES_IN_FLIGHT);
        inFlightFences_.resize(cfg::MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores_.resize(swapChainImages_.size());

        VkSemaphoreCreateInfo sem{};
        sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < cfg::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(device_, &sem, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
                vkCreateFence(device_, &fence, nullptr, &inFlightFences_[i]) != VK_SUCCESS)
                throw std::runtime_error("failed to create per-frame sync objects");
        }
        // One render-finished semaphore per swapchain image avoids reusing a
        // still-in-use semaphore when present and acquire indices diverge.
        for (size_t i = 0; i < swapChainImages_.size(); ++i) {
            if (vkCreateSemaphore(device_, &sem, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create per-image sync objects");
            }
        }
    }

    // ---------------------------------------------------------------- swapchain recreate
    void cleanupSwapChain() {
        vkDestroyImageView(device_, colorImageView_, nullptr);
        vkDestroyImage(device_, colorImage_, nullptr);
        vkFreeMemory(device_, colorImageMemory_, nullptr);

        vkDestroyImageView(device_, depthImageView_, nullptr);
        vkDestroyImage(device_, depthImage_, nullptr);
        vkFreeMemory(device_, depthImageMemory_, nullptr);

        for (auto fb : swapChainFramebuffers_) {
            vkDestroyFramebuffer(device_, fb, nullptr);
        }
        for (auto view : swapChainImageViews_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        vkDestroySwapchainKHR(device_, swapChain_, nullptr);
    }

    void recreateSwapChain() {
        int w = 0, h = 0;
        glfwGetFramebufferSize(window_, &w, &h);
        while (w == 0 || h == 0) { // minimized: wait until restored
            glfwGetFramebufferSize(window_, &w, &h);
            glfwWaitEvents();
        }
        vkDeviceWaitIdle(device_);

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createColorResources();
        createDepthResources();
        createFramebuffers();

        // The image count can change on recreate; keep one render-finished
        // semaphore per swapchain image so imageIndex stays in bounds.
        if (renderFinishedSemaphores_.size() != swapChainImages_.size()) {
            for (auto s : renderFinishedSemaphores_) {
                vkDestroySemaphore(device_, s, nullptr);
            }
            renderFinishedSemaphores_.resize(swapChainImages_.size());
            VkSemaphoreCreateInfo sem{};
            sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            for (auto& s : renderFinishedSemaphores_) {
                if (vkCreateSemaphore(device_, &sem, nullptr, &s) != VK_SUCCESS) {
                    throw std::runtime_error("failed to recreate per-image semaphores");
                }
            }
        }
    }

    // ---------------------------------------------------------------- frame
    PushConstants buildPushConstants() {
        const float aspect = swapChainExtent_.height == 0
                                 ? 1.0f
                                 : static_cast<float>(swapChainExtent_.width) /
                                       static_cast<float>(swapChainExtent_.height);
        glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect,
                                          cfg::NEAR_PLANE, cfg::FAR_PLANE);
        proj[1][1] *= -1.0f; // GLM was built for OpenGL; flip Y for Vulkan clip space
        PushConstants pc{};
        pc.viewProj = proj * camera.view();
        pc.cameraPos = glm::vec4(camera.position, cfg::GRID_EXTENT);
        pc.surface = glm::vec4(patchCenterX_.load(std::memory_order_relaxed),
                               patchCenterZ_.load(std::memory_order_relaxed),
                               cfg::DOMAIN_HALF, 0.0f);
        return pc;
    }

    void drawFrame() {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult acquire = vkAcquireNextImageKHR(
            device_, swapChain_, UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_], VK_NULL_HANDLE, &imageIndex);
        if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image");
        }

        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);

        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex, buildPushConstants());

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[imageIndex]};

        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = waitSemaphores;
        submit.pWaitDstStageMask = waitStages;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffers_[currentFrame_];
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = signalSemaphores;
        if (vkQueueSubmit(graphicsQueue_, 1, &submit, inFlightFences_[currentFrame_]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer");
        }

        VkPresentInfoKHR present{};
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = signalSemaphores;
        present.swapchainCount = 1;
        present.pSwapchains = &swapChain_;
        present.pImageIndices = &imageIndex;

        VkResult res = vkQueuePresentKHR(presentQueue_, &present);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image");
        }

        currentFrame_ = (currentFrame_ + 1) % cfg::MAX_FRAMES_IN_FLIGHT;
    }

    // ---------------------------------------------------------------- teardown
    void cleanup() {
        stopProducer();
        // ImGui's vertex buffers can still be referenced by an in-flight frame,
        // and its pipeline outlives the swapchain but not the render pass.
        vkDeviceWaitIdle(device_);
        cleanupImGui();
        cleanupSwapChain();

        vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
        vkDestroyPipeline(device_, gridPipeline_, nullptr);
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        vkDestroyRenderPass(device_, renderPass_, nullptr);

        vkDestroyBuffer(device_, indexBuffer_, nullptr);
        vkFreeMemory(device_, indexBufferMemory_, nullptr);
        vkUnmapMemory(device_, vertexBufferMemory_);
        vkDestroyBuffer(device_, vertexBuffer_, nullptr);
        vkFreeMemory(device_, vertexBufferMemory_, nullptr);

        for (auto s : renderFinishedSemaphores_) {
            vkDestroySemaphore(device_, s, nullptr);
        }
        for (int i = 0; i < cfg::MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            vkDestroyFence(device_, inFlightFences_[i], nullptr);
        }

        vkDestroyCommandPool(device_, commandPool_, nullptr);
        vkDestroyDevice(device_, nullptr);

        if (cfg::kEnableValidation && debugMessenger_) {
            auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (fn) {
                fn(instance_, debugMessenger_, nullptr);
            }
        }

        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        vkDestroyInstance(instance_, nullptr);

        glfwDestroyWindow(window_);
        glfwTerminate();
    }
};
