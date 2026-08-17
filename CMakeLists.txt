cmake_minimum_required(VERSION 3.16)
project(graphing_calculator_3d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
# Vulkan SDK must be installed (provides headers, the loader, glslc, and on
# macOS the MoltenVK ICD). GLFW and GLM are fetched automatically.
find_package(Vulkan REQUIRED)

include(FetchContent)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# Dear ImGui ships no CMakeLists, so MakeAvailable only downloads it and we
# compile the core plus the two backends ourselves.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# ----------------------------------------------------------------------------
# Shader compilation (GLSL -> SPIR-V)
# ----------------------------------------------------------------------------
if(NOT Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE glslc)
endif()
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK (it ships glslc).")
endif()

set(SHADER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/graph.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/graph.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/grid.frag)

set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SPV_FILES)
foreach(shader ${SHADER_SRC})
    get_filename_component(fname ${shader} NAME)
    set(spv ${SHADER_OUT_DIR}/${fname}.spv)
    add_custom_command(
        OUTPUT  ${spv}
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} ${shader} -o ${spv}
        DEPENDS ${shader}
        COMMENT "Compiling shader ${fname}")
    list(APPEND SPV_FILES ${spv})
endforeach()
add_custom_target(shaders DEPENDS ${SPV_FILES})

# ----------------------------------------------------------------------------
# Executable
# ----------------------------------------------------------------------------
add_executable(GraphingCalculator3D src/main.cpp)
add_dependencies(GraphingCalculator3D shaders)

# A CMake target whose name starts with a digit is not portable across all
# generators, so the target keeps a valid name and only the binary is renamed.
set_target_properties(GraphingCalculator3D PROPERTIES
                      OUTPUT_NAME "3DGraphingCalculator")

target_include_directories(GraphingCalculator3D PRIVATE src)
target_link_libraries(GraphingCalculator3D PRIVATE Vulkan::Vulkan glfw glm::glm imgui)

# Absolute path to compiled shaders so the app finds them regardless of CWD.
target_compile_definitions(GraphingCalculator3D PRIVATE SHADER_DIR="${SHADER_OUT_DIR}")
