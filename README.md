# 3D Graphing Calculator

A cross-platform (Windows / Linux / macOS) Vulkan renderer that graphs
`height = f(x, y, t)` as a lit, height-colored 3D surface you can fly around.
The expression is editable at runtime — type a new one and the surface changes
without recompiling.

`t` is elapsed seconds, so `sin(x - t)` animates.

## Controls

| Input | Action |
| --- | --- |
| `W` `A` `S` `D` | walk along the ground plane — looking down does not fly you into it |
| `E` / `Space`, `Q` / `Ctrl` | up, down |
| `Shift` | sprint |
| drag left mouse | aim |
| scroll | change fly speed |
| `Esc` | quit |

Type into the **Graph** panel and press Enter (or **Apply**) to change `f`.
Parse errors appear under the input with the column that failed. The
**Examples** section has presets to click through.

You can also pass an expression on the command line:

```sh
./3DGraphingCalculator "sin(hypot(x, y) * 3 - t * 2)"
```

## Project layout

```
vulkan-grapher/
├── CMakeLists.txt                # finds Vulkan, fetches GLFW + GLM + ImGui, compiles shaders
├── src/
│   ├── main.cpp                  # entry point; optional expression on the command line
│   ├── GraphingCalculator3D.hpp  # the app (Vulkan setup, ImGui panel, producer thread)
│   ├── Expression.hpp            # parses f(x,y,t) into a flat postfix program
│   ├── Camera.hpp                # free-fly camera
│   ├── Input.hpp                 # keys and mouse -> camera motion
│   └── GraphMesh.hpp             # samples f(x,y) into a triangle mesh w/ filtered normals
└── shaders/
    ├── graph.vert                # MVP via push constant
    ├── graph.frag                # height colormap + Lambert lighting + distance fog
    ├── grid.vert                 # camera-centred ground quad, no vertex buffer
    └── grid.frag                 # procedural antialiased grid, LOD decades + distance fade
```

## Expression syntax

- **Variables** — `x`, `y`, `t` (seconds)
- **Constants** — `pi`, `tau`, `e`
- **Operators** — `+ - * / ^`, unary minus. `^` is right-associative, and binds
  tighter than unary minus, so `-x^2` is `-(x^2)`
- **Implicit multiplication** — `2x`, `3(x+1)` and `x y` all work
- **Unary functions** — `sin cos tan asin acos atan sinh cosh tanh exp ln log
  log2 log10 sqrt abs floor ceil round sign`
- **Binary functions** — `atan2 min max pow hypot`

## Prerequisites

1. **Vulkan SDK** — install from <https://vulkan.lunarg.com/sdk/home>.
   Provides the headers, loader, the `glslc` shader compiler, validation layers,
   and (on macOS) the MoltenVK driver. Make sure the SDK environment is active
   (the installer sets `VULKAN_SDK`; on Linux `source ~/VulkanSDK/x.y.z/setup-env.sh`).
2. **CMake ≥ 3.16** and a **C++17** compiler (MSVC, Clang, or GCC).
3. **Git** (CMake's FetchContent clones the dependencies for you).

> GLFW, GLM and Dear ImGui are downloaded and built automatically — you do
> **not** need to install them yourself.

## Build & run

### Linux / macOS

```sh
cmake -S . -B build
cmake --build build -j
./build/3DGraphingCalculator
```

### Windows (Visual Studio generator)

```sh
cmake -S . -B build
cmake --build build --config Debug
./build/Debug/3DGraphingCalculator.exe
```

### Windows (MinGW / Ninja)

```sh
cmake -S . -B build -G Ninja
cmake --build build
./build/3DGraphingCalculator.exe
```

## Notes on the implementation

- **Expressions compile to a flat postfix program, not an AST.** The mesh is
  320×320 and each vertex is filtered over its own cell, so a rebuild costs
  around a million evaluations. Walking a contiguous `vector<Instr>` with a small
  stack beats chasing pointers through a tree of virtual `eval()` calls.
- **Sample density and sample *filtering* both follow the cell size.** Cells grow
  geometrically outward, so the rim cell is fifty times the centre one. A vertex
  there box-filters `f` over its whole cell and takes its normal from the
  least-squares slope of the same taps, instead of point-sampling a function that
  may oscillate several times within one cell — which produced a height and a
  normal that agreed with neither each other nor the triangle drawn from them.
- **Mesh generation runs off the render thread, and across all of them.** A
  producer thread rebuilds the surface into a ring buffer and publishes it with a
  release store, so the frame loop never blocks on evaluation; rows within a
  rebuild are independent and split over every core. It also skips the rebuild
  outright when the expression has no `t` in it and the patch has not moved.
- **The ground grid is procedural, not geometry.** Six vertices built from
  `gl_VertexIndex` form a quad centred on the camera, and the lines are computed
  in the fragment shader with `fwidth()` antialiasing. The grid is therefore
  effectively infinite and stays one pixel wide at any distance — uploaded line
  geometry aliases into crawling noise once the lines fall between samples. Three
  decades (1, 10, 100) are drawn, each fading on how many pixels apart its lines
  land, so a decade retires before it crowds below a pixel and washes out to grey.
- **4x MSAA.** Everything renders into a multisampled colour buffer that is
  resolved into the swapchain image at the end of the single subpass. The far
  field is mostly triangles a pixel or two across, which is where the last of
  the aliasing lives once the sampling above is fixed.
- **Push-constant MVP, no descriptor sets.** The only per-frame uniform is one
  `mat4` (`projection * view`), which fits comfortably in the 128-byte
  push-constant guarantee. This avoids the descriptor pool / uniform buffer
  boilerplate entirely.
- **World-space mesh.** The model matrix is identity; the camera moves via the
  view matrix. That keeps vertex normals valid in world space with no normal
  matrix in the shader.
- **macOS / MoltenVK.** The instance enables `VK_KHR_portability_enumeration`
  (and the device enables `VK_KHR_portability_subset` if present), which is
  required for the non-conformant MoltenVK driver.
- **Resize-safe.** The swapchain (and the depth and multisample buffers) are
  recreated on resize and on `VK_ERROR_OUT_OF_DATE_KHR`; the loop pauses while
  minimized.
- **Validation layers** are enabled automatically in Debug builds and compiled
  out in Release (`-DCMAKE_BUILD_TYPE=Release`).

## Possible next steps

- Axis labels and tick marks for a real "calculator" feel.
- Adjustable domain and mesh resolution from the UI.
