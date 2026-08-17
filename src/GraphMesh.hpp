#pragma once

// =============================================================================
// Turns a height function into renderable geometry.
//
// generate() samples f(x, z) around a centre point and triangulates the result.
// Normals come from central differences rather than an analytic derivative, so
// any function works without hand-deriving its gradient — the price is four
// extra evaluations per vertex, which is why the evaluator on the other side
// (Expression.hpp) is built for throughput rather than flexibility.
//
// Sample spacing is NOT uniform. Cells grow geometrically with distance from the
// centre, which is the same trick terrain renderers use: detail goes where the
// camera is, and the far field costs almost nothing. A fixed vertex budget then
// covers a far larger area — at falloff 50 the same 320x320 grid that used to
// span 30 world units spans 600, with cells running 0.15 near the centre out to
// 7.4 at the rim. Reach improves twentyfold for a 1.6x coarsening up close;
// raising the falloff trades the far field for the near one and wins that back.
//
// The alternative was a clipmap of concentric rings at halving resolution. This
// is one mesh with one static index buffer instead, so there are no holes to cut
// and no cracks to stitch where levels meet.
//
// Also defines Vertex and Mesh: the vertex format the graph pipeline binds and
// the buffer pair the renderer uploads.
// =============================================================================

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

namespace GraphMesh {

// Maps a uniform parameter in [-1, 1] onto a warped offset in [-half, half].
// falloff is the ratio between the outermost and innermost cell, so 1 gives an
// ordinary evenly spaced grid.
inline float warpAxis(float u, float half, float falloff) {
    if (falloff <= 1.001f) {
        return u * half;
    }
    const float k = std::log(falloff);
    const float s = u < 0.0f ? -1.0f : 1.0f;
    return s * half * std::expm1(k * std::fabs(u)) / std::expm1(k);
}

// Width of the cell at the very centre of the patch, which is the finest one.
// The caller snaps the patch centre to this so near-field vertices keep landing
// on the same world positions as the camera moves.
inline float centreCellSize(float half, float falloff, int resolution) {
    const int n = resolution < 2 ? 2 : resolution;
    const float du = 2.0f / static_cast<float>(n - 1);
    return warpAxis(du, half, falloff);
}

// Triangulate the surface y = f(x, z) over a square of side 2*half centred on
// (centerX, centerZ), with sample density falling off toward the edges.
inline Mesh generate(const std::function<float(float, float)>& f,
                     float centerX, float centerZ,
                     float half, float falloff,
                     int resolution) {
    Mesh mesh;
    const int N = resolution < 2 ? 2 : resolution;
    mesh.vertices.reserve(static_cast<size_t>(N) * N);

    // Both axes share the same warp, so it is worth computing once.
    std::vector<float> offset(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const float u = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N - 1);
        offset[static_cast<size_t>(i)] = warpAxis(u, half, falloff);
    }

    const float eps = 1e-3f;

    for (int i = 0; i < N; ++i) {
        const float x = centerX + offset[static_cast<size_t>(i)];
        for (int j = 0; j < N; ++j) {
            const float z = centerZ + offset[static_cast<size_t>(j)];
            const float h = f(x, z);

            const float dhdx = (f(x + eps, z) - f(x - eps, z)) / (2.0f * eps);
            const float dhdz = (f(x, z + eps) - f(x, z - eps)) / (2.0f * eps);
            // Upward-facing normal of the height field (+y component positive).
            const glm::vec3 n = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));

            mesh.vertices.push_back(Vertex{glm::vec3(x, h, z), n});
        }
    }

    mesh.indices.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);
    auto idx = [N](int i, int j) { return static_cast<uint32_t>(i * N + j); };
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            mesh.indices.push_back(idx(i, j));
            mesh.indices.push_back(idx(i, j + 1));
            mesh.indices.push_back(idx(i + 1, j));

            mesh.indices.push_back(idx(i + 1, j));
            mesh.indices.push_back(idx(i, j + 1));
            mesh.indices.push_back(idx(i + 1, j + 1));
        }
    }
    return mesh;
}

} // namespace GraphMesh
