#pragma once

// =============================================================================
// Turns a height function into renderable geometry.
//
// generate() samples f(x, z) around a centre point and triangulates the result.
// Heights and normals are both numeric, so any function works without
// hand-deriving its gradient — the price is several evaluations per vertex,
// which is why the evaluator on the other side (Expression.hpp) is built for
// throughput rather than flexibility.
//
// Sample spacing is NOT uniform. Cells grow geometrically with distance from the
// centre, which is the same trick terrain renderers use: detail goes where the
// camera is, and the far field costs almost nothing. A fixed vertex budget then
// covers a far larger area — at falloff 50 the same 320x320 grid that used to
// span 30 world units spans 600, with cells running 0.15 near the centre out to
// 7.4 at the rim. Reach improves twentyfold for a 1.6x coarsening up close;
// raising the falloff trades the far field for the near one and wins that back.
//
// Because the spacing varies, so must the sampling. A vertex does not take one
// point sample: it box-filters f over the cell it stands for, and takes its
// normal from the least-squares slope of those same samples. Point-sampling a
// 7.5-unit rim cell of, say, sin(hypot(x,z)*3) — three whole periods inside one
// cell — gives a height and a normal unrelated to each other or to the triangle
// actually drawn, which is what made the far field shimmer. Filtering converges
// it to the local mean instead. Tap count follows cell size, so the near field
// costs about what the old central difference did.
//
// The alternative was a clipmap of concentric rings at halving resolution. This
// is one mesh with one static index buffer instead, so there are no holes to cut
// and no cracks to stitch where levels meet.
//
// Also defines Vertex and Mesh: the vertex format the graph pipeline binds and
// the buffer pair the renderer uploads.
// =============================================================================

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <thread>
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

// Most taps taken across one cell on one axis. Four is where the filter stops
// paying for itself: past that the mean has already settled and the extra
// evaluations only slow the producer thread down.
constexpr int kMaxTaps = 4;

// Taps to spend on a cell, given the finest cell in the patch. A cell twice the
// finest is barely undersampled; one eight times it can hide whole features.
inline int tapsForCell(float cell, float finest) {
    const float ratio = finest > 0.0f ? cell / finest : 1.0f;
    int n = 2;
    while (n < kMaxTaps && ratio >= static_cast<float>(1 << n)) {
        ++n;
    }
    return n;
}

// Centre of the a-th of n equal sub-cells spanning [-radius, radius].
inline float tapOffset(int a, int n, float radius) {
    return radius * (2.0f * (static_cast<float>(a) + 0.5f) / static_cast<float>(n) - 1.0f);
}

// Sample y = f(x, z) over a square of side 2*half centred on (centerX, centerZ),
// with sample density falling off toward the edges.
//
// out is resized, not cleared, so a caller that samples repeatedly at a fixed
// resolution — which the producer thread does — reuses one allocation.
//
// f is called from several threads at once and must tolerate that. Rows are
// independent and write to disjoint spans of out, so nothing else is shared.
inline void sample(const std::function<float(float, float)>& f,
                   float centerX, float centerZ,
                   float half, float falloff,
                   int resolution,
                   std::vector<Vertex>& out) {
    const int N = resolution < 2 ? 2 : resolution;
    out.resize(static_cast<size_t>(N) * N);

    // Both axes share the same warp, so it is worth computing once.
    std::vector<float> offset(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const float u = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(N - 1);
        offset[static_cast<size_t>(i)] = warpAxis(u, half, falloff);
    }

    // Half-width of the cell each sample stands for, and the taps it earns. The
    // two end samples have no outer neighbour to measure against, so it is
    // extrapolated — and those are the rim samples with the widest cells, the
    // ones that most need the filter sized correctly.
    std::vector<float> radius(static_cast<size_t>(N));
    std::vector<int> taps(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        const size_t s = static_cast<size_t>(i);
        const float lo = i > 0 ? offset[s - 1] : 2.0f * offset[0] - offset[1];
        const float hi = i < N - 1 ? offset[s + 1]
                                   : 2.0f * offset[static_cast<size_t>(N - 1)] -
                                         offset[static_cast<size_t>(N - 2)];
        radius[s] = 0.25f * (hi - lo);
    }
    const float finest = centreCellSize(half, falloff, N) * 0.5f;
    for (int i = 0; i < N; ++i) {
        taps[static_cast<size_t>(i)] = tapsForCell(radius[static_cast<size_t>(i)], finest);
    }

    auto buildRows = [&](int rowBegin, int rowEnd) {
    for (int i = rowBegin; i < rowEnd; ++i) {
        const float x = centerX + offset[static_cast<size_t>(i)];
        const float rx = radius[static_cast<size_t>(i)];
        const int nx = taps[static_cast<size_t>(i)];

        for (int j = 0; j < N; ++j) {
            const float z = centerZ + offset[static_cast<size_t>(j)];
            const float rz = radius[static_cast<size_t>(j)];
            const int nz = taps[static_cast<size_t>(j)];

            // One pass over the footprint feeds all three results: the column
            // and row sums give the two slopes, and either one totals to the
            // height. Sampling separately for the gradient would cost as much
            // again and could disagree with the height it was taken for.
            float column[kMaxTaps] = {0.0f};
            float row[kMaxTaps] = {0.0f};
            for (int a = 0; a < nx; ++a) {
                const float dx = tapOffset(a, nx, rx);
                for (int b = 0; b < nz; ++b) {
                    const float v = f(x + dx, z + tapOffset(b, nz, rz));
                    column[a] += v;
                    row[b] += v;
                }
            }

            float total = 0.0f;
            for (int a = 0; a < nx; ++a) {
                total += column[a];
            }
            const float h = total / static_cast<float>(nx * nz);

            // Least-squares slope through the tap means. Unlike a difference of
            // halves this is well defined for an odd tap count, and on a smooth
            // function it recovers the true derivative exactly.
            float sxx = 0.0f, sxh = 0.0f, szz = 0.0f, szh = 0.0f;
            for (int a = 0; a < nx; ++a) {
                const float dx = tapOffset(a, nx, rx);
                sxx += dx * dx;
                sxh += dx * (column[a] / static_cast<float>(nz) - h);
            }
            for (int b = 0; b < nz; ++b) {
                const float dz = tapOffset(b, nz, rz);
                szz += dz * dz;
                szh += dz * (row[b] / static_cast<float>(nx) - h);
            }
            const float dhdx = sxx > 0.0f ? sxh / sxx : 0.0f;
            const float dhdz = szz > 0.0f ? szh / szz : 0.0f;

            // Upward-facing normal of the height field (+y component positive).
            const glm::vec3 n = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));

            out[static_cast<size_t>(i) * N + j] = Vertex{glm::vec3(x, h, z), n};
        }
    }
    };

    // Filtering costs several evaluations per vertex where a central difference
    // cost four, which on one thread is enough to make an animated surface
    // stutter. Rows are independent, so spread them: the work is pure
    // arithmetic over disjoint output, and the caller's thread takes a share
    // rather than idling.
    unsigned hw = std::thread::hardware_concurrency();
    const int workers = static_cast<int>(hw == 0 ? 1u : hw);
    if (workers <= 1) {
        buildRows(0, N);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(workers - 1));
    const int stride = (N + workers - 1) / workers;
    for (int w = 1; w < workers; ++w) {
        const int begin = w * stride;
        if (begin >= N) {
            break;
        }
        pool.emplace_back(buildRows, begin, std::min(begin + stride, N));
    }
    buildRows(0, std::min(stride, N));
    for (auto& th : pool) {
        th.join();
    }
}

// The triangulation, which depends only on the resolution: the sample count
// never changes, only where the samples land. The renderer uploads this once.
inline std::vector<uint32_t> indices(int resolution) {
    const int N = resolution < 2 ? 2 : resolution;
    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(N - 1) * (N - 1) * 6);
    auto idx = [N](int i, int j) { return static_cast<uint32_t>(i * N + j); };
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            out.push_back(idx(i, j));
            out.push_back(idx(i, j + 1));
            out.push_back(idx(i + 1, j));

            out.push_back(idx(i + 1, j));
            out.push_back(idx(i, j + 1));
            out.push_back(idx(i + 1, j + 1));
        }
    }
    return out;
}

// Both halves together, for the one-off that sizes the buffers at startup.
inline Mesh generate(const std::function<float(float, float)>& f,
                     float centerX, float centerZ,
                     float half, float falloff,
                     int resolution) {
    Mesh mesh;
    sample(f, centerX, centerZ, half, falloff, resolution, mesh.vertices);
    mesh.indices = indices(resolution);
    return mesh;
}

} // namespace GraphMesh
