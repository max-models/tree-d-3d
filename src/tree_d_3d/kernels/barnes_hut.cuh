// Shared by the CUDA versions of the Barnes-Hut and FMM kernels
// (tree_d_3d/kernels/<name>/<name>_cuda.cu).

#pragma once

// Per-thread traversal stack of the tree evaluations: a target pushes at most
// three cells per level beyond the four children of the root, so 512 entries hold
// trees up to depth 168.  A deeper tree reports status -1.
#define TREE_D_3D_MAX_STACK 512

// Highest multipole order of the FMM kernels' per-thread scratch arrays
// (FlatQuadtree checks the order before it calls them).
#define TREE_D_3D_MAX_ORDER 64

// Report a failure in status(0): the first thread to report wins.  The CPU
// version reports the first failing index; on the GPU it is any failing one.
__device__ __forceinline__ void tree_d_3d_report(long long* status, long long value) {
    atomicCAS(reinterpret_cast<unsigned long long*>(status), 0ULL,
              static_cast<unsigned long long>(value));
}
