// The CUDA version (GPU) of the fmm_traverse kernel; fmm_traverse_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Launch once per traversal
// generation with n_threads=n_frontier: one thread per node pair, which reserves its
// list entries with an atomic add on counts (the order of the lists then differs from
// the CPU's, which only changes the order in which M2L contributions are summed).

#include "cunumpy/array_view.cuh"

__device__ __forceinline__ long long tree_d_3d_reserve(long long* count, long long n) {
    return static_cast<long long>(atomicAdd(
        reinterpret_cast<unsigned long long*>(count), static_cast<unsigned long long>(n)));
}

extern "C" __global__
void fmm_traverse(
    Array2D<double> node_center,
    Array1D<double> node_half,
    Array1D<long long> node_start,
    Array1D<long long> node_end,
    Array1D<long long> node_child,
    double theta,
    Array2D<long long> frontier,
    long long n_frontier,
    Array2D<long long> next_frontier,
    Array2D<long long> m2l,
    Array2D<long long> p2p,
    Array1D<long long> counts
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= n_frontier) return;
    const long long a = frontier(k, 0);
    const long long b = frontier(k, 1);
    if (node_end(a) == node_start(a) || node_end(b) == node_start(b)) return;
    const bool leaf_a = node_child(a) < 0;
    const bool leaf_b = node_child(b) < 0;
    if (a == b) {
        if (leaf_a) {
            const long long i = tree_d_3d_reserve(&counts(1), 1);
            if (i < p2p.shape[0]) {
                p2p(i, 0) = a;
                p2p(i, 1) = a;
            } else {
                counts(3) = -1;
            }
            return;
        }
        const long long first = node_child(a);
        long long i = tree_d_3d_reserve(&counts(2), 10);
        if (i + 10 > next_frontier.shape[0]) {
            counts(3) = -1;
            return;
        }
        for (int c = 0; c < 4; ++c) {
            for (int d = c; d < 4; ++d) {
                next_frontier(i, 0) = first + c;
                next_frontier(i, 1) = first + d;
                ++i;
            }
        }
        return;
    }
    const double dx = node_center(a, 0) - node_center(b, 0);
    const double dy = node_center(a, 1) - node_center(b, 1);
    const double distance = sqrt(dx * dx + dy * dy);
    const double radii = sqrt(2.0) * (node_half(a) + node_half(b));
    if (radii < theta * distance) {
        const long long i = tree_d_3d_reserve(&counts(0), 1);
        if (i < m2l.shape[0]) {
            m2l(i, 0) = a;
            m2l(i, 1) = b;
        } else {
            counts(3) = -1;
        }
        return;
    }
    if (leaf_a && leaf_b) {
        const long long i = tree_d_3d_reserve(&counts(1), 1);
        if (i < p2p.shape[0]) {
            p2p(i, 0) = a;
            p2p(i, 1) = b;
        } else {
            counts(3) = -1;
        }
        return;
    }
    const long long i = tree_d_3d_reserve(&counts(2), 4);
    if (i + 4 > next_frontier.shape[0]) {
        counts(3) = -1;
        return;
    }
    const bool split_a = leaf_b || (!leaf_a && node_half(a) >= node_half(b));
    const long long first = split_a ? node_child(a) : node_child(b);
    for (int c = 0; c < 4; ++c) {
        next_frontier(i + c, 0) = split_a ? first + c : a;
        next_frontier(i + c, 1) = split_a ? b : first + c;
    }
}
