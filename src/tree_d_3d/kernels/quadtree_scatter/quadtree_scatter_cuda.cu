// The CUDA version (GPU) of the quadtree_scatter kernel; quadtree_scatter_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of particles: one thread per sorted slot, each writing its own
// target slot (the targets are a permutation).

#include "cunumpy/array_view.cuh"

extern "C" __global__
void quadtree_scatter(
    Array1D<long long> perm,
    Array1D<long long> slot_node,
    Array1D<long long> quadrant,
    Array2D<long long> below,
    Array1D<long long> node_start,
    Array1D<long long> node_child,
    Array1D<long long> new_perm,
    Array1D<long long> new_slot_node
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= perm.shape[0]) return;
    const long long q = quadrant(k);
    const long long node = slot_node(k);
    if (q < 0) {
        new_perm(k) = perm(k);
        new_slot_node(k) = node;
        return;
    }
    const long long start = node_start(node);
    const long long before = start > 0 ? below(start - 1, q) : 0;
    const long long child = node_child(node) + q;
    const long long slot = node_start(child) + below(k, q) - 1 - before;
    new_perm(slot) = perm(k);
    new_slot_node(slot) = child;
}
