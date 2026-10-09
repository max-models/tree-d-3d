// The CUDA version (GPU) of the quadtree_quadrants kernel; quadtree_quadrants_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of particles: one thread per sorted slot.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void quadtree_quadrants(
    Array2D<double> pos,
    Array1D<long long> perm,
    Array1D<long long> slot_node,
    Array2D<double> node_center,
    Array1D<long long> node_child,
    Array1D<long long> quadrant
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= perm.shape[0]) return;
    const long long node = slot_node(k);
    if (node_child(node) < 0) {
        quadrant(k) = -1;
        return;
    }
    const long long j = perm(k);
    quadrant(k) = (pos(0, j) > node_center(node, 0) ? 1 : 0)
        + (pos(1, j) < node_center(node, 1) ? 2 : 0);
}
