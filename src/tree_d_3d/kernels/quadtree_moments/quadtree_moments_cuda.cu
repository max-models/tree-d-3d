// The CUDA version (GPU) of the quadtree_moments kernel; quadtree_moments_pyccel.py next
// to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=node_count: one thread per node, each summing its own particles.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void quadtree_moments(
    Array2D<double> pos,
    Array1D<double> charge,
    Array1D<long long> perm,
    long long node_count,
    Array2D<double> node_center,
    Array1D<long long> node_start,
    Array1D<long long> node_end,
    Array1D<double> node_q,
    Array1D<double> node_qabs,
    Array2D<double> node_p,
    Array2D<double> node_s
) {
    const long long node = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (node >= node_count) return;
    double q = 0.0, qabs = 0.0, px = 0.0, py = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
    const double xmid = node_center(node, 0);
    const double ymid = node_center(node, 1);
    for (long long k = node_start(node); k < node_end(node); ++k) {
        const long long j = perm(k);
        const double c = charge(j);
        const double dx = pos(0, j) - xmid;
        const double dy = pos(1, j) - ymid;
        q += c;
        qabs += fabs(c);
        px += c * dx;
        py += c * dy;
        sxx += c * dx * dx;
        sxy += c * dx * dy;
        syy += c * dy * dy;
    }
    node_q(node) = q;
    node_qabs(node) = qabs;
    node_p(node, 0) = px;
    node_p(node, 1) = py;
    node_s(node, 0) = sxx;
    node_s(node, 1) = sxy;
    node_s(node, 2) = syy;
}
