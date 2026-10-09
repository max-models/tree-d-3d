// The CUDA version (GPU) of the log_quadtree_moments kernel; log_quadtree_moments_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=node_count: one thread per node.

#include "cunumpy/array_view.cuh"

extern "C" __global__
void log_quadtree_moments(
    Array2D<double> pos,
    Array1D<double> charge,
    Array1D<long long> perm,
    long long node_count,
    Array2D<double> node_center,
    Array1D<long long> node_start,
    Array1D<long long> node_end,
    Array1D<double> node_qabs,
    Array2D<double> moment_re,
    Array2D<double> moment_im
) {
    const long long node = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (node >= node_count) return;
    const long long order = moment_re.shape[1] - 1;
    const double xmid = node_center(node, 0);
    const double ymid = node_center(node, 1);
    double qabs = 0.0;
    for (long long n = 0; n <= order; ++n) {
        moment_re(node, n) = 0.0;
        moment_im(node, n) = 0.0;
    }
    for (long long k = node_start(node); k < node_end(node); ++k) {
        const long long j = perm(k);
        const double q = charge(j);
        qabs += fabs(q);
        const double dx = pos(0, j) - xmid;
        const double dy = pos(1, j) - ymid;
        double power_re = q, power_im = 0.0;
        for (long long n = 0; n <= order; ++n) {
            moment_re(node, n) += power_re;
            moment_im(node, n) += power_im;
            const double next_re = power_re * dx - power_im * dy;
            power_im = power_re * dy + power_im * dx;
            power_re = next_re;
        }
    }
    node_qabs(node) = qabs;
}
