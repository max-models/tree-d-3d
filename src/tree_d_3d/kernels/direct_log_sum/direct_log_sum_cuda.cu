// The CUDA version (GPU) of the direct_log_sum kernel; direct_log_sum_pyccel.py next to
// this file is the Pyccel version (CPU), with the same arguments.  One thread per target.

#include "cunumpy/array_view.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void direct_log_sum(
    Array2D<double> targets,
    Array1D<long long> exclude,
    Array2D<double> pos,
    Array1D<double> charge,
    double softening,
    Array1D<double> phi,
    Array2D<double> field,
    Array1D<long long> status
) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i >= targets.shape[1]) return;
    const double eps2 = softening * softening;
    const double tx = targets(0, i);
    const double ty = targets(1, i);
    const long long skip = exclude(i);
    double potential = 0.0, ex = 0.0, ey = 0.0;
    for (long long j = 0; j < pos.shape[1]; ++j) {
        const double c = charge(j);
        if (j == skip || c == 0.0) continue;
        const double dx = tx - pos(0, j);
        const double dy = ty - pos(1, j);
        const double r2 = dx * dx + dy * dy + eps2;
        if (r2 == 0.0) {
            tree_d_3d_report(&status(0), i + 1);
            return;
        }
        potential -= c * log(r2);
        ex += 2.0 * c * dx / r2;
        ey += 2.0 * c * dy / r2;
    }
    phi(i) = potential;
    field(0, i) = ex;
    field(1, i) = ey;
}
