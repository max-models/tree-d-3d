// The CUDA version (GPU) of the soften_pairs kernel; soften_pairs_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=number of pairs: one thread per pair, adding to both partners atomically.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void soften_pairs(
    Array2D<double> pos,
    Array1D<double> charge,
    Array2D<long long> pairs,
    double eps2,
    long long with_potential,
    Array1D<double> phi,
    Array2D<double> field,
    Array1D<long long> status
) {
    const long long k = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (k >= pairs.shape[0]) return;
    const long long i = pairs(k, 0);
    const long long j = pairs(k, 1);
    const double dx = pos(0, i) - pos(0, j);
    const double dy = pos(1, i) - pos(1, j);
    const double r2 = dx * dx + dy * dy;
    if (r2 == 0.0) {
        tree_d_3d_report(&status(0), k + 1);
        return;
    }
    const double soft = r2 + eps2;
    if (with_potential != 0) {
        const double dphi = -log(soft / r2);
        cunumpy_atomic_add(&phi(i), charge(j) * dphi);
        cunumpy_atomic_add(&phi(j), charge(i) * dphi);
    }
    const double scale = 2.0 / soft - 2.0 / r2;
    cunumpy_atomic_add(&field(0, i), charge(j) * scale * dx);
    cunumpy_atomic_add(&field(1, i), charge(j) * scale * dy);
    cunumpy_atomic_add(&field(0, j), -charge(i) * scale * dx);
    cunumpy_atomic_add(&field(1, j), -charge(i) * scale * dy);
}
