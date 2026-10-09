// The CUDA version (GPU) of the pair_energy kernel; pair_energy_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  One thread per particle i
// sums its pairs with j > i and adds them to energy(0) atomically (the summation order,
// hence the last bits, differ from the CPU).

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void pair_energy(
    Array2D<double> pos,
    Array1D<double> charge,
    double softening,
    Array1D<long long> status,
    Array1D<double> energy
) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    const long long n = pos.shape[1];
    if (i >= n) return;
    const double ci = charge(i);
    if (ci == 0.0) return;
    const double eps2 = softening * softening;
    double partial = 0.0;
    for (long long j = i + 1; j < n; ++j) {
        const double cj = charge(j);
        if (cj == 0.0) continue;
        const double dx = pos(0, i) - pos(0, j);
        const double dy = pos(1, i) - pos(1, j);
        const double r2 = dx * dx + dy * dy + eps2;
        if (r2 == 0.0) {
            tree_d_3d_report(&status(0), 1);
            return;
        }
        partial += cj / sqrt(r2);
    }
    cunumpy_atomic_add(&energy(0), ci * partial);
}
