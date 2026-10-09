// The CUDA version (GPU) of the fmm_evaluate kernel; fmm_evaluate_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Launch with
// n_threads=max(node_count, count): thread t evaluates the local expansion of leaf t at
// its particles (far field) and the exact interactions of leaf pair t (near field),
// adding to phi and field atomically.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void fmm_evaluate(
    Array2D<double> pos,
    Array1D<double> charge,
    Array1D<long long> perm,
    long long node_count,
    Array2D<double> node_center,
    Array1D<long long> node_start,
    Array1D<long long> node_end,
    Array1D<long long> node_child,
    Array2D<double> local_re,
    Array2D<double> local_im,
    Array2D<long long> pairs,
    long long count,
    double softening,
    long long with_potential,
    Array1D<double> phi,
    Array2D<double> field,
    Array1D<long long> status
) {
    const long long t = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    const long long order = local_re.shape[1] - 1;
    const double eps2 = softening * softening;
    if (t < node_count && node_child(t) < 0) {
        const long long node = t;
        for (long long k = node_start(node); k < node_end(node); ++k) {
            const long long i = perm(k);
            const double u_re = pos(0, i) - node_center(node, 0);
            const double u_im = pos(1, i) - node_center(node, 1);
            // Horner for W = sum b_l u**l and W' = sum l b_l u**(l-1).
            double w_re = local_re(node, order), w_im = local_im(node, order);
            double d_re = 0.0, d_im = 0.0;
            for (long long l = order - 1; l >= 0; --l) {
                const double nd_re = d_re * u_re - d_im * u_im + w_re;
                d_im = d_re * u_im + d_im * u_re + w_im;
                d_re = nd_re;
                const double nw_re = w_re * u_re - w_im * u_im + local_re(node, l);
                w_im = w_re * u_im + w_im * u_re + local_im(node, l);
                w_re = nw_re;
            }
            cunumpy_atomic_add(&phi(i), -2.0 * w_re);
            cunumpy_atomic_add(&field(0, i), 2.0 * d_re);
            cunumpy_atomic_add(&field(1, i), -2.0 * d_im);
        }
    }
    if (t >= count) return;
    const long long a = pairs(t, 0);
    const long long b = pairs(t, 1);
    for (long long ka = node_start(a); ka < node_end(a); ++ka) {
        const long long i = perm(ka);
        const long long start_b = a == b ? ka + 1 : node_start(b);
        for (long long kb = start_b; kb < node_end(b); ++kb) {
            const long long j = perm(kb);
            const double dx = pos(0, i) - pos(0, j);
            const double dy = pos(1, i) - pos(1, j);
            const double r2 = dx * dx + dy * dy + eps2;
            if (r2 == 0.0) {
                if (charge(i) != 0.0 && charge(j) != 0.0) {
                    tree_d_3d_report(&status(0), i + 1);
                    return;
                }
                continue;
            }
            if (with_potential != 0) {
                const double log_r2 = log(r2);
                cunumpy_atomic_add(&phi(i), -charge(j) * log_r2);
                cunumpy_atomic_add(&phi(j), -charge(i) * log_r2);
            }
            const double fx = 2.0 * dx / r2;
            const double fy = 2.0 * dy / r2;
            cunumpy_atomic_add(&field(0, i), charge(j) * fx);
            cunumpy_atomic_add(&field(1, i), charge(j) * fy);
            cunumpy_atomic_add(&field(0, j), -charge(i) * fx);
            cunumpy_atomic_add(&field(1, j), -charge(i) * fy);
        }
    }
}
