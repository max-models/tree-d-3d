// The CUDA version (GPU) of the fmm_m2l kernel; fmm_m2l_pyccel.py next to this file is
// the Pyccel version (CPU), with the same arguments.  Launch with n_threads=count: one
// thread per cell pair, translating both ways and adding to the local expansions
// atomically.  The powers of 1/t live in local memory (inv_re/inv_im are the CPU's
// scratch and unused here), for orders up to TREE_D_3D_MAX_ORDER.

#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void fmm_m2l(
    Array2D<long long> pairs,
    long long count,
    Array2D<double> node_center,
    Array2D<double> moment_re,
    Array2D<double> moment_im,
    Array2D<double> binomial,
    Array1D<double> inv_re,
    Array1D<double> inv_im,
    Array2D<double> local_re,
    Array2D<double> local_im
) {
    const long long pair = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (pair >= count) return;
    const long long order = moment_re.shape[1] - 1;
    double powers_re[2 * TREE_D_3D_MAX_ORDER + 2];
    double powers_im[2 * TREE_D_3D_MAX_ORDER + 2];
    for (int direction = 0; direction < 2; ++direction) {
        const long long src = direction == 0 ? pairs(pair, 0) : pairs(pair, 1);
        const long long dst = direction == 0 ? pairs(pair, 1) : pairs(pair, 0);
        // t = c_dst - c_src; powers of 1/t.
        const double t_re = node_center(dst, 0) - node_center(src, 0);
        const double t_im = node_center(dst, 1) - node_center(src, 1);
        const double t2 = t_re * t_re + t_im * t_im;
        const double w_re = t_re / t2;
        const double w_im = -t_im / t2;
        powers_re[0] = 1.0;
        powers_im[0] = 0.0;
        for (long long k = 1; k < 2 * order + 2; ++k) {
            powers_re[k] = powers_re[k - 1] * w_re - powers_im[k - 1] * w_im;
            powers_im[k] = powers_re[k - 1] * w_im + powers_im[k - 1] * w_re;
        }
        const double a0_re = moment_re(src, 0);
        const double a0_im = moment_im(src, 0);
        // b_0 = a_0 log t - sum_k a_k / (k t**k)
        const double log_re = 0.5 * log(t2);
        const double log_im = atan2(t_im, t_re);
        double b_re = a0_re * log_re - a0_im * log_im;
        double b_im = a0_re * log_im + a0_im * log_re;
        for (long long k = 1; k <= order; ++k) {
            const double ak_re = moment_re(src, k) / k;
            const double ak_im = moment_im(src, k) / k;
            b_re -= ak_re * powers_re[k] - ak_im * powers_im[k];
            b_im -= ak_re * powers_im[k] + ak_im * powers_re[k];
        }
        cunumpy_atomic_add(&local_re(dst, 0), b_re);
        cunumpy_atomic_add(&local_im(dst, 0), b_im);
        // b_l = (-1)**(l+1) a_0 / (l t**l) - (-1)**l / t**l sum_k (a_k/k) C(k+l-1, l) / t**k
        double sign = 1.0;
        for (long long l = 1; l <= order; ++l) {
            sign = -sign;  // (-1)**l
            double s_re = 0.0, s_im = 0.0;
            for (long long k = 1; k <= order; ++k) {
                const double c = binomial(k + l - 1, l) / k;
                const double ak_re = moment_re(src, k) * c;
                const double ak_im = moment_im(src, k) * c;
                s_re += ak_re * powers_re[k] - ak_im * powers_im[k];
                s_im += ak_re * powers_im[k] + ak_im * powers_re[k];
            }
            const double term_re = -sign * a0_re / l - sign * s_re;
            const double term_im = -sign * a0_im / l - sign * s_im;
            cunumpy_atomic_add(&local_re(dst, l), term_re * powers_re[l] - term_im * powers_im[l]);
            cunumpy_atomic_add(&local_im(dst, l), term_re * powers_im[l] + term_im * powers_re[l]);
        }
    }
}
