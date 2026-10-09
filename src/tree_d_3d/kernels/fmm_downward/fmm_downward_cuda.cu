// The CUDA version (GPU) of the fmm_downward kernel; fmm_downward_pyccel.py next to this
// file is the Pyccel version (CPU), with the same arguments.  Launch once per tree level
// with n_threads=last_node - first_node: one thread per parent, which shifts its local
// expansion into its four children.  The powers of the shift live in local memory
// (power_re/power_im are the CPU's scratch and unused here).

#include "cunumpy/array_view.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void fmm_downward(
    long long first_node,
    long long last_node,
    Array2D<double> node_center,
    Array1D<long long> node_child,
    Array2D<double> binomial,
    Array1D<double> power_re,
    Array1D<double> power_im,
    Array2D<double> local_re,
    Array2D<double> local_im
) {
    const long long node = first_node + blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (node >= last_node) return;
    const long long first = node_child(node);
    if (first < 0) return;
    const long long order = local_re.shape[1] - 1;
    double s_pow_re[TREE_D_3D_MAX_ORDER + 1];
    double s_pow_im[TREE_D_3D_MAX_ORDER + 1];
    for (long long child = first; child < first + 4; ++child) {
        const double s_re = node_center(child, 0) - node_center(node, 0);
        const double s_im = node_center(child, 1) - node_center(node, 1);
        s_pow_re[0] = 1.0;
        s_pow_im[0] = 0.0;
        for (long long k = 1; k <= order; ++k) {
            s_pow_re[k] = s_pow_re[k - 1] * s_re - s_pow_im[k - 1] * s_im;
            s_pow_im[k] = s_pow_re[k - 1] * s_im + s_pow_im[k - 1] * s_re;
        }
        for (long long m = 0; m <= order; ++m) {
            double acc_re = 0.0, acc_im = 0.0;
            for (long long l = m; l <= order; ++l) {
                const double c = binomial(l, m);
                const double b_re = local_re(node, l) * c;
                const double b_im = local_im(node, l) * c;
                acc_re += b_re * s_pow_re[l - m] - b_im * s_pow_im[l - m];
                acc_im += b_re * s_pow_im[l - m] + b_im * s_pow_re[l - m];
            }
            local_re(child, m) += acc_re;
            local_im(child, m) += acc_im;
        }
    }
}
