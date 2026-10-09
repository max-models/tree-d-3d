// The CUDA version (GPU) of the evaluate_log_quadtree kernel; evaluate_log_quadtree_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  One thread
// per target, each walking the tree with a stack in local memory (the stack argument
// is the CPU's scratch and unused here); a tree too deep for it reports status -1.

#include "cunumpy/array_view.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void evaluate_log_quadtree(
    Array2D<double> targets,
    Array1D<long long> exclude,
    Array2D<double> pos,
    Array1D<double> charge,
    Array1D<long long> perm,
    Array2D<double> node_center,
    Array1D<double> node_half,
    Array1D<long long> node_start,
    Array1D<long long> node_end,
    Array1D<long long> node_child,
    Array1D<double> node_qabs,
    Array2D<double> moment_re,
    Array2D<double> moment_im,
    double theta,
    double softening,
    Array1D<long long> stack,
    Array1D<double> phi,
    Array2D<double> field,
    Array1D<long long> status
) {
    const long long i = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (i >= targets.shape[1]) return;
    const long long order = moment_re.shape[1] - 1;
    const double eps2 = softening * softening;
    const double tx = targets(0, i);
    const double ty = targets(1, i);
    const long long skip = exclude(i);
    double potential = 0.0, ex = 0.0, ey = 0.0;
    long long cells[TREE_D_3D_MAX_STACK];
    cells[0] = 0;
    int top = 1;
    while (top > 0) {
        const long long node = cells[--top];
        if (node_qabs(node) == 0.0) continue;
        const long long first = node_child(node);
        if (first < 0) {
            for (long long k = node_start(node); k < node_end(node); ++k) {
                const long long j = perm(k);
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
            continue;
        }
        const double xmid = node_center(node, 0);
        const double ymid = node_center(node, 1);
        const double half = node_half(node);
        const double rx = tx - xmid;
        const double ry = ty - ymid;
        const double d2 = rx * rx + ry * ry;
        const bool inside = tx >= xmid - half && tx <= xmid + half && ty >= ymid - half &&
                            ty <= ymid + half;
        if (!inside && 4.0 * half * half < theta * theta * d2) {
            const double w_re = rx / d2;
            const double w_im = -ry / d2;
            // prev = w**n, cur = w**(n+1)
            double prev_re = 1.0, prev_im = 0.0, cur_re = w_re, cur_im = w_im;
            double f_re = 0.0, f_im = 0.0, series = 0.0;
            for (long long n = 0; n <= order; ++n) {
                const double a_re = moment_re(node, n);
                const double a_im = moment_im(node, n);
                f_re += a_re * cur_re - a_im * cur_im;
                f_im += a_re * cur_im + a_im * cur_re;
                if (n > 0) series += (a_re * prev_re - a_im * prev_im) / n;
                prev_re = cur_re;
                prev_im = cur_im;
                const double next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
            potential += -moment_re(node, 0) * log(d2) + 2.0 * series;
            ex += 2.0 * f_re;
            ey -= 2.0 * f_im;
        } else {
            if (top + 4 > TREE_D_3D_MAX_STACK) {
                tree_d_3d_report(&status(0), -1);
                return;
            }
            for (int child = 0; child < 4; ++child) cells[top++] = first + child;
        }
    }
    phi(i) = potential;
    field(0, i) = ex;
    field(1, i) = ey;
}
