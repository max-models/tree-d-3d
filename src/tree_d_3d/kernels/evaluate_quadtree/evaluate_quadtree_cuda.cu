// The CUDA version (GPU) of the evaluate_quadtree kernel; evaluate_quadtree_pyccel.py
// next to this file is the Pyccel version (CPU), with the same arguments.  One thread
// per target, each walking the tree with a stack in local memory (the stack argument
// is the CPU's scratch and unused here); a tree too deep for it reports status -1.

#include "cunumpy/array_view.cuh"
#include "tree_d_3d/kernels/barnes_hut.cuh"

extern "C" __global__
void evaluate_quadtree(
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
    Array1D<double> node_q,
    Array1D<double> node_qabs,
    Array2D<double> node_p,
    Array2D<double> node_s,
    double theta,
    double softening,
    Array1D<long long> stack,
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
                const double inv_r = 1.0 / sqrt(r2);
                const double inv_r3 = inv_r * inv_r * inv_r;
                potential += c * inv_r;
                ex += c * inv_r3 * dx;
                ey += c * inv_r3 * dy;
            }
            continue;
        }
        const double xmid = node_center(node, 0);
        const double ymid = node_center(node, 1);
        const double half = node_half(node);
        const double rx = tx - xmid;
        const double ry = ty - ymid;
        const double distance = sqrt(rx * rx + ry * ry);
        const bool inside = tx >= xmid - half && tx <= xmid + half && ty >= ymid - half &&
                            ty <= ymid + half;
        if (!inside && 2.0 * half < theta * distance) {
            const double inv_r = 1.0 / sqrt(distance * distance + eps2);
            const double inv_r2 = inv_r * inv_r;
            const double inv_r3 = inv_r2 * inv_r;
            const double inv_r5 = inv_r3 * inv_r2;
            const double inv_r7 = inv_r5 * inv_r2;
            const double q = node_q(node);
            const double px = node_p(node, 0);
            const double py = node_p(node, 1);
            const double sxx = node_s(node, 0);
            const double sxy = node_s(node, 1);
            const double syy = node_s(node, 2);
            const double pr = px * rx + py * ry;
            const double srx = sxx * rx + sxy * ry;
            const double sry = sxy * rx + syy * ry;
            const double rsr = rx * srx + ry * sry;
            const double trace = sxx + syy;
            potential += q * inv_r + pr * inv_r3 + 0.5 * (3.0 * rsr * inv_r5 - trace * inv_r3);
            const double radial =
                q * inv_r3 + 3.0 * pr * inv_r5 + 7.5 * rsr * inv_r7 - 1.5 * trace * inv_r5;
            ex += radial * rx - px * inv_r3 - 3.0 * srx * inv_r5;
            ey += radial * ry - py * inv_r3 - 3.0 * sry * inv_r5;
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
