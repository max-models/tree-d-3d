"""A line-charge FMM with smooth cell membership.

Forces are the exact gradient of a smooth potential energy, and mutual pair forces
are exactly antisymmetric.

This is a NumPy prototype for method research, not a fast code: everything is
complex arithmetic on small arrays, and it is meant for a few thousand particles.
It separates two things the production tree ties together:

* a :class:`BoxTree`: a static set of square cells (split until a cell holds at most
  ``leaf_capacity`` of the positions it was built from). The cells can be kept across
  time steps, so the interaction lists depend on nothing that moves;
* a :class:`SmoothFMM` over that tree for given positions and charges.

**Membership.** A particle belongs to every leaf whose box, grown by ``band``, contains
it, with a weight ``w_L(z)`` that is the tensor product of quintic smoothsteps across
the leaf's edges. The weights form a partition of unity on any quadtree leaf
partition when ``band`` is at most half the smallest leaf width. The interaction of
particles ``i`` and ``j`` is split over all their membership pairs (routes)
``(L, L')`` with weight ``w_L(z_i) w_L'(z_j)``; each route uses the exact kernel if
``(L, L')`` is a near-field pair and the symmetric multipole-to-local expansion of the
accepted ancestor pair otherwise. ``band = 0`` is the ordinary tree (hard membership).
Softening is compact (a smooth charge cloud of radius ``softening``), so the far field of
a softened particle is exactly the point charge's and the tree solves the same model as the direct sum.

**Energy and forces.** In the units of :mod:`tree_d_3d.flat_tree` (potential of a line
charge ``-2 q ln r``, field ``2 q d / r^2``, no ``k_e``),

    U = -2 sum_routes w_m w_m' q_i q_j Phi(z_i - z_j),
    F_i = -dU/dx_i = 2 q_i sum_{m: i in m} [ w_m G_m + Phi_m grad w_m ],

where ``Phi_m`` and ``G_m`` are the potential and its gradient at ``z_i`` seen through
membership ``m`` (local expansion of its leaf plus its near-field partners). The
second term, potential times the gradient of the weight, is what makes ``F`` the
exact gradient; without it (or with ``band = 0``) the energy is not conserved when a
particle changes leaf.
"""

from __future__ import annotations

from math import comb, sqrt

import numpy as np

from tree_d_3d.quad import Quad


def _disk_kernel(dz: np.ndarray, eps2: float) -> tuple[np.ndarray, np.ndarray]:
    """Pair potential and gradient of a compact, smooth charge cloud of radius ``a``.

    The cloud has density ``(1 - r^2/a^2)^2`` inside ``r < a`` (``a^2 = eps2``), so
    with ``u = r^2/a^2`` the enclosed charge fraction is ``1 - (1 - u)^3`` and
    ``Phi = log a - 11/12 + (3u - 3u^2/2 + u^3/3) / 2`` inside, ``log r`` outside:
    the far field is exactly the point charge's and the field is C^2 at the rim.
    The gradient ``dPhi/dz_i`` is returned as ``G_x + i G_y``.
    """
    r2 = np.abs(dz) ** 2
    if eps2 <= 0.0:
        return 0.5 * np.log(r2), dz / r2
    inside = r2 < eps2
    safe = np.where(inside, eps2, r2)
    u = np.where(inside, r2 / eps2, 1.0)
    phi_in = (
        0.5 * np.log(eps2) - 11.0 / 12.0 + (3.0 * u - 1.5 * u * u + u**3 / 3.0) / 2.0
    )
    phi = np.where(inside, phi_in, 0.5 * np.log(safe))
    grad = np.where(inside, dz * (3.0 - 3.0 * u + u * u) / eps2, dz / safe)
    return phi, grad


def _smoothstep(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quintic C^2 step on [-1, 1] with s(t) + s(-t) = 1; returns (s, ds/dt)."""
    u = np.clip((t + 1.0) / 2.0, 0.0, 1.0)
    s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    ds = 30.0 * u * u * (1.0 - u) ** 2  # ds/du; dt = 2 du
    return s, 0.5 * ds


class BoxTree:
    """Square cells of a quadtree over ``root``, split to hold ``leaf_capacity`` points."""

    def __init__(self, pos: np.ndarray, root: Quad, *, leaf_capacity: int = 16) -> None:
        z = pos[0] + 1j * pos[1]
        center = [complex(root.xmid, root.ymid)]
        half = [float(root.half)]
        parent = [-1]
        child = [-1]
        depth = [0]
        members = [np.arange(len(z))]
        node = 0
        while node < len(center):
            idx = members[node]
            if len(idx) > leaf_capacity and depth[node] < 40:
                first = len(center)
                child[node] = first
                c, h = center[node], 0.5 * half[node]
                for sx, sy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):
                    cc = c + h * (sx + 1j * sy)
                    center.append(cc)
                    half.append(h)
                    parent.append(node)
                    child.append(-1)
                    depth.append(depth[node] + 1)
                    zz = z[idx]
                    right = np.real(zz) > np.real(c)
                    top = np.imag(zz) > np.imag(c)
                    members.append(idx[(right == (sx > 0)) & (top == (sy > 0))])
            node += 1
        self.center = np.array(center)
        self.half = np.array(half)
        self.parent = np.array(parent)
        self.child = np.array(child)
        self.depth = np.array(depth)
        self.root = root
        self.leaves = np.flatnonzero(self.child < 0)
        self.min_leaf_half = float(self.half[self.leaves].min())

    def memberships(self, z: np.ndarray, band: float):
        """(particle, leaf, weight, dw/dx, dw/dy) for every particle-leaf pair with w > 0."""
        n = len(z)
        if band <= 0.0:
            leaf = np.zeros(n, dtype=np.int64)
            node = np.zeros(n, dtype=np.int64)
            while True:
                first = self.child[node]
                inner = first >= 0
                if not inner.any():
                    break
                c = self.center[node[inner]]
                zz = z[inner]
                q = (np.real(zz) > np.real(c)).astype(int) + 2 * (
                    np.imag(zz) <= np.imag(c)
                ).astype(int)
                node = node.copy()
                node[inner] = first[inner] + q
            leaf[:] = node
            ones = np.ones(n)
            zeros = np.zeros(n)
            return np.arange(n), leaf, ones, zeros, zeros
        if band > self.min_leaf_half:
            raise ValueError("band must not exceed the smallest leaf half-width")
        part, leaf = [], []
        # descend keeping every child whose box grown by ``band`` contains the point
        frontier_p = np.arange(n)
        frontier_n = np.zeros(n, dtype=np.int64)
        while len(frontier_p):
            first = self.child[frontier_n]
            is_leaf = first < 0
            part.append(frontier_p[is_leaf])
            leaf.append(frontier_n[is_leaf])
            fp, fn = frontier_p[~is_leaf], first[~is_leaf]
            next_p, next_n = [], []
            for q in range(4):
                cand = fn + q
                c, h = self.center[cand], self.half[cand] + band
                zz = z[fp]
                inside = (np.abs(np.real(zz) - np.real(c)) < h) & (
                    np.abs(np.imag(zz) - np.imag(c)) < h
                )
                next_p.append(fp[inside])
                next_n.append(cand[inside])
            frontier_p = np.concatenate(next_p)
            frontier_n = np.concatenate(next_n)
        part = np.concatenate(part)
        leaf = np.concatenate(leaf)
        c, h = self.center[leaf], self.half[leaf]
        zz = z[part]
        sxl, dxl = _smoothstep((np.real(zz) - (np.real(c) - h)) / band)
        sxr, dxr = _smoothstep(((np.real(c) + h) - np.real(zz)) / band)
        syl, dyl = _smoothstep((np.imag(zz) - (np.imag(c) - h)) / band)
        syr, dyr = _smoothstep(((np.imag(c) + h) - np.imag(zz)) / band)
        # no blending across the root boundary (there is no neighbour there): a leaf
        # edge on the root edge keeps weight 1, so the partition of unity holds up to
        # the boundary of the root box
        r = self.root
        for s_, ds_, on_root in (
            (sxl, dxl, np.real(c) - h <= r.xmid - r.half + 1e-12 * r.half),
            (sxr, dxr, np.real(c) + h >= r.xmid + r.half - 1e-12 * r.half),
            (syl, dyl, np.imag(c) - h <= r.ymid - r.half + 1e-12 * r.half),
            (syr, dyr, np.imag(c) + h >= r.ymid + r.half - 1e-12 * r.half),
        ):
            s_[on_root] = 1.0
            ds_[on_root] = 0.0
        wx, wy = sxl * sxr, syl * syr
        w = wx * wy
        dwx = (dxl * sxr - sxl * dxr) / band * wy
        dwy = (dyl * syr - syl * dyr) / band * wx
        keep = w > 0.0
        return part[keep], leaf[keep], w[keep], dwx[keep], dwy[keep]


class SmoothFMM:
    """Potential energy and exact-gradient forces of line charges on a :class:`BoxTree`."""

    # -- structure -------------------------------------------------------------
    def _traverse(self, occupied: np.ndarray):
        """Dual-tree traversal over occupied nodes: lists of M2L and P2P node pairs."""
        t = self.tree
        m2l, p2p = [], []
        stack = [(0, 0)]
        root2 = sqrt(2.0)
        while stack:
            a, b = stack.pop()
            if not (occupied[a] and occupied[b]):
                continue
            leaf_a, leaf_b = t.child[a] < 0, t.child[b] < 0
            if a == b:
                if leaf_a:
                    p2p.append((a, a))
                else:
                    f = t.child[a]
                    stack += [(f + c, f + d) for c in range(4) for d in range(c, 4)]
                continue
            d = abs(t.center[a] - t.center[b])
            if root2 * (t.half[a] + t.half[b]) < self.theta * d:
                m2l.append((a, b))
            elif leaf_a and leaf_b:
                p2p.append((a, b))
            elif leaf_b or (not leaf_a and t.half[a] >= t.half[b]):
                f = t.child[a]
                stack += [(f + c, b) for c in range(4)]
            else:
                f = t.child[b]
                stack += [(a, f + c) for c in range(4)]
        return np.array(m2l, dtype=np.int64).reshape(-1, 2), np.array(
            p2p, dtype=np.int64
        ).reshape(-1, 2)

    # -- the computation -------------------------------------------------------
    def _compute(self) -> None:
        t, p = self.tree, self.order
        z, q = self.z, self.q
        part, leaf, w, dwx, dwy = t.memberships(z, self.band)
        self.part, self.leaf, self.w = part, leaf, w
        self.dw = dwx + 1j * dwy  # as a complex number, for convenience only
        n_nodes = len(t.center)
        # multipoles of every node from its leaves' weighted members
        moments = np.zeros((n_nodes, p + 1), dtype=complex)
        occupied = np.zeros(n_nodes, dtype=bool)
        node = leaf.copy()
        while True:
            alive = node >= 0
            if not alive.any():
                break
            nd, pw = node[alive], (q[part] * w)[alive]
            u = z[part][alive] - t.center[nd]
            powers = np.ones_like(u)
            for k in range(p + 1):
                np.add.at(moments[:, k], nd, pw * powers)
                powers = powers * u
            occupied[nd] = True
            node = np.where(alive, t.parent[np.maximum(node, 0)], -1)
        self.moments = moments
        m2l, p2p = self._traverse(occupied)
        self.m2l_pairs, self.p2p_pairs = m2l, p2p
        # M2L, both directions, rectangle or triangle truncation
        local = np.zeros((n_nodes, p + 1), dtype=complex)
        if len(m2l):
            binom = np.array(
                [[comb(n, k) for k in range(2 * p + 2)] for n in range(2 * p + 2)],
                dtype=float,
            )
            ks = np.arange(1, p + 1)
            ls = np.arange(1, p + 1)
            # coefficient matrix C[ell-1, k-1] = (-1)^ell C(k+ell-1, ell) / k, masked for the triangle
            coef = np.array(
                [
                    [(-1.0) ** ell * binom[k + ell - 1, ell] / k for k in ks]
                    for ell in ls
                ]
            )
            if self.symmetric:
                coef *= ks[None, :] + ls[:, None] <= p
            for src, dst in np.concatenate((m2l, m2l[:, ::-1])):
                tt = t.center[dst] - t.center[src]
                inv = 1.0 / tt
                a = moments[src]
                inv_pow = inv ** np.arange(0, p + 1)
                local[dst, 0] += a[0] * np.log(tt) - np.sum(a[1:] / ks * inv_pow[1:])
                s = coef @ (a[1:] * inv_pow[1:])
                local[dst, 1:] += ((-1.0) ** (ls + 1) * a[0] / ls - s) * inv_pow[1:]
        # L2L down the tree, level by level
        for d in range(int(t.depth.max())):
            for node_ in np.flatnonzero((t.depth == d) & (t.child >= 0)):
                f = t.child[node_]
                for c in range(f, f + 4):
                    s = t.center[c] - t.center[node_]
                    for m in range(p + 1):
                        local[c, m] += sum(
                            comb(ell, m) * local[node_, ell] * s ** (ell - m)
                            for ell in range(m, p + 1)
                        )
        self.local = local
        # per-membership potential Phi_m and gradient G_m: far field (L2P) ...
        u = z[part] - t.center[leaf]
        phi_m = np.zeros(len(part))
        g_m = np.zeros(len(part), dtype=complex)  # G_x + i G_y
        wv = np.zeros_like(u)
        dv = np.zeros_like(u)
        for ell in range(p, -1, -1):
            dv = dv * u + wv
            wv = wv * u + local[leaf, ell]
        phi_m += np.real(wv)
        g_m += np.conj(dv)  # W' = G_x - i G_y
        # ... plus the exact near field over membership routes
        by_leaf = {}
        for m, L in enumerate(leaf):
            by_leaf.setdefault(int(L), []).append(m)
        for a, b in p2p:
            ma = np.array(by_leaf.get(int(a), []), dtype=np.int64)
            mb = np.array(by_leaf.get(int(b), []), dtype=np.int64)
            if not len(ma) or not len(mb):
                continue
            ia, ib = part[ma], part[mb]
            dz = z[ia][:, None] - z[ib][None, :]
            same = ia[:, None] == ib[None, :]
            if a == b:
                mask = ia[:, None] < ib[None, :]  # each unordered pair once
            else:
                mask = ~same
            kern, grad = _disk_kernel(np.where(mask, dz, 1.0), self.eps2)
            kern = np.where(mask, kern, 0.0)
            grad = np.where(mask, grad, 0.0)  # d/dz_i of Phi, as G_x + i G_y
            qa, qb = q[ia] * w[ma], q[ib] * w[mb]
            # route (m in a, m' in b): contributions to both members
            np.add.at(phi_m, ma, kern @ qb)
            np.add.at(g_m, ma, grad @ qb)
            np.add.at(phi_m, mb, kern.T @ qa)
            np.add.at(g_m, mb, (-grad).T @ qa)
        self.phi_m, self.g_m = phi_m, g_m

    def __init__(
        self,
        tree: BoxTree,
        pos: np.ndarray,
        charge: np.ndarray,
        *,
        order: int = 12,
        theta: float = 0.5,
        band: float = 0.0,
        symmetric: bool = True,
        softening: float = 0.0,
    ) -> None:
        """Build the tree data for these positions and charges.

        ``softening`` is the radius of the compact charge cloud each particle is
        (beyond it the kernel is the bare one, so the far field is exact).
        """
        self.tree = tree
        self.z = np.asarray(pos[0], dtype=float) + 1j * np.asarray(pos[1], dtype=float)
        self.q = np.asarray(charge, dtype=float)
        self.order = int(order)
        self.theta = float(theta)
        self.band = float(band)
        self.symmetric = bool(symmetric)
        self.eps2 = float(softening) ** 2
        self._compute()

    # -- results ---------------------------------------------------------------
    @property
    def energy(self) -> float:
        """U = -2 sum over unordered routes, in the units of ``flat_tree``."""
        return float(-np.sum(self.q[self.part] * self.w * self.phi_m))

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Potential ``(N,)`` and field ``(2, N)`` such that ``F_i = q_i E_i = -dU/dx_i``."""
        n = len(self.z)
        phi = np.zeros(n)
        np.add.at(phi, self.part, -2.0 * self.w * self.phi_m)
        e = np.zeros(n, dtype=complex)
        np.add.at(e, self.part, 2.0 * (self.w * self.g_m + self.phi_m * self.dw))
        return phi, np.stack((np.real(e), np.imag(e)))

    def potential_at(self, points: np.ndarray) -> np.ndarray:
        """Direct-sum potential at external points, for checks (no tree)."""
        zp = points[0] + 1j * points[1]
        d = zp[:, None] - self.z[None, :]
        return -2.0 * _disk_kernel(d, self.eps2)[0] @ self.q


def direct_energy(pos, charge, softening=0.0):
    """Exact pair energy, same units as :attr:`SmoothFMM.energy`."""
    z = pos[0] + 1j * pos[1]
    iu = np.triu_indices(len(z), 1)
    phi, _ = _disk_kernel(z[iu[0]] - z[iu[1]], softening**2)
    return float(-2.0 * np.sum(charge[iu[0]] * charge[iu[1]] * phi))


def direct_fields(pos, charge, softening=0.0):
    """Exact potential and field, same units as :meth:`SmoothFMM.fields`."""
    z = pos[0] + 1j * pos[1]
    d = z[:, None] - z[None, :]
    np.fill_diagonal(d, 1.0)
    phi, g = _disk_kernel(d, softening**2)
    np.fill_diagonal(phi, 0.0)
    np.fill_diagonal(g, 0.0)
    e = 2.0 * (g @ charge)
    return -2.0 * (phi @ charge), np.stack((np.real(e), np.imag(e)))


def comoving_fields(
    tree: BoxTree,
    pos: np.ndarray,
    charge: np.ndarray,
    mass: np.ndarray,
    center: np.ndarray,
    **options,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Energy, potential and field with the boxes attached to the centre of mass.

    The box set of ``tree`` is used in a frame whose origin follows the centre of
    mass ``c(x)``, i.e. ``U(x) = U_tree(x - (c(x) - center))``. ``U`` is then
    translation invariant, so its exact gradient has zero net force: the
    gradient of the shift subtracts the mass-weighted net force of the
    smooth-membership terms. With this, both energy (smooth ``U``) and momentum
    (Noether) are conserved by a symplectic integrator. ``center`` is where the
    centre of mass was when the boxes were built.
    """
    c = (mass * pos).sum(axis=1) / mass.sum()
    f = SmoothFMM(tree, pos - (c - center)[:, None], charge, **options)
    phi, field = f.fields()
    net = (charge * field).sum(axis=1)
    field = field - (mass / mass.sum())[None, :] * net[:, None] / charge[None, :]
    return f.energy, phi, field
