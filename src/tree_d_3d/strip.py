"""Line charges between two grounded plates, by conformal mapping and the FMM.

Charges are infinite line charges along z (2D Cartesian electrostatics) in the
strip ``0 < x < L`` between grounded plates at ``x = 0`` and ``x = L``, which
extend to infinity in y. The map

    zeta = exp(i pi z / L),   z = x + i y,

sends the strip onto the upper half-plane and both plates onto the real axis.
There, a grounded boundary needs a single image per charge, at the complex
conjugate point with the opposite charge, so the strip's Green's function is

    phi(z) = -2 k_e sum_j q_j ln| (zeta - zeta_j) / (zeta - conj(zeta_j)) |.

That is one free-space line-charge sum over 2N sources, which
:meth:`FlatQuadtree.fmm` evaluates in O(N). The field follows from the
complex derivative, ``E_x - i E_y = 2 k_e W'(zeta) (i pi / L) zeta`` with
``W = sum q log(zeta - zeta_j) - sum q log(zeta - conj(zeta_j))``. Every
charge feels the field of all other charges and of all images, including its
own: the image force of the plates. In the mapped plane a charge's own term
log(zeta - zeta_i) is not purely a self-interaction: it differs from
log(z - z_i) by a smooth part whose gradient, ``i pi / 2L``, is added back.

The map compresses distances by ``|d zeta / dz| = pi |zeta| / L``, which
changes by ``exp(pi Delta y / L)`` across the strip, so a softening length in
the physical plane has no fixed counterpart in the mapped plane. Softening is
therefore applied in the physical plane as a local correction: the strip's
Green's function is the free-space kernel plus a smooth harmonic part, and
softening only changes the free-space kernel near the charge, so pairs closer
than ``cutoff * softening`` get the difference between the softened and the
bare free-space interaction (the remainder beyond the cutoff is below 3% of
one pair's field there). Coincident charges are an error either way.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from tree_d_3d._utils import _host
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import _kernels
from tree_d_3d.quad import Quad


class GroundedStripField:
    """Field and potential of line charges between plates at x = 0 and x = L."""

    def __init__(
        self,
        width: float,
        *,
        order: int = 14,
        theta: float = 0.5,
        leaf_capacity: int = 32,
        k_e: float = K_E,
        method: str = "fmm",
        softening: float = 0.0,
        cutoff: float = 6.0,
    ) -> None:
        if not np.isfinite(width) or width <= 0:
            raise ValueError("width must be finite and positive")
        if method not in ("fmm", "direct"):
            raise ValueError("method must be 'fmm' or 'direct'")
        self.width = float(width)
        self.order = int(order)
        self.theta = float(theta)
        self.leaf_capacity = int(leaf_capacity)
        self.k_e = float(k_e)
        self.method = method
        if not np.isfinite(softening) or softening < 0:
            raise ValueError("softening must be finite and nonnegative")
        if not cutoff > 1:
            raise ValueError("cutoff must exceed 1")
        self.softening = float(softening)
        self.cutoff = float(cutoff)

    def to_plane(self, positions) -> np.ndarray:
        """Zeta = exp(i pi z / L), as a complex array; ``positions`` is ``(2, N)``."""
        xy = np.asarray(positions, dtype=float).reshape(2, -1)
        if np.any((xy[0] <= 0) | (xy[0] >= self.width)):
            raise ValueError("Charges must lie strictly between the plates")
        return np.exp(1j * np.pi * (xy[0] + 1j * xy[1]) / self.width)

    def _sources(self, positions, charges):
        """Return the mapped charges, and the ``(2, 2N)`` charges and images with their charges."""
        zeta = self.to_plane(positions)
        q = np.asarray(charges, dtype=float).ravel()
        if len(q) != len(zeta):
            raise ValueError("positions and charges must have the same length")
        points = np.concatenate((zeta, np.conj(zeta)))
        xy = np.stack((np.real(points), np.imag(points)), axis=0)
        return zeta, xy, np.concatenate((q, -q))

    @staticmethod
    def _root(xy) -> Quad:
        lo, hi = xy.min(axis=1), xy.max(axis=1)
        centre = (lo + hi) / 2
        half = float(np.max(hi - lo) / 2) * (1 + 1e-9) + 1e-300
        return Quad(float(centre[0]), float(centre[1]), half)

    def _soften_pairs(self, xy, q, potential, field, with_potential=True) -> None:
        """Replace the bare free-space kernel by the softened one for close pairs.

        ``xy`` is ``(2, N)``, ``field`` ``(2, N)``.
        """
        eps2 = self.softening**2
        pairs = cKDTree(xy.T).query_pairs(
            self.cutoff * self.softening, output_type="ndarray"
        )
        if not len(pairs):
            return
        # The pairs come from a k-d tree on the host, so the kernel runs there too
        # (its Pyccel version, or its NumPy version without Pyccel).
        pairs = np.ascontiguousarray(pairs, dtype=np.int64)
        status = np.zeros(1, dtype=np.int64)
        _kernels().soften_pairs(
            np.ascontiguousarray(xy, dtype=float),
            np.ascontiguousarray(q, dtype=float),
            pairs,
            eps2,
            1 if with_potential else 0,
            potential,
            field,
            status,
            n_threads=len(pairs),
        )
        if status[0]:
            raise ValueError("Coincident charges")

    def field_at_charges(
        self, positions, charges, *, potential: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m, shape (2, N)) acting on every charge.

        ``positions`` has shape ``(2, N)``; ``charges`` are line charges per unit
        length (C/m). With
        ``potential=False`` only the field is computed (the potential is
        returned as zeros), which is cheaper.
        """
        zeta, xy, q = self._sources(positions, charges)
        n = len(zeta)
        if not n:
            return np.zeros(0), np.zeros((2, 0))
        if self.method == "fmm":
            tree = FlatQuadtree(
                xy,
                q,
                self._root(xy),
                kernel="log",
                order=self.order,
                leaf_capacity=self.leaf_capacity,
            )
            phi, raw = tree.fmm(theta=self.theta, potential=potential)
        else:
            phi, raw = direct_field(xy, xy, q, exclude=np.arange(2 * n), kernel="log")
        # the tree runs on cunumpy's active backend; the mapping below on the host
        phi, raw = _host(phi), _host(raw)
        # Raw field in the mapped plane: E_x - i E_y = 2 W'(zeta).
        mapped = raw[0, :n] - 1j * raw[1, :n]
        complex_field = mapped * (1j * np.pi / self.width) * zeta
        # Excluding log(zeta - zeta_i) also dropped the regular part of a
        # charge's own term, log((zeta - zeta_i) / (z - z_i)); only log(z - z_i)
        # is the self-interaction. Restore it: its z-derivative at z_i is
        # zeta'' / (2 zeta') = i pi / (2 L), and its value is ln|zeta'(z_i)|.
        q_own = q[:n]
        complex_field += 2 * q_own * (1j * np.pi / (2 * self.width))
        field = np.stack((complex_field.real, -complex_field.imag), axis=0)
        if potential:
            phi_own = phi[:n] - 2 * q_own * np.log(np.pi * np.abs(zeta) / self.width)
        else:
            phi_own = np.zeros(n)
        if self.softening > 0:
            self._soften_pairs(
                np.ascontiguousarray(positions, dtype=float).reshape(2, -1),
                np.ascontiguousarray(q_own),
                phi_own,
                field,
                potential,
            )
        return self.k_e * phi_own, self.k_e * field

    def potential(self, positions, charges, points) -> np.ndarray:
        """Potential (V) at arbitrary ``points`` strictly between the plates."""
        return self.probe(positions, charges, points)[0]

    def probe(self, positions, charges, points) -> tuple[np.ndarray, np.ndarray]:
        """Potential (V) and field (V/m, shape (2, M)) at ``points`` ``(2, M)``.

        The points must lie strictly between the plates; they are probes, not
        charges, so no self term applies.
        """
        _, xy, q = self._sources(positions, charges)
        probes = self.to_plane(points)
        probe_xy = np.stack((np.real(probes), np.imag(probes)), axis=0)
        num_probes = probe_xy.shape[1]
        if not len(q):
            return np.zeros(num_probes), np.zeros((2, num_probes))
        if self.method == "fmm":
            everything = np.concatenate((xy, probe_xy), axis=1)
            tree = FlatQuadtree(
                xy,
                q,
                self._root(everything),
                kernel="log",
                order=self.order,
                leaf_capacity=self.leaf_capacity,
            )
            phi, raw = tree.evaluate(probe_xy, theta=self.theta)
        else:
            phi, raw = direct_field(probe_xy, xy, q, kernel="log")
        phi, raw = _host(phi), _host(raw)
        mapped = raw[0] - 1j * raw[1]
        complex_field = mapped * (1j * np.pi / self.width) * probes
        field = np.stack((complex_field.real, -complex_field.imag), axis=0)
        if self.softening > 0:
            # marker-major for the k-d tree and the pair differences below
            sources = np.asarray(positions, dtype=float).reshape(2, -1).T
            charges = np.asarray(charges, dtype=float).ravel()
            targets = np.asarray(points, dtype=float).reshape(2, -1).T
            near = cKDTree(sources).query_ball_point(
                targets, self.cutoff * self.softening, return_sorted=False
            )
            eps2 = self.softening**2
            counts = np.fromiter(
                (len(n) for n in near), dtype=np.int64, count=len(near)
            )
            if counts.sum():
                target = np.repeat(np.arange(len(near)), counts)
                source = np.concatenate([n for n in near if n]).astype(np.int64)
                d = targets[target] - sources[source]
                r2 = np.einsum("ij,ij->i", d, d)
                if np.any(r2 == 0):
                    raise ValueError("A probe coincides with a charge")
                # Raw kernels: potential -q ln(r^2), field 2 q d / r^2.
                change = charges[source] * np.log((r2 + eps2) / r2)
                phi -= np.bincount(target, change, minlength=len(targets))
                weight = charges[source] * (2 / (r2 + eps2) - 2 / r2)
                for axis in range(2):
                    field[axis] += np.bincount(
                        target, weight * d[:, axis], minlength=len(targets)
                    )
        return self.k_e * phi, self.k_e * field
