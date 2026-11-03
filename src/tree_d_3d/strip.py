r"""Line charges between two grounded plates, by conformal mapping and the FMM.

Charges are infinite line charges along z (2D Cartesian electrostatics) in the
strip ``0 < x < L`` between grounded plates at ``x = 0`` and ``x = L``, which
extend to infinity in y. The map

$$
\zeta = e^{i\pi z/L}, \qquad z = x + iy,
$$

sends the strip onto the upper half-plane and both plates onto the real axis.
There, a grounded boundary needs a single image per charge, at the complex
conjugate point with the opposite charge, so the strip's Green's function is

$$
\phi(z) = -2k_e\sum_j q_j \ln\left|\frac{\zeta - \zeta_j}{\zeta - \bar\zeta_j}\right|.
$$

That is one free-space line-charge sum over 2N sources, which
`FlatQuadtree.fmm` evaluates in O(N). The field follows from the
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
import scope_profiler as sp
from methodcard import Rule, method
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from tree_d_3d._profiling import _PROFILE_TAGS
from tree_d_3d._utils import _host
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatQuadtree, direct_field
from tree_d_3d.kernels import _kernels
from tree_d_3d.quad import Quad


def _green_rules() -> dict:
    """The SymPy rule of the strip's Green's function in the physical plane (for the tests).

    ``-2 ln|sin((z - z_0) / 2a) / sin((z + conj(z_0)) / 2a)|``, ``a = L / pi``, with
    ``|sin(X + iY)|^2 = sin^2 X + sinh^2 Y``.
    """
    import sympy

    x, y, x0, y0 = sympy.symbols("x y x0 y0", real=True)
    q, k_e, width = sympy.symbols("q k_e width", positive=True)
    a = width / sympy.pi

    def sin2(real, imag):
        return sympy.sin(real / (2 * a)) ** 2 + sympy.sinh(imag / (2 * a)) ** 2

    rule = -k_e * q * sympy.log(sin2(x - x0, y - y0) / sin2(x + x0, y - y0))

    def call(x, y, x0, y0, q, k_e, width):
        strip = GroundedStripField(width, k_e=k_e)
        phi, _ = strip.probe(
            np.array([[x0], [y0]]), np.array([q]), np.array([[x], [y]])
        )
        return float(phi[0])

    ranges = {"x": (0.1, 1.9), "x0": (0.1, 1.9), "y": (-1.0, 1.0), "y0": (-1.0, 1.0)}
    return {"green": Rule(rule, call, ranges | {"width": (2.0, 3.0)})}


def _image_force_rules() -> dict:
    """The SymPy rule of the image force on a lone charge (for the tests)."""
    import sympy

    x, y, q, k_e, width = sympy.symbols("x y q k_e width", positive=True)
    rule = -sympy.pi * k_e * q / width * sympy.cot(sympy.pi * x / width)

    def call(x, q, k_e, width):
        strip = GroundedStripField(width, k_e=k_e)
        _, field = strip.field_at_charges(np.array([[x], [0.3]]), np.array([q]))
        return float(field[0, 0])

    ranges = {"x": (0.1, 1.9), "width": (2.0, 3.0)}
    return {"image_force": Rule(rule, call, ranges)}


@method(
    equations={
        "map": r"\zeta = e^{i\pi z/L}, \qquad z = x + iy: \quad 0 < x < L \ \to\ \operatorname{Im}\zeta > 0, \quad x = 0, L \ \to\ \operatorname{Im}\zeta = 0",
        "green": r"\phi(z) = -2k_e\sum_j q_j \ln\left|\frac{\zeta - \zeta_j}{\zeta - \bar\zeta_j}\right| = -2k_e\sum_j q_j \ln\left|\frac{\sin\big((z - z_j)/2a\big)}{\sin\big((z + \bar z_j)/2a\big)}\right|, \qquad a = \frac{L}{\pi}",
        "field": r"E_x - iE_y = 2k_e\, W'(\zeta)\, \frac{i\pi}{L}\zeta, \qquad W = \sum_j q_j \big[\ln(\zeta - \zeta_j) - \ln(\zeta - \bar\zeta_j)\big]",
    },
    assumptions=(
        r"line charges strictly between grounded plates at $x = 0$ and $x = L$ ($\phi = 0$ there exactly), open in $y$",
        r"one image $-q_j$ at $\bar\zeta_j$ per charge: one free-space log sum over $2N$ sources in the mapped plane, by the FMM (`order` 14, $\theta$ 0.5) or directly",
        r"probes at arbitrary points use the Barnes-Hut walk in the mapped plane",
    ),
    components=(
        "tree_d_3d.flat_tree.FlatQuadtree.fmm",
        "tree_d_3d.flat_tree.FlatQuadtree.evaluate",
    ),
    symbolic=_green_rules,
)
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

    @method(
        equations={
            "self_term": r"E_x - iE_y \mathrel{+}= 2k_e q_i\,\frac{\zeta''}{2\zeta'} = \frac{i\pi k_e q_i}{L}, \qquad \phi \mathrel{-}= 2k_e q_i \ln|\zeta'(z_i)| = 2k_e q_i \ln\frac{\pi|\zeta_i|}{L}",
            "image_force": r"E_x = -\frac{\pi k_e q}{L}\cot\frac{\pi x}{L}, \qquad E_y = 0 \qquad \text{(a lone charge)}",
            "softening": r"\Delta\phi = -k_e q \ln\frac{r^2 + \varepsilon^2}{r^2}, \qquad \Delta\mathbf E = 2k_e q\, \mathbf d \left(\frac{1}{r^2 + \varepsilon^2} - \frac{1}{r^2}\right), \qquad r < c\,\varepsilon",
        },
        assumptions=(
            r"a charge's own term $\ln(\zeta - \zeta_i)$ is dropped from the sum and its regular part $\ln\frac{\zeta - \zeta_i}{z - z_i} \to \ln\zeta'(z_i)$ restored: the image force of the plates plus all other charges and their images",
            r"softening has no fixed image in the mapped plane ($|d\zeta/dz| = \pi|\zeta|/L$): pairs closer than $c\,\varepsilon$ ($c$ = `cutoff` = 6) in the physical plane get the free-space difference; beyond, the neglected change is below $1/(1 + c^2)$ of a pair field",
            r"a lone charge at mid-gap feels no force",
        ),
        symbolic=_image_force_rules,
    )
    @sp.profile("tree_d_3d.strip.field_at_charges", tags=_PROFILE_TAGS)
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

    @sp.profile("tree_d_3d.strip.probe", tags=_PROFILE_TAGS)
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
