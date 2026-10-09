"""The object quadtree: a readable reference implementation of the flat tree.

It evaluates one target at a time in Python, with the same moments (through the
quadrupole) and opening criterion as :class:`tree_d_3d.flat_tree.FlatQuadtree`; kept for
inspection and tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tree_d_3d._utils import _finite
from tree_d_3d.bodies import ChargedBody
from tree_d_3d.constants import K_E
from tree_d_3d.quad import Quad


class ReferenceNode:
    """Bucket node with signed moments about its geometric centre."""

    def __init__(self, quad: Quad) -> None:
        self.quad = quad
        self.body = None  # Compatibility: populated for a single-particle leaf.
        self.bodies = []
        self.count = 0
        self.total_charge = 0.0
        self.total_abs_charge = 0.0
        self.center_of_charge = np.zeros(2)
        self.dipole = np.zeros(2)
        self.second_moment = np.zeros((2, 2))
        self.NW = self.NE = self.SW = self.SE = None

    def is_external(self) -> bool:
        return self.NW is None

    @property
    def children(self):
        return self.NW, self.NE, self.SW, self.SE


class ReferenceQuadtree:
    """Bounded-depth quadtree with signed moments through quadrupole order.

    Coincident particles remain in leaf buckets. Theta=0 evaluates all pairs.
    Cells containing the target are always opened. The opening criterion uses
    unsoftened distance so softening cannot make a nearby cell look distant.
    """

    def __init__(
        self, root_quad: Quad, *, leaf_capacity: int = 8, max_depth: int = 48
    ) -> None:
        _finite("half_size", root_quad.half, positive=True)
        _finite("xmid", root_quad.xmid)
        _finite("ymid", root_quad.ymid)
        for name, value in (("leaf_capacity", leaf_capacity), ("max_depth", max_depth)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.root = ReferenceNode(root_quad)
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self._identities = set()

    def _insert(self, node, body, depth):
        q = node.quad
        offset = body.pos - (q.xmid, q.ymid)
        total_abs = node.total_abs_charge + abs(body.charge)
        if total_abs:
            node.center_of_charge += (
                abs(body.charge) / total_abs * (body.pos - node.center_of_charge)
            )
        node.total_abs_charge = total_abs
        node.total_charge += body.charge
        node.dipole += body.charge * offset
        node.second_moment += body.charge * np.outer(offset, offset)
        node.count += 1

        if node.is_external():
            if len(node.bodies) < self.leaf_capacity or depth >= self.max_depth:
                node.bodies.append(body)
                node.body = body if len(node.bodies) == 1 else None
                return
            node.NW = ReferenceNode(q.northwest())
            node.NE = ReferenceNode(q.northeast())
            node.SW = ReferenceNode(q.southwest())
            node.SE = ReferenceNode(q.southeast())
            previous = node.bodies
            node.bodies = []
            node.body = None
            for existing in previous:
                self._put_into_child(node, existing, depth)
        self._put_into_child(node, body, depth)

    def insert(self, body: ChargedBody) -> None:
        if not np.all(np.isfinite(body.pos)):
            raise ValueError("Particle position must be finite")
        if not self.root.quad.contains(*body.pos):
            raise ValueError("Particle lies outside tree bounds")
        if id(body) in self._identities:
            raise ValueError("Cannot insert the same particle twice")
        self._identities.add(id(body))
        self._insert(self.root, body, 0)

    def _put_into_child(self, node, body, depth):
        right = body.pos[0] > node.quad.xmid
        bottom = body.pos[1] < node.quad.ymid
        child = node.children[2 * int(bottom) + int(right)]
        self._insert(child, body, depth + 1)

    def _evaluate(self, node, position, theta, softening, exclude):
        if node is None or node.total_abs_charge == 0:
            return 0.0, np.zeros(2)
        if node.is_external():
            sources = [b for b in node.bodies if b is not exclude and b.charge != 0]
            if not sources:
                return 0.0, np.zeros(2)
            delta = position - np.array([b.pos for b in sources])
            r2 = np.einsum("ij,ij->i", delta, delta) + softening**2
            if np.any(r2 == 0):
                raise ValueError("Coincident charges require positive softening")
            inv_r = 1 / np.sqrt(r2)
            charges = np.array([b.charge for b in sources])
            return float(charges @ inv_r), np.sum(
                (charges * inv_r**3)[:, None] * delta, axis=0
            )

        q = node.quad
        r = position - (q.xmid, q.ymid)
        distance = float(np.linalg.norm(r))
        if not q.contains(*position) and 2 * q.half < theta * distance:
            inv_r = 1 / np.sqrt(distance**2 + softening**2)
            pr = float(node.dipole @ r)
            sr = node.second_moment @ r
            rsr = float(r @ sr)
            trace = float(np.trace(node.second_moment))
            potential = (
                node.total_charge * inv_r
                + pr * inv_r**3
                + 0.5 * (3 * rsr * inv_r**5 - trace * inv_r**3)
            )
            field = (
                node.total_charge * r * inv_r**3
                + 3 * pr * r * inv_r**5
                - node.dipole * inv_r**3
                + 7.5 * rsr * r * inv_r**7
                - 3 * sr * inv_r**5
                - 1.5 * trace * r * inv_r**5
            )
            return potential, field
        potential, field = 0.0, np.zeros(2)
        for child in node.children:
            phi, electric = self._evaluate(child, position, theta, softening, exclude)
            potential += phi
            field += electric
        return potential, field

    def field_at(
        self,
        position: Sequence[float],
        *,
        theta: float = 0.5,
        k_e: float = K_E,
        softening: float = 0.0,
        exclude: ChargedBody | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return (potential, electric field), excluding one source by identity.

        Potential and field use the same softened multipole expansion. Rebuild
        the tree after moving or changing any source particle.
        """
        point = np.asarray(position, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("position must be a finite two-vector")
        theta = _finite("theta", theta, minimum=0)
        k_e = _finite("k_e", k_e, minimum=0)
        softening = _finite("softening", softening, minimum=0)
        # Exclusion also works for off-particle probes.
        if exclude is not None and not np.array_equal(point, exclude.pos):
            theta = 0.0
        if k_e == 0:
            return 0.0, np.zeros(2)
        phi, electric = self._evaluate(self.root, point, theta, softening, exclude)
        return k_e * phi, k_e * electric

    def compute_lorentz_force_on(
        self,
        body: ChargedBody,
        theta: float = 0.5,
        k_e: float = K_E,
        softening: float = 0.0,
        B_z: float = 0.0,
    ) -> None:
        """Store q(E + v cross B), with no self-force."""
        B_z = _finite("B_z", B_z)
        if body.charge == 0:
            body.reset_force()
            return
        _, electric = self.field_at(
            body.pos, theta=theta, k_e=k_e, softening=softening, exclude=body
        )
        body.force[:] = body.charge * (
            electric + B_z * np.array([body.vel[1], -body.vel[0]])
        )

    def collect_nodes(self) -> list[ReferenceNode]:
        nodes = []

        def collect(node):
            if node is not None and node.count:
                nodes.append(node)
                for child in node.children:
                    collect(child)

        collect(self.root)
        return nodes
