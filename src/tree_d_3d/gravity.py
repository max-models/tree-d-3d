"""Barnes-Hut N-body solver for fast gravitational force calculations.

This module implements the Barnes-Hut algorithm for O(N log N) computation of
gravitational forces in N-body simulations. It uses quadtree (2D) or octree
(3D) spatial decomposition to approximate distant bodies.

References:
    Barnes, J., & Hut, P. (1986). A hierarchical O(N log N) force-calculation
    algorithm. Nature, 324(6096), 446-449.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np

from tree_d_3d.quad import Quad


class Body:
    """Represents a body (particle) in N-body simulation."""

    pos: np.ndarray
    vel: np.ndarray
    force: np.ndarray
    mass: float

    def __init__(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        mass: float = 1.0,
    ) -> None:
        """Initialize a body with position, velocity, and mass.

        Args:
            x, y: Position coordinates
            vx, vy: Velocity components (default: 0)
            mass: Body mass (default: 1.0)
        """
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([vx, vy], dtype=np.float64)
        self.force = np.zeros(2, dtype=np.float64)
        self.mass = float(mass)

    def reset_force(self) -> None:
        """Reset force to zero."""
        self.force.fill(0.0)

    def add_force(self, fx: float, fy: float) -> None:
        """Add force component to the body.

        Args:
            fx, fy: Force components to add
        """
        self.force[0] += fx
        self.force[1] += fy


class Node:
    """Represents a node in the quadtree."""

    quad: Quad
    body: Body | None
    center_of_mass: np.ndarray
    mass: float
    NW: Node | None
    NE: Node | None
    SW: Node | None
    SE: Node | None

    def __init__(self, quad: Quad) -> None:
        """Initialize a quadtree node.

        Args:
            quad: The quadrant this node represents
        """
        self.quad = quad
        self.body = None
        self.center_of_mass = np.array([0.0, 0.0], dtype=np.float64)
        self.mass = 0.0
        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None

    def is_external(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return (
            self.NW is None and self.NE is None and self.SW is None and self.SE is None
        )


class BarnesHutTree:
    """Quadtree-based Barnes-Hut N-body tree."""

    root: Node

    def __init__(self, root_quad: Quad) -> None:
        """Initialize the Barnes-Hut tree.

        Args:
            root_quad: The root quadrant
        """
        self.root = Node(root_quad)

    def _insert(self, node: Node, body: Body) -> None:
        """Recursively insert a body into the tree.

        Args:
            node: Current tree node
            body: Body to insert
        """
        # If node is an empty leaf, put body here
        if node.body is None and node.is_external() and node.mass == 0.0:
            node.body = body
            node.mass = body.mass
            node.center_of_mass = body.pos.copy()
            return

        # If node is internal, update COM and recurse into child
        if not node.is_external():
            total_mass = node.mass + body.mass
            node.center_of_mass = (
                node.center_of_mass * node.mass + body.pos * body.mass
            ) / total_mass
            node.mass = total_mass
            self._put_into_child(node, body)
            return

        # Leaf node that already contains a single body -> subdivide
        if node.is_external() and node.body is not None:
            existing = node.body
            node.body = None
            # create children
            node.NW = Node(node.quad.northwest())
            node.NE = Node(node.quad.northeast())
            node.SW = Node(node.quad.southwest())
            node.SE = Node(node.quad.southeast())
            # insert existing then new body
            self._put_into_child(node, existing)
            # update this node's mass & com using both bodies
            node.mass = existing.mass + body.mass
            node.center_of_mass = (
                existing.pos * existing.mass + body.pos * body.mass
            ) / node.mass
            self._put_into_child(node, body)
            return

    def insert(self, body: Body) -> None:
        """Insert a body into the tree.

        Args:
            body: Body to insert
        """
        self._insert(self.root, body)

    def _put_into_child(self, node: Node, body: Body) -> None:
        """Recursively put body into appropriate child node.

        Args:
            node: Current node
            body: Body to place in child
        """
        x, y = body.pos[0], body.pos[1]
        q = node.quad
        if x <= q.xmid and y >= q.ymid:
            if node.NW is None:
                node.NW = Node(q.northwest())
            self._insert(node.NW, body)
        elif x > q.xmid and y >= q.ymid:
            if node.NE is None:
                node.NE = Node(q.northeast())
            self._insert(node.NE, body)
        elif x <= q.xmid and y < q.ymid:
            if node.SW is None:
                node.SW = Node(q.southwest())
            self._insert(node.SW, body)
        else:
            if node.SE is None:
                node.SE = Node(q.southeast())
            self._insert(node.SE, body)

    def _compute_force(
        self,
        node: Node | None,
        body: Body,
        theta: float,
        G: float,
        softening: float,
    ) -> None:
        """Recursively compute force on body from tree.

        Args:
            node: Current tree node
            body: Body to compute force on
            theta: Opening angle parameter (smaller = more accurate, slower)
            G: Gravitational constant
            softening: Softening length to avoid singularities
        """
        if node is None or node.mass == 0.0:
            return

        # if leaf
        if node.is_external():
            if node.body is None or node.body is body:
                return
            dx = node.body.pos[0] - body.pos[0]
            dy = node.body.pos[1] - body.pos[1]
            dist = sqrt(dx * dx + dy * dy + softening * softening)
            if dist == 0.0:
                return
            force = G * body.mass * node.body.mass / (dist * dist)
            body.add_force(force * dx / dist, force * dy / dist)
            return

        # internal node: distance to center of mass
        dx = node.center_of_mass[0] - body.pos[0]
        dy = node.center_of_mass[1] - body.pos[1]
        dist = sqrt(dx * dx + dy * dy + softening * softening)
        s = node.quad.half * 2.0
        if dist > 0.0 and (s / dist) < theta:
            force = G * body.mass * node.mass / (dist * dist)
            body.add_force(force * dx / dist, force * dy / dist)
        else:
            self._compute_force(node.NW, body, theta, G, softening)
            self._compute_force(node.NE, body, theta, G, softening)
            self._compute_force(node.SW, body, theta, G, softening)
            self._compute_force(node.SE, body, theta, G, softening)

    def compute_force_on(
        self,
        body: Body,
        theta: float = 0.5,
        G: float = 1.0,
        softening: float = 0.1,
    ) -> None:
        """Compute force on a body from all other bodies in tree.

        Args:
            body: Body to compute force on
            theta: Opening angle (trade-off accuracy vs speed, typically 0.3-0.9)
            G: Gravitational constant
            softening: Softening length to prevent force singularities
        """
        body.reset_force()
        self._compute_force(self.root, body, theta, G, softening)

    def collect_nodes(self) -> list[Node]:
        """Collect all nodes in the tree (for visualization).

        Returns:
            List of all non-empty nodes
        """
        nodes: list[Node] = []

        def _collect(n: Node | None) -> None:
            if n is None or n.mass == 0.0:
                return
            nodes.append(n)
            if not n.is_external():
                _collect(n.NW)
                _collect(n.NE)
                _collect(n.SW)
                _collect(n.SE)

        _collect(self.root)
        return nodes


class BarnesHutSimulation:
    """N-body simulation using Barnes-Hut algorithm.

    This class manages particles and implements the main simulation loop
    using leapfrog integration with force calculation via the Barnes-Hut
    tree algorithm.
    """

    bodies: list[Body]
    bounds: Quad
    theta: float
    G: float
    softening: float
    dt: float

    def __init__(
        self,
        bodies: Sequence[Body],
        bounds: Quad,
        theta: float = 0.5,
        G: float = 1.0,
        softening: float = 0.1,
        dt: float = 0.01,
    ) -> None:
        """Initialize the Barnes-Hut simulation.

        Args:
            bodies: Initial list of bodies
            bounds: Simulation domain boundaries
            theta: Opening angle parameter (0.1-1.0)
            G: Gravitational constant
            softening: Softening length
            dt: Timestep
        """
        self.bodies = list(bodies)
        self.bounds = bounds
        self.theta = theta
        self.G = G
        self.softening = softening
        self.dt = dt

    def build_tree(self) -> BarnesHutTree:
        """Build a new Barnes-Hut tree from current particle positions.

        Returns:
            New BarnesHutTree
        """
        tree = BarnesHutTree(self.bounds)
        for b in self.bodies:
            if self.bounds.contains(b.pos[0], b.pos[1]):
                tree.insert(b)
        return tree

    def step(self) -> BarnesHutTree:
        """Advance simulation by one timestep using leapfrog integration.

        Returns:
            The updated Barnes-Hut tree (for visualization/diagnostics)
        """
        # Build tree and compute initial forces
        tree = self.build_tree()
        for b in self.bodies:
            tree.compute_force_on(
                b,
                theta=self.theta,
                G=self.G,
                softening=self.softening,
            )

        # Leapfrog half-step: update velocities
        for b in self.bodies:
            ax = b.force[0] / b.mass
            ay = b.force[1] / b.mass
            b.vel += 0.5 * np.array([ax, ay]) * self.dt

        # Full-step: update positions
        for b in self.bodies:
            b.pos += b.vel * self.dt

        # Rebuild tree and finish velocity update
        tree = self.build_tree()
        for b in self.bodies:
            tree.compute_force_on(
                b,
                theta=self.theta,
                G=self.G,
                softening=self.softening,
            )
        for b in self.bodies:
            ax = b.force[0] / b.mass
            ay = b.force[1] / b.mass
            b.vel += 0.5 * np.array([ax, ay]) * self.dt

        return tree

    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy of all bodies.

        Returns:
            Total kinetic energy
        """
        ekin = 0.0
        for b in self.bodies:
            ekin += 0.5 * b.mass * np.dot(b.vel, b.vel)
        return ekin

    def get_center_of_mass(self) -> np.ndarray:
        """Calculate center of mass of all bodies.

        Returns:
            Position vector of center of mass
        """
        total_mass = sum(b.mass for b in self.bodies)
        if total_mass == 0:
            return np.zeros(2)
        com = np.zeros(2)
        for b in self.bodies:
            com += b.mass * b.pos
        return com / total_mass
