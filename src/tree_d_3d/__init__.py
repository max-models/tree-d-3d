"""Barnes-Hut trees and fast multipole methods for charged particles in a plane.

* :class:`FlatQuadtree` and :func:`direct_field`: fields of point charges (the 3D
  Coulomb kernel) or line charges (``kernel="log"``, the 2D Poisson kernel) at
  arbitrary targets, by tree walk, FMM or direct sum, on NumPy or CuPy arrays.
* :class:`ElectrostaticSimulation`: time integration of :class:`ChargedBody`
  particles with an optional uniform magnetic field and absorbing walls.
* :class:`ReferenceQuadtree`: a readable object tree computing the same as the flat tree.
* :class:`GroundedStripField`: line charges between two grounded plates, by
  conformal mapping and the FMM.
* :mod:`tree_d_3d.gravity`: the classic gravitational Barnes-Hut N-body code.

The compiled kernels are in :mod:`tree_d_3d.kernels`.
"""

from tree_d_3d.bodies import ChargedBody
from tree_d_3d.constants import K_E
from tree_d_3d.flat_tree import FlatNode, FlatQuadtree, direct_field
from tree_d_3d.kernels import kernels_compiled
from tree_d_3d.quad import Quad
from tree_d_3d.reference import ReferenceNode, ReferenceQuadtree
from tree_d_3d.simulation import AbsorbedParticles, ElectrostaticSimulation
from tree_d_3d.strip import GroundedStripField

__version__ = "0.1.0"  # x-release-please-version

__all__ = [
    "K_E",
    "AbsorbedParticles",
    "ChargedBody",
    "ElectrostaticSimulation",
    "FlatNode",
    "FlatQuadtree",
    "GroundedStripField",
    "Quad",
    "ReferenceNode",
    "ReferenceQuadtree",
    "direct_field",
    "kernels_compiled",
]
