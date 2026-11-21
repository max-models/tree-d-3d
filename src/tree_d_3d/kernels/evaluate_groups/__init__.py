"""The evaluate_groups kernel of the Barnes-Hut walk on the CPU: Pyccel and NumPy versions
with the same arguments.

The tree walk of ``evaluate_quadtree`` (``kernel="coulomb"``) and
``evaluate_log_quadtree`` (``kernel="log"``) for groups of nearby targets at once: the
group walks the tree together, each target keeping the cells it would take alone (a
bit mask of the targets still walking), so every target gets the same terms in the same
order as its own walk. The sources are read in tree order. The GPU walks one target per
thread instead.
"""

from tree_d_3d.kernels import KERNEL_PACKAGE

kernel = KERNEL_PACKAGE.from_folder(__name__)
