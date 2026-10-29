"""The animated figure at the top of the docs home page: ``docs/src/assets/hero.svg``.

tree-d-3d builds a Barnes-Hut quadtree (``FlatQuadtree``, compiled kernels) on a few hundred
charges of both signs, clustered on the right. A probe point circles through the clusters; at
twelve positions along its loop the figure walks the tree with the opening test of
``FlatQuadtree.evaluate`` (theta = 0.5) and highlights the boxes the probe takes as a single
multipole, which grow with the distance and are redrawn as the probe moves. The leaves it sums
directly are outlined. Takes about a second.

Rerun with ``make docs-hero`` or ``python docs/hero.py`` (needs ``plasma_plots.hero``).
"""

from pathlib import Path

import numpy as np
from plasma_plots.hero import HeroFigure

from tree_d_3d import FlatQuadtree, Quad

OUTPUT = Path(__file__).parent / "src" / "assets" / "hero.svg"
XLIM, YLIM = (0.0, 8.0), (0.0, 3.0)
THETA = 0.5
CYCLE = 18.0
FRAMES = 12

rng = np.random.default_rng(3)
centres = [(4.6, 1.9, 0.35, 120), (6.3, 1.0, 0.3, 120), (6.9, 2.3, 0.2, 70)]
clusters = [rng.normal((cx, cy), s, (n, 2)) for cx, cy, s, n in centres]
background = rng.uniform((0.2, 0.1), (7.9, 2.9), (90, 2))
pos = np.concatenate(clusters + [background]).T
pos = pos[:, (pos[0] > 0.05) & (pos[0] < 7.95) & (pos[1] > 0.05) & (pos[1] < 2.95)]
q = rng.choice([-1.0, 1.0], pos.shape[1])
tree = FlatQuadtree(pos, q, Quad(4.0, 1.5, 4.0), leaf_capacity=6)


def interactions(target):
    """The boxes the tree walk for ``target`` accepts as multipoles, and the leaves it opens."""
    accepted, leaves = [], []

    def walk(node):
        quad = node.quad
        if node.is_external():
            leaves.append(quad)
            return
        distance = np.hypot(target[0] - quad.xmid, target[1] - quad.ymid)
        if not quad.contains(*target) and 2 * quad.half < THETA * distance:
            accepted.append(quad)
            return
        for child in node.children:
            if child is not None and child.count > 0:
                walk(child)

    walk(tree.root)
    return accepted, leaves


# the probe's loop, a circle through the clusters; frames evenly spaced along it
angle = np.linspace(0, 2 * np.pi, 121)
loop_x, loop_y = 5.7 + 1.1 * np.cos(angle), 1.5 - 1.1 * np.sin(angle)
frames = [
    interactions((5.7 + 1.1 * np.cos(a), 1.5 - 1.1 * np.sin(a)))
    for a in np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
]

fig = HeroFigure(xlim=XLIM, ylim=YLIM)


def key(quad):
    """A hashable label of a box."""
    return (round(quad.xmid, 6), round(quad.ymid, 6), round(quad.half, 6))


for index in (0, 1):  # 0: accepted multipoles, 1: leaves summed directly
    seen = {}
    for frame, boxes in enumerate(frames):
        for quad in boxes[index]:
            seen.setdefault(key(quad), [0.0] * FRAMES)[frame] = 1.0
    for (x, y, h), shown in seen.items():
        if index == 0:
            box = fig.rect(x - h, y - h, x + h, y + h, tone="low", opacity=0.55)
            box.attrs["class"] += " s-accent"
            box.attrs["stroke-width"] = "1"
        else:
            box = fig.rect(x - h, y - h, x + h, y + h, tone="bg", opacity=0.0)
            box.attrs["class"] += " s-muted"
            box.attrs["stroke-width"] = "0.8"
        box.animate("opacity", shown + shown[:1], CYCLE)

for x, y, charge in zip(pos[0], pos[1], q, strict=True):
    fig.circle(x, y, 1.8, tone="high" if charge > 0 else "accent")

path = fig.line(loop_x, loop_y, tone="muted", width=1, opacity=0.5, dash="3 6")
fig.tracer(path, duration=CYCLE, r=6, tone="high")

print(f"Saved {fig.save(OUTPUT)}")
