"""The Python side of the live demo: a quadtree, its Barnes-Hut walk and the fields it gives.

It runs in the browser (Pyodide); ``run`` takes the values of the controls and returns the
figure of the particles, the cells of the tree and the walk for one target, with the
accuracy and the cost of the tree walk against the direct sum.
"""

import time

import numpy as np

from tree_d_3d import FlatQuadtree, Quad, direct_field

SOFTENING = 1e-3
# viridis without its pale yellow end, readable on light and dark pages
DEPTH_COLORS = [
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
]


def particles(distribution, n):
    """``n`` positions in the unit square and charges of both signs."""
    rng = np.random.default_rng(1)
    if distribution == "clusters":
        centers = rng.uniform(-0.6, 0.6, (2, 5))
        widths = rng.uniform(0.04, 0.15, 5)
        which = rng.integers(0, 5, n)
        pos = centers[:, which] + widths[which] * rng.normal(size=(2, n))
    elif distribution == "ring":
        angle = rng.uniform(0, 2 * np.pi, n)
        radius = 0.6 + 0.05 * rng.normal(size=n)
        pos = radius * np.array([np.cos(angle), np.sin(angle)])
    elif distribution == "spiral":
        s = rng.uniform(0, 1, n)
        angle = 4 * np.pi * s
        pos = 0.9 * s * np.array([np.cos(angle), np.sin(angle)])
        pos += 0.02 * rng.normal(size=(2, n))
    else:
        pos = rng.uniform(-0.95, 0.95, (2, n))
    return np.clip(pos, -0.99, 0.99), rng.choice([-1.0, 1.0], n)


def walk(tree, target, theta):
    """The cells the walk for ``target`` takes as multipoles, and the leaves it sums."""
    multipoles, leaves, cells = [], [], [0]
    while cells:
        cell = cells.pop()
        if tree.node_qabs[cell] == 0.0:
            continue  # an empty quadrant
        first = tree.node_child[cell]
        if first < 0:
            leaves.append(cell)
            continue
        r = target - tree.node_center[cell]
        half = tree.node_half[cell]
        inside = np.all(np.abs(r) <= half)
        if not inside and 2 * half < theta * np.hypot(*r):
            multipoles.append(cell)
        else:
            cells.extend(range(first, first + 4))
    return multipoles, leaves


def outlines(tree, cells):
    """The outlines of ``cells`` as one line with gaps, for a scatter trace."""
    if len(cells) == 0:
        return [], []
    c = tree.node_center[cells]
    h = tree.node_half[cells][:, None]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    x = c[:, :1] + h * corners[:, 0]
    y = c[:, 1:] + h * corners[:, 1]
    gap = np.full((len(cells), 1), None)
    return np.hstack([x, gap]).ravel().tolist(), np.hstack([y, gap]).ravel().tolist()


def run(
    distribution="clusters",
    n=2000,
    leaf_capacity=8,
    theta=0.5,
    target_x=0.2,
    target_y=-0.1,
):
    """Build the tree, walk it for one target and compare the fields with the direct sum.

    Args:
        distribution: How the particles are spread: uniform, clusters, ring or spiral.
        n: Number of particles.
        leaf_capacity: The most particles a leaf holds.
        theta: Opening angle of the walk.
        target_x, target_y: The point whose walk the figure shows.

    Returns:
        The figure and the counts, errors and timings.
    """
    n, leaf_capacity = int(n), int(leaf_capacity)
    pos, charge = particles(distribution, n)
    begin = time.perf_counter()
    tree = FlatQuadtree(pos, charge, Quad(0.0, 0.0, 1.0), leaf_capacity=leaf_capacity)
    built = time.perf_counter() - begin

    # the field at every particle, by the tree and by the direct sum
    others = np.arange(n)
    begin = time.perf_counter()
    field = tree.evaluate(pos, exclude=others, theta=theta, softening=SOFTENING)[1]
    walked = time.perf_counter() - begin
    begin = time.perf_counter()
    exact = direct_field(pos, pos, charge, softening=SOFTENING, exclude=others)[1]
    summed = time.perf_counter() - begin
    # each particle's error over the RMS field: a field near zero has no relative error
    scale = np.sqrt(np.mean(np.sum(exact**2, axis=0)))
    error = np.linalg.norm(field - exact, axis=0) / scale

    target = np.array([target_x, target_y])
    multipoles, leaves = walk(tree, target, theta)
    direct = sum(int(tree.node_end[c] - tree.node_start[c]) for c in leaves)

    # the cells of the tree by depth, the walk's cells filled, the particles on top
    held = np.flatnonzero(tree.node_qabs > 0)
    depth = tree.node_depth[held]
    traces = []
    for d in range(int(depth.max()) + 1):
        x, y = outlines(tree, held[depth == d])
        color = DEPTH_COLORS[min(d, len(DEPTH_COLORS) - 1)]
        traces.append(
            {
                "type": "scatter",
                "x": x,
                "y": y,
                "mode": "lines",
                "line": {"color": color, "width": 1.6 if d < 3 else 0.8},
                "opacity": 0.8,
                "hoverinfo": "skip",
                "name": "cells (dark: shallow, green: deep)",
                "legendgroup": "cells",
                "showlegend": d == 0,
            }
        )
    for cells, name, color in (
        (multipoles, "multipole cells", "rgba(255, 140, 0, 0.3)"),
        (leaves, "leaves summed directly", "rgba(220, 40, 60, 0.3)"),
    ):
        x, y = outlines(tree, cells)
        traces.append(
            {
                "type": "scatter",
                "x": x,
                "y": y,
                "mode": "lines",
                "fill": "toself",
                "fillcolor": color,
                "line": {"color": color.replace("0.3)", "1)"), "width": 1.2},
                "hoverinfo": "skip",
                "name": f"{name} ({len(cells)})",
            }
        )
    for sign, name, color in ((1, "+", "#d62728"), (-1, "−", "#1f77b4")):
        mask = charge == sign
        traces.append(
            {
                "type": "scattergl",
                "x": pos[0, mask],
                "y": pos[1, mask],
                "mode": "markers",
                "marker": {"size": 3 if n > 2000 else 4, "color": color},
                "customdata": error[mask],
                "hovertemplate": "field error %{customdata:.1e}<extra></extra>",
                "name": f"charges {name}",
            }
        )
    traces.append(
        {
            "type": "scatter",
            "x": [target_x],
            "y": [target_y],
            "mode": "markers",
            "marker": {
                "symbol": "star",
                "size": 16,
                "color": "#ffd700",
                "line": {"color": "black", "width": 1},
            },
            "name": "target",
        }
    )
    axis = {"range": [-1.0, 1.0], "showgrid": False, "zeroline": False}
    return {
        "figure": {
            "data": traces,
            "layout": {
                "xaxis": {**axis, "title": {"text": "x"}, "constrain": "domain"},
                "yaxis": {
                    **axis,
                    "title": {"text": "y"},
                    "scaleanchor": "x",
                    "constrain": "domain",
                },
                "legend": {"orientation": "h", "y": -0.15},
                "margin": {"t": 20},
            },
        },
        "numbers": {
            "Cells (leaves)": f"{len(held)} ({int(np.sum(tree.node_child[held] < 0))})",
            "Depth": int(depth.max()),
            "Target: multipoles + particles": f"{len(multipoles)} + {direct} of {n}",
            "Median field error": float(np.median(error)),
            "Largest field error": float(error.max()),
            "Tree build + walk [ms]": f"{1e3 * built:.0f} + {1e3 * walked:.0f}",
            "Direct sum [ms]": f"{1e3 * summed:.0f}",
        },
    }
