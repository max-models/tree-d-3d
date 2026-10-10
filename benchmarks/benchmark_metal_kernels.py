"""Benchmark the Metal (Apple-silicon GPU) versions of the tree-d-3d kernels against the CPU.

Every kernel in ``tree_d_3d.kernels.METAL_KERNEL_NAMES`` runs on the same arrays
through ``KERNELS[name]`` (the compiled Pyccel version) and through ``metal_kernel(name)``
(MLX, float32, NumPy arrays in and out, so every Metal call includes the copies). The
table gives the best of ``--repeat`` timings with the first-call compile excluded, the
speed-up and the largest difference relative to the largest CPU value.

Usage::

    python benchmarks/benchmark_metal_kernels.py
        [--direct-sizes 2000,10000,30000] [--tree-size 100000] [--repeat 5]
        [--dtype float32]

With ``--dtype float32`` every float array handed to the Metal kernels is float32 (no
cast inside ``MetalKernel``, half the bytes copied); the CPU kernels only take float64,
so their reference runs on the float32-rounded values. The table then also shows the
Metal float64 run of the same session.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tree_d_3d.flat_tree import FlatQuadtree, _binomial_table
from tree_d_3d.kernels import KERNELS, metal_kernel
from tree_d_3d.quad import Quad

SOFTENING = 0.01
ORDER = 8
THETA_TREE = 0.6
THETA_FMM = 0.5


@dataclass
class Case:
    """One benchmark row: ``prepare()`` builds fresh arguments, ``call`` runs a kernel."""

    family: str
    name: str
    label: str
    prepare: Callable[[], object]
    call: Callable[[Callable[..., object], object], object]
    outputs: Callable[[object], Sequence[np.ndarray]]


def to_float32(state: object) -> object:
    """Round the float64 arrays and floats of an argument tuple to float32 (ints stay)."""
    if isinstance(state, tuple):
        return tuple(to_float32(item) for item in state)
    if isinstance(state, np.ndarray) and state.dtype == np.float64:
        return state.astype(np.float32)
    if isinstance(state, float):
        return float(np.float32(state))
    return state


def to_float64(state: object) -> object:
    """Widen the float32 arrays of an argument tuple back to float64."""
    if isinstance(state, tuple):
        return tuple(to_float64(item) for item in state)
    if isinstance(state, np.ndarray) and state.dtype == np.float32:
        return state.astype(np.float64)
    return state


def _best(function: Callable[[], None], repeat: int, budget: float = 3.0) -> float:
    """Best of ``repeat`` wall times in s, stopping early once a call exceeds ``budget``."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        if elapsed > budget:
            break
    return best


def _time(
    case: Case,
    function: Callable[..., object],
    repeat: int,
    transform: Callable[[object], object] = lambda state: state,
) -> tuple[float, Sequence, float]:
    """Return the best time of ``case.call(function, state)``, its outputs, the first call.

    ``transform`` converts each freshly prepared state, outside the timing.
    """
    state = transform(case.prepare())
    start = time.perf_counter()
    case.call(function, state)  # first call: compile and warm up
    first = time.perf_counter() - start
    best = float("inf")
    outputs: Sequence = ()
    for _ in range(repeat):
        state = transform(case.prepare())  # fresh arguments, outside the timing
        start = time.perf_counter()
        case.call(function, state)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        outputs = [np.array(a, copy=True) for a in case.outputs(state)]
        if elapsed > 3.0:
            break
    return best, outputs, first


def _difference(gpu: Sequence[np.ndarray], cpu: Sequence[np.ndarray]) -> float:
    worst = 0.0
    for a, b in zip(gpu, cpu):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        scale = max(float(np.nanmax(np.abs(b))) if b.size else 0.0, 1e-30)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            worst = max(worst, float(np.abs(a - b)[finite].max()) / scale)
    return worst


# --------------------------------------------------------------------------- #
# problem builders
# --------------------------------------------------------------------------- #
def _cloud(n: int, seed: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # component-major (2, N)
    pos = np.clip(
        np.concatenate(
            (rng.normal(0, 0.05, (2, n // 2)), rng.uniform(-1, 1, (2, n - n // 2))),
            axis=1,
        ),
        -0.99,
        0.99,
    )
    charge = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    return pos, charge


def _simple(
    family: str,
    name: str,
    label: str,
    make: Callable[[], tuple],
    outputs: Callable[[tuple], Sequence[np.ndarray]],
    fresh: Callable[[tuple], tuple] | None = None,
) -> Case:
    """A kernel called as ``function(*args)``; ``fresh`` copies the mutated arguments."""
    base = make()

    def prepare() -> tuple:
        return fresh(base) if fresh else base

    return Case(family, name, label, prepare, lambda f, args: f(*args), outputs)


def direct_cases(sizes: Sequence[int]) -> list[Case]:
    """The direct sums and the pair energy on clouds of each size."""
    cases = []
    for n in sizes:
        pos, charge = _cloud(n)
        exclude = np.arange(n, dtype=np.int64)
        label = f"n={n}"

        for name in ("direct_sum", "direct_log_sum"):

            def fresh(_, pos=pos, charge=charge, exclude=exclude):
                return (
                    pos, exclude, pos, charge, SOFTENING,
                    np.zeros(pos.shape[1]), np.zeros((2, pos.shape[1])), np.zeros(1, dtype=np.int64),
                )  # fmt: skip

            cases.append(
                _simple("direct sums", name, label, lambda: (), lambda a: a[5:7], fresh)
            )

        def fresh_energy(_, pos=pos, charge=charge):
            return pos, charge, SOFTENING, np.zeros(1, dtype=np.int64), np.zeros(1)

        cases.append(
            _simple(
                "direct sums",
                "pair_energy",
                label,
                lambda: (),
                lambda a: a[4:5],
                fresh_energy,
            )
        )
    return cases


def tree_cases(n: int) -> list[Case]:
    """Tree evaluation and the FMM/tree-build kernels on a tree of ``n`` points."""
    pos, charge = _cloud(n)
    root = Quad(0.0, 0.0, 1.0)
    log_tree = FlatQuadtree(
        pos, charge, root, kernel="log", order=ORDER, leaf_capacity=4
    )
    coulomb_tree = FlatQuadtree(pos, charge, root, kernel="coulomb", leaf_capacity=8)
    label = f"n={n}"
    exclude = np.arange(n, dtype=np.int64)
    stack = np.empty(512, dtype=np.int64)
    cases = []

    def evaluate_args(tree, with_p):
        moments = (
            (tree.node_child, tree.node_q, tree.node_qabs, tree.node_p, tree.node_s)
            if with_p
            else (tree.node_child, tree.node_qabs, tree.moment_re, tree.moment_im)
        )
        return (
            pos, exclude, tree.pos, tree.charge, tree.perm,
            tree.node_center, tree.node_half, tree.node_start, tree.node_end,
            *moments, THETA_TREE, SOFTENING, stack,
        )  # fmt: skip

    for name, tree, with_p in (
        ("evaluate_quadtree", coulomb_tree, True),
        ("evaluate_log_quadtree", log_tree, False),
    ):
        fixed = evaluate_args(tree, with_p)

        def fresh(_, fixed=fixed):
            return (*fixed, np.zeros(n), np.zeros((2, n)), np.zeros(1, dtype=np.int64))

        cases.append(
            _simple(
                "tree evaluation", name, label, lambda: (), lambda a: a[-3:-1], fresh
            )
        )

    # --- dual-tree traversal (all generations) -------------------------------
    cap = 40 * log_tree.node_count + 64

    def traverse(function, state):
        node_center, node_half = state
        m2l = np.empty((cap, 2), dtype=np.int64)
        p2p = np.empty((cap, 2), dtype=np.int64)
        frontier = np.zeros((1, 2), dtype=np.int64)
        counts = np.zeros(4, dtype=np.int64)
        while len(frontier):
            counts[2] = 0
            nxt = np.empty((cap, 2), dtype=np.int64)
            function(
                node_center, node_half, log_tree.node_start,
                log_tree.node_end, log_tree.node_child, THETA_FMM, frontier,
                len(frontier), nxt, m2l, p2p, counts,
            )  # fmt: skip
            frontier = nxt[: counts[2]].copy()
        traverse.result = (m2l[: counts[0]].copy(), p2p[: counts[1]].copy())

    cases.append(
        Case(
            "FMM",
            "fmm_traverse",
            f"{label} (all generations)",
            lambda: (log_tree.node_center, log_tree.node_half),
            traverse,
            lambda _s: [np.sort(a, axis=0).astype(float) for a in traverse.result],
        )  # fmt: skip
    )

    # CPU lists for the downstream kernels
    traverse(KERNELS["fmm_traverse"], (log_tree.node_center, log_tree.node_half))
    m2l_pairs, p2p_pairs = traverse.result
    n_nodes = log_tree.node_count
    width = ORDER + 1
    binomial = _binomial_table(2 * ORDER + 1)

    def fresh_m2l(_):
        scratch = np.zeros(2 * ORDER + 2)
        return (
            m2l_pairs, len(m2l_pairs), log_tree.node_center, log_tree.moment_re,
            log_tree.moment_im, binomial, scratch, scratch.copy(),
            np.zeros((n_nodes, width)), np.zeros((n_nodes, width)),
        )  # fmt: skip

    cases.append(
        _simple(
            "FMM",
            "fmm_m2l",
            f"{label} ({len(m2l_pairs)} pairs)",
            lambda: (),
            lambda a: a[-2:],
            fresh_m2l,
        )
    )

    local_re, local_im = np.zeros((n_nodes, width)), np.zeros((n_nodes, width))
    KERNELS["fmm_m2l"](*fresh_m2l(None)[:8], local_re, local_im)

    def downward(function, state):
        re, im, node_center, table = state
        scratch = np.zeros(width, dtype=re.dtype)
        for first, last in zip(log_tree.levels[:-2], log_tree.levels[1:-1]):
            function(
                int(first), int(last), node_center, log_tree.node_child,
                table, scratch, scratch.copy(), re, im,
            )  # fmt: skip

    cases.append(
        Case(
            "FMM",
            "fmm_downward",
            f"{label} (all levels)",
            lambda: (local_re.copy(), local_im.copy(), log_tree.node_center, binomial),
            downward,
            lambda s: s[:2],
        )  # fmt: skip
    )

    shifted_re, shifted_im = local_re.copy(), local_im.copy()
    downward(
        KERNELS["fmm_downward"],
        (shifted_re, shifted_im, log_tree.node_center, binomial),
    )

    def fresh_evaluate(_):
        return (
            log_tree.pos, log_tree.charge, log_tree.perm, n_nodes, log_tree.node_center,
            log_tree.node_start, log_tree.node_end, log_tree.node_child,
            shifted_re, shifted_im, p2p_pairs, len(p2p_pairs), SOFTENING, 1,
            np.zeros(n), np.zeros((2, n)), np.zeros(1, dtype=np.int64),
        )  # fmt: skip

    cases.append(
        _simple(
            "FMM",
            "fmm_evaluate",
            f"{label} ({len(p2p_pairs)} P2P pairs)",
            lambda: (),
            lambda a: a[-3:-1],
            fresh_evaluate,
        )
    )

    rng = np.random.default_rng(1)
    pairs = np.unique(np.sort(rng.integers(0, n, (n, 2)), axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].astype(np.int64)

    def fresh_soften(_):
        return (
            pos, charge, pairs, SOFTENING**2, 1,
            np.zeros(n), np.zeros((2, n)), np.zeros(1, dtype=np.int64),
        )  # fmt: skip

    cases.append(
        _simple(
            "FMM",
            "soften_pairs",
            f"{label} ({len(pairs)} pairs)",
            lambda: (),
            lambda a: a[5:7],
            fresh_soften,
        )
    )

    # --- tree build -----------------------------------------------------------
    perm0 = np.arange(n, dtype=np.int64)
    slot_node = np.zeros(n, dtype=np.int64)

    def fresh_quadrants(_):
        return (
            pos,
            perm0,
            slot_node,
            log_tree.node_center,
            log_tree.node_child,
            np.zeros(n, dtype=np.int64),
        )

    cases.append(
        _simple(
            "tree build",
            "quadtree_quadrants",
            label,
            lambda: (),
            lambda a: a[5:],
            fresh_quadrants,
        )
    )

    quadrant = np.zeros(n, dtype=np.int64)
    KERNELS["quadtree_quadrants"](*fresh_quadrants(None)[:5], quadrant)
    below = np.cumsum(np.eye(4, dtype=np.int64)[quadrant], axis=0)
    node_start = log_tree.node_start.copy()
    node_start[log_tree.node_child[0] : log_tree.node_child[0] + 4] = np.concatenate(
        ([0], np.cumsum(below[-1])[:-1])
    )

    def fresh_scatter(_):
        return (
            perm0, slot_node, quadrant, below, node_start, log_tree.node_child,
            np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
        )  # fmt: skip

    cases.append(
        _simple(
            "tree build",
            "quadtree_scatter",
            label,
            lambda: (),
            lambda a: a[6:],
            fresh_scatter,
        )
    )

    def fresh_moments(_):
        return (
            log_tree.pos, log_tree.charge, log_tree.perm, n_nodes, log_tree.node_center,
            log_tree.node_start, log_tree.node_end,
            np.zeros(n_nodes), np.zeros(n_nodes), np.zeros((n_nodes, 2)), np.zeros((n_nodes, 3)),
        )  # fmt: skip

    cases.append(
        _simple(
            "tree build",
            "quadtree_moments",
            f"{label} ({n_nodes} nodes)",
            lambda: (),
            lambda a: a[7:],
            fresh_moments,
        )
    )

    def fresh_log_moments(_):
        return (
            log_tree.pos, log_tree.charge, log_tree.perm, n_nodes, log_tree.node_center,
            log_tree.node_start, log_tree.node_end,
            np.zeros(n_nodes), np.zeros((n_nodes, width)), np.zeros((n_nodes, width)),
        )  # fmt: skip

    cases.append(
        _simple(
            "tree build",
            "log_quadtree_moments",
            f"{label} ({n_nodes} nodes)",
            lambda: (),
            lambda a: a[7:],
            fresh_log_moments,
        )
    )
    return cases


def run(cases: Sequence[Case], repeat: int, dtype: str = "float64") -> list[dict]:
    """Time each case on the CPU and on Metal and print a table row."""
    float32 = dtype == "float32"
    rows = []
    for case in cases:
        gpu_function = metal_kernel(case.name)
        cpu_function = KERNELS[case.name]
        try:
            cpu_time, cpu_out, _ = _time(case, cpu_function, repeat)
            gpu_time, gpu_out, gpu_first = _time(case, gpu_function, repeat)
            row = {
                "family": case.family,
                "kernel": case.name,
                "case": case.label,
                "cpu_ms": cpu_time * 1e3,
                "metal_ms": gpu_time * 1e3,
                "speedup": cpu_time / gpu_time,
                "difference": _difference(gpu_out, cpu_out),
                "first_ms": gpu_first * 1e3,
            }
            if float32:
                # CPU reference on the float32-rounded values isolates the kernel error
                _, ref_out, _ = _time(
                    case,
                    cpu_function,
                    1,
                    transform=lambda s: to_float64(to_float32(s)),
                )
                f32_time, f32_out, f32_first = _time(
                    case, gpu_function, repeat, transform=to_float32
                )
                row.update(
                    metal32_ms=f32_time * 1e3,
                    speedup32=cpu_time / f32_time,
                    gain=gpu_time / f32_time,
                    difference32=_difference(f32_out, ref_out),
                    first32_ms=f32_first * 1e3,
                )
        except Exception as error:  # report and go on
            print(
                f"# {case.name} ({case.label}) failed: {type(error).__name__}: {error}",
                flush=True,
            )
            continue
        rows.append(row)
        if float32:
            print(
                f"| {row['family']} | {row['kernel']} | {row['case']} | {row['cpu_ms']:.3f} | "
                f"{row['metal_ms']:.3f} | {row['metal32_ms']:.3f} | {row['speedup32']:.2f}x | "
                f"{row['gain']:.2f}x | {row['difference']:.1e} | {row['difference32']:.1e} | "
                f"{row['first32_ms']:.0f} |",
                flush=True,
            )
        else:
            print(
                f"| {row['family']} | {row['kernel']} | {row['case']} | {row['cpu_ms']:.3f} | "
                f"{row['metal_ms']:.3f} | {row['speedup']:.2f}x | {row['difference']:.1e} | {row['first_ms']:.0f} |",
                flush=True,
            )
    return rows


def main() -> None:
    """Parse the arguments and print the timing table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--direct-sizes", default="2000,10000,30000")
    parser.add_argument("--tree-size", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help="dtype of the float arrays given to the Metal kernels",
    )
    parser.add_argument("--skip", default="", help="comma-separated families to skip")
    args = parser.parse_args()

    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
    ).stdout.strip()
    print(
        f"machine: {chip or platform.processor()}; repeat={args.repeat}; "
        f"dtype={args.dtype}\n"
    )
    if args.dtype == "float32":
        print(
            "| family | kernel | case | CPU f64 ms | Metal f64 ms | Metal f32 ms | "
            "speed-up (f32) | f32 gain | diff f64 | diff f32 | Metal f32 1st call ms |"
        )
        print(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            flush=True,
        )
    else:
        print(
            "| family | kernel | case | CPU ms | Metal ms | speed-up | max rel. diff | Metal 1st call ms |"
        )
        print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |", flush=True)

    skip = set(filter(None, args.skip.split(",")))
    groups: dict[str, Callable[[], list[Case]]] = {
        "direct": lambda: direct_cases(list(map(int, args.direct_sizes.split(",")))),
        "tree": lambda: tree_cases(args.tree_size),
    }
    all_cases: list[Case] = []
    for key, build in groups.items():
        if key in skip:
            continue
        cases = build()
        all_cases += cases
        run(cases, args.repeat, args.dtype)

    if "tree" not in skip:
        n = args.tree_size
        pos, charge = _cloud(n)
        tree = FlatQuadtree(
            pos, charge, Quad(0.0, 0.0, 1.0), kernel="coulomb", leaf_capacity=8
        )
        exclude = np.arange(n, dtype=np.int64)
        start = time.perf_counter()
        tree.evaluate(pos, exclude=exclude, theta=THETA_TREE, softening=SOFTENING)
        print(
            f"\nFlatQuadtree.evaluate end-to-end on CPU (n={n}, coulomb): "
            f"{(time.perf_counter() - start) * 1e3:.1f} ms"
        )
        log_tree = FlatQuadtree(
            pos, charge, Quad(0.0, 0.0, 1.0), kernel="log", order=ORDER, leaf_capacity=4
        )
        start = time.perf_counter()
        log_tree.fmm(theta=THETA_FMM, softening=SOFTENING, potential=True)
        print(
            f"FlatQuadtree.fmm end-to-end on CPU (n={n}, log, order {ORDER}): "
            f"{(time.perf_counter() - start) * 1e3:.1f} ms"
        )


if __name__ == "__main__":
    main()
