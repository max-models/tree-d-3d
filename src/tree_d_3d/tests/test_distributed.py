"""MPI runs agree with serial ones: thread ranks (`maybempi.testing`), and mpirun if present."""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from maybempi import collectives
from maybempi.testing import run_ranks
from simgeometry import Circle, Polygon

from tree_d_3d import (
    BoundaryIntegralElectrodes,
    ChargedBody,
    ElectrostaticSimulation,
    Face,
    FlatQuadtree,
    Quad,
    conductors_from_shape,
)

RANKS = 3


def _cloud(n, seed=1):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-0.9, 0.9, (2, n))
    velocities = rng.normal(0.0, 0.1, (2, n))
    return positions, velocities, rng.choice([-1.0, 1.0], n)


@pytest.mark.parametrize("compiled", [True, False])
def test_tree_moments_and_fmm_on_ranks(compiled):
    positions, _, charges = _cloud(3000)
    # a fixed, neutral pair on one spot drops out of the FMM's near field
    positions[:, :2] = 0.2
    charges[:2] = 0.0

    def tree(comm):
        return FlatQuadtree(
            positions,
            charges,
            Quad(0, 0, 1),
            kernel="log",
            order=10,
            leaf_capacity=16,
            compiled=compiled,
            comm=comm,
        )

    serial = tree(None)
    expected = serial.fmm(theta=0.5) if compiled else None

    def work(comm):
        mine = tree(comm)
        moments = (mine.moment_re.copy(), mine.node_qabs.copy())
        mine.set_charges(-charges)
        flipped = mine.moment_im.copy()
        mine.set_charges(charges)
        return moments, flipped, mine.fmm(theta=0.5) if compiled else None

    for (moment_re, qabs), flipped, fields in run_ranks(RANKS, work):
        assert np.allclose(moment_re, serial.moment_re, rtol=1e-12, atol=1e-12)
        assert np.allclose(qabs, serial.node_qabs, rtol=1e-12)
        assert np.allclose(flipped, -serial.moment_im, rtol=1e-12, atol=1e-12)
        if compiled:
            assert expected is not None and fields is not None
            for a, b in zip(fields, expected):
                assert np.allclose(a, b, rtol=1e-10, atol=1e-10)
    # coulomb moments too
    coulomb = FlatQuadtree(positions, charges, Quad(0, 0, 1), compiled=compiled)
    for node_s in run_ranks(
        2,
        lambda comm: (
            FlatQuadtree(
                positions, charges, Quad(0, 0, 1), compiled=compiled, comm=comm
            ).node_s
        ),
    ):
        assert np.allclose(node_s, coulomb.node_s, rtol=1e-12, atol=1e-14)


def test_fmm_reports_coincident_charges_on_every_rank():
    positions = np.array([[0.1, 0.1, 0.5], [0.2, 0.2, -0.3]])

    def work(comm):
        tree = FlatQuadtree(
            positions, np.ones(3), Quad(0, 0, 1), kernel="log", comm=comm
        )
        with pytest.raises(ValueError):
            tree.fmm()
        return True

    assert all(run_ranks(2, work))


def _simulation(comm, n, method, kernel, steps=4):
    positions, velocities, charges = _cloud(n)
    mine = collectives.block(n, comm)
    bodies = [
        ChargedBody(*positions[:, i], *velocities[:, i], charge=charges[i])
        for i in range(mine.start, mine.stop)
    ]
    if comm is None or comm.Get_rank() == 0:
        bodies[0].fixed = True
    sim = ElectrostaticSimulation(
        bodies,
        Quad(0, 0, 1),
        k_e=1e-4,
        dt=1e-3,
        softening=1e-3,
        kernel=kernel,
        method=method,
        theta=0.5,
        absorbing_walls={"x_max": 0.95},
        comm=comm,
    )
    for _ in range(steps):
        sim.step()
    sim.compute_fields(include_fixed=True)
    positions, _ = collectives.gather_columns(comm, sim.positions)
    return {
        "positions": positions,
        "kinetic": sim.get_kinetic_energy(),
        "potential": sim.get_potential_energy(),
        "exact": sim.get_potential_energy(exact=True),
        "charge": sim.get_total_charge(),
        "momentum": np.asarray(sim.get_momentum()),
        "centre": np.asarray(sim.get_center_of_mass()),
        "angular": sim.get_angular_momentum(),
        "count": collectives.allsum(comm, sim.num_particles),
        "tree": sim.build_tree() is not None,
    }


@pytest.mark.parametrize(
    ("method", "kernel"),
    [
        ("direct", "log"),
        ("tree", "coulomb"),
        ("tree", "log"),
        ("fmm", "log"),
        ("auto", "log"),
    ],
)
def test_simulation_on_ranks(method, kernel):
    serial = _simulation(None, 400, method, kernel)
    for result in run_ranks(RANKS, lambda comm: _simulation(comm, 400, method, kernel)):
        assert result["count"] == serial["count"]
        assert result["tree"] == serial["tree"]
        assert np.allclose(result["positions"], serial["positions"], atol=1e-12)
        for key in ("kinetic", "potential", "exact", "charge", "angular"):
            assert result[key] == pytest.approx(serial[key], rel=1e-9, abs=1e-15)
        for key in ("momentum", "centre"):
            assert np.allclose(result[key], serial[key], rtol=1e-9, atol=1e-15)


def test_simulation_with_an_empty_rank():
    """A rank without particles still takes part in every collective."""

    def work(comm):
        positions, velocities, charges = _cloud(50)
        bodies = (
            []
            if comm.Get_rank() == 1
            else [
                ChargedBody(*positions[:, i], *velocities[:, i], charge=charges[i])
                for i in range(50)
            ]
        )
        sim = ElectrostaticSimulation(
            bodies, Quad(0, 0, 1), k_e=1e-4, dt=1e-3, softening=1e-3, comm=comm
        )
        sim.step()
        return collectives.allsum(comm, sim.num_particles), sim.get_potential_energy(
            exact=True
        )

    (count, energy), (_, same) = run_ranks(2, work)
    assert count == 50 and energy == pytest.approx(same)


def _shapes(panel):
    shapes = conductors_from_shape(Circle(midpoint=(0, 0), radius=1.0), 1.0, panel)
    box = Polygon(vertices=((1.5, -1), (3, -1), (3, 1), (1.5, 1)))
    return shapes + conductors_from_shape(box, -1.0, panel)


def _panels(comm, solver, images):
    rng = np.random.default_rng(3)
    sources = np.stack((rng.uniform(1.05, 1.45, 300), rng.uniform(-1.4, 1.4, 300)))
    charges = rng.choice([-1.0, 1.0], 300) * 1e-3
    probes = rng.uniform(-1.4, 3.5, (2, 200))
    electrodes = BoundaryIntegralElectrodes(
        _shapes(0.05), solver=solver, k_e=1.0, comm=comm, **images
    )
    electrodes.solve(sources, charges)
    return (
        electrodes.charge,
        np.vstack(electrodes.probe(probes)),
        np.vstack(electrodes.panel_field(probes)),
        np.vstack(electrodes.force_on_sources()),
        electrodes.tabulate((-1.4, -1.4), (3.5, 1.4), (5, 4)).field,
        electrodes.iterations,
    )


@pytest.mark.parametrize("solver", ["direct", "gmres"])
@pytest.mark.parametrize(
    "images", [{}, {"period": 3.0}, {"face": Face("x", -1.5, "dirichlet", 0.5)}]
)
def test_panel_solver_on_ranks(solver, images):
    serial = _panels(None, solver, images)
    for result in run_ranks(RANKS, lambda comm: _panels(comm, solver, images)):
        assert result[-1] == serial[-1]
        for a, b in zip(result[:-1], serial[:-1]):
            assert np.allclose(a, b, rtol=1e-11, atol=1e-11 * np.abs(b).max())


def test_a_probe_with_fewer_points_than_ranks():
    def work(comm):
        electrodes = BoundaryIntegralElectrodes(_shapes(0.1), k_e=1.0, comm=comm)
        electrodes.solve()
        return electrodes.probe(np.array([[0.0, 4.0], [3.0, 0.0]]))

    serial = BoundaryIntegralElectrodes(_shapes(0.1), k_e=1.0)
    serial.solve()
    expected = serial.probe(np.array([[0.0, 4.0], [3.0, 0.0]]))
    for phi, field in run_ranks(RANKS, work):
        assert np.allclose(phi, expected[0]) and np.allclose(field, expected[1])


_MPI_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from maybempi import MPI
    from maybempi import collectives
    from tree_d_3d import ChargedBody, ElectrostaticSimulation, Quad

    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(1)
    pos, q = rng.uniform(-0.9, 0.9, (2, 300)), rng.choice([-1.0, 1.0], 300)
    mine = collectives.block(300, comm)
    bodies = [ChargedBody(*pos[:, i], charge=q[i]) for i in range(mine.start, mine.stop)]
    sim = ElectrostaticSimulation(
        bodies, Quad(0, 0, 1), k_e=1e-4, dt=1e-3, softening=1e-3, kernel="log",
        method="fmm", comm=comm,
    )
    sim.step()
    positions, _ = collectives.gather_columns(comm, sim.positions)
    energy = sim.get_total_energy()  # collective: every rank calls it
    if comm.Get_rank() == 0:
        print(comm.Get_size(), float(energy), float(positions.sum()))
    """
)


def _mpi4py_available() -> bool:
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    shutil.which("mpirun") is None or not _mpi4py_available(),
    reason="needs mpirun and mpi4py",
)
def test_mpirun_matches_serial(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(_MPI_SCRIPT)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
    )
    serial = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True
    )
    parallel = subprocess.run(
        ["mpirun", "--oversubscribe", "-n", "2", sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert serial.returncode == 0, serial.stderr
    assert parallel.returncode == 0, parallel.stderr
    size, energy, total = parallel.stdout.split()
    _, serial_energy, serial_total = serial.stdout.split()
    assert size == "2"
    assert float(energy) == pytest.approx(float(serial_energy), rel=1e-10)
    assert float(total) == pytest.approx(float(serial_total), rel=1e-12)
