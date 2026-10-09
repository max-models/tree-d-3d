"""Compile Pyccel kernel modules once and reuse the result across processes.

:func:`compile_cached` builds a kernel module with the ``pyccel compile`` command
line interface, in a subprocess, once. It keys a build on the kernel source, the
Pyccel and Python versions and the platform, and stores it under ``TREE_D_3D_KERNEL_CACHE``
(default ``~/.cache/tree-d-3d/kernels``). Later processes import the stored
extension directly.

Kernels are built with OpenMP, so the loops marked ``#$ omp parallel for`` run on
all cores (limit them with ``OMP_NUM_THREADS``; under MPI set it to the cores per
rank). ``TREE_D_3D_OPENMP=0`` builds them serial, except the kernels that call the OpenMP
runtime; the setting is part of the cache key.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType


def cache_root() -> Path:
    """Directory holding compiled kernel builds."""
    default = Path.home() / ".cache" / "tree-d-3d" / "kernels"
    return Path(os.environ.get("TREE_D_3D_KERNEL_CACHE", default))


def _fallback_root() -> Path:
    return Path(tempfile.gettempdir()) / "tree-d-3d-kernels"


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def openmp_enabled() -> bool:
    """Whether kernels are built with OpenMP (``TREE_D_3D_OPENMP=0`` turns it off)."""
    return os.environ.get("TREE_D_3D_OPENMP", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


#: Bumped when the layout of a cached build changes (it was an ``epyccel`` folder
#: before the ``pyccel compile`` command line was used).
_LAYOUT = "pyccel-cli-1"


def _uses_openmp_runtime(source: Path) -> bool:
    """Whether the kernel calls the OpenMP runtime (thread ids, thread counts).

    Such a kernel only compiles with OpenMP, so it is built with it even if
    ``TREE_D_3D_OPENMP=0``; ``OMP_NUM_THREADS=1`` then runs it on one thread.
    """
    return "pyccel.stdlib.internal.openmp" in source.read_text()


def _build_key(module: ModuleType, language: str) -> str:
    digest = hashlib.sha256()
    for part in (
        inspect.getsource(module),
        language,
        _LAYOUT,
        f"openmp={openmp_enabled() or _uses_openmp_runtime(Path(inspect.getsourcefile(module) or ''))}",
        importlib.metadata.version("pyccel"),
        sys.version,
        platform.platform(),
    ):
        digest.update(part.encode())
    return digest.hexdigest()[:20]


def _load_extension(directory: Path, stem: str) -> ModuleType | None:
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(stem, path)
        if spec is None or spec.loader is None:
            continue
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
        return extension
    return None


def _pyccel_command() -> list[str]:
    """The ``pyccel`` executable of the running environment."""
    beside_python = Path(sys.executable).parent / "pyccel"
    found = str(beside_python) if beside_python.is_file() else shutil.which("pyccel")
    if found is None:
        raise FileNotFoundError("the pyccel command line tool is not installed")
    return [found]


def _build(module: ModuleType, language: str, staging: Path) -> Path:
    """Run ``pyccel compile`` on a copy of the module's file; return the build folder."""
    source = inspect.getsourcefile(module)
    if source is None:
        raise FileNotFoundError(f"No source file for {module.__name__}")
    stem = module.__name__.rsplit(".", 1)[-1]
    sources = staging / "src"
    sources.mkdir()
    shutil.copy(source, sources / f"{stem}.py")
    output = staging / "build"
    command = [
        *_pyccel_command(),
        "compile",
        str(sources / f"{stem}.py"),
        "--language",
        language,
        "--output",
        str(output),
    ]
    if openmp_enabled() or _uses_openmp_runtime(Path(source)):
        command.append("--openmp")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}):\n"
            f"{result.stdout}{result.stderr}"
        )
    return output


def compile_cached(module: ModuleType, *, language: str = "c") -> ModuleType:
    """Return the compiled form of ``module``, building it on first use.

    Raises ``RuntimeError`` with the output of ``pyccel compile`` if the build
    fails; callers decide on a fallback.
    """
    stem = module.__name__.rsplit(".", 1)[-1]
    name = f"{stem}-{_build_key(module, language)}"
    roots = [cache_root(), _fallback_root()]
    for root in roots:
        if (root / name).is_dir():
            extension = _load_extension(root / name, stem)
            if extension is not None:
                return extension
    # An unwritable cache (e.g. a CI container whose HOME belongs to another
    # user) must not force the slow Python kernels, so build in /tmp instead.
    target = next((root / name for root in roots if _writable(root)), None)
    if target is None:
        raise PermissionError(f"No writable kernel cache among {roots}")
    staging = Path(tempfile.mkdtemp(prefix=f".{stem}-", dir=target.parent))
    try:
        build = _build(module, language, staging)
        try:
            # Atomic, so concurrent builders never see a partial directory.
            os.rename(build, target)
        except OSError:
            pass  # Another process published the same build first.
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    extension = _load_extension(target, stem)
    if extension is None:
        raise ImportError(f"Compiled kernel for {module.__name__} not found")
    return extension
