"""Run the scripts in ``examples/`` and convert their figures into Starlight pages.

Each figure is a Plotly figure saved with ``plasma_plots.save_figure(fig, output / name)``,
which writes ``<name>.plotly.json`` (embedded as an interactive chart), ``<name>.png`` (its
static fallback) and ``<name>.html`` (a standalone copy, not used by the docs).

``run``
    Runs each example with ``--quick --output docs/.cache/examples/<slug>``. A script is rerun
    only when its source hash changed (``--force`` reruns all of them, ``--full`` drops
    ``--quick``). The examples run in parallel (``--jobs N``, by default one per core, longest
    first), and ``--shard I/N`` runs only the I-th of N balanced groups, so that a CI matrix can
    spread them over several machines (the weights are in ``example_weights.json``);
    ``--dry-run`` prints the plan.
``convert``
    Writes ``docs/src/content/docs/examples/<slug>.mdx`` from the figures of a run, copying their
    ``.plotly.json`` into ``docs/public/examples/<slug>/``. Captions and the intro text come from
    the script's module docstring. A script that has not been run is skipped (``--strict`` fails
    on it).
``clean``
    Removes the cache, the generated pages and the public assets.

Usage: python docs/tools/examples.py <run|convert|clean> [options]
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
EXAMPLES = REPO / "examples"
CACHE = DOCS / ".cache" / "examples"
CONTENT = DOCS / "src" / "content" / "docs" / "examples"
PUBLIC = DOCS / "public" / "examples"
COMPONENTS = "../../../components"


def slug_of(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")


def scripts() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def script_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# Parsing the module docstring ---------------------------------------------------------- #


def flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def rst_to_md(text: str) -> str:
    """RST's ``double backtick`` code spans to Markdown's `single backtick`."""
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def parse_docstring(doc: str) -> dict:
    """Return title, intro/notes markdown, the usage line and the ordered figure captions."""
    paragraphs = re.split(r"\n\s*\n", doc.strip("\n"))
    title = flatten(paragraphs[0])
    figures: list[tuple[str, str]] = []
    intro: list[str] = []
    notes: list[str] = []
    usage = None
    seen_list = False
    for paragraph in paragraphs[1:]:
        stripped = paragraph.strip()
        if stripped.startswith("Usage::"):
            continue
        if re.match(r"^\s*python\s+\S+", stripped):
            usage = flatten(stripped)
            continue
        if re.match(r"^\d+\.\s+``", stripped):
            seen_list = True
            for item in re.split(r"\n(?=\d+\.\s)", stripped):
                item = re.sub(r"^\d+\.\s+", "", flatten(item))
                match = re.match(r"``([\w-]+)``:\s*(.*)", item)
                if match:
                    figures.append((match.group(1), match.group(2)))
            continue
        (notes if seen_list else intro).append(flatten(stripped))
    return {
        "title": title,
        "intro": "\n\n".join(intro),
        "notes": "\n\n".join(notes),
        "usage": usage,
        "figures": figures,
    }


def docstring_of(script: Path) -> dict:
    module = ast.parse(script.read_text())
    doc = ast.get_docstring(module) or ""
    return parse_docstring(doc)


# Running --------------------------------------------------------------------------------- #

WEIGHTS = Path(__file__).with_name("example_weights.json")
#: Libraries that start one thread per core: with parallel examples each gets its share.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def weights() -> dict[str, float]:
    """Estimated minutes of each example by slug (1 for a script that is not listed)."""
    table = json.loads(WEIGHTS.read_text()) if WEIGHTS.exists() else {}
    return {k: float(v) for k, v in table.items() if not k.startswith("_")}


def parse_shard(text: str) -> tuple[int, int]:
    """``"2/4"`` to (2, 4): the second of four shards."""
    try:
        index, count = (int(part) for part in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"a shard is I/N, such as 2/4, not {text!r}"
        ) from None
    if not 1 <= index <= count:
        raise argparse.ArgumentTypeError(f"shard {text}: need 1 <= I <= N")
    return index, count


def shard_of(paths: list[Path], index: int, count: int) -> list[Path]:
    """The ``index``-th of ``count`` groups of ``paths``, balanced by their weights.

    Greedy: the heaviest script first, each into the group with the least weight so far. It
    depends only on the slugs and the weights file, so every shard job computes the same split.
    """
    table = weights()
    load = [0.0] * count
    groups: list[list[Path]] = [[] for _ in range(count)]
    for path in sorted(paths, key=lambda p: (-table.get(slug_of(p), 1.0), slug_of(p))):
        lightest = min(range(count), key=lambda g: (load[g], g))
        groups[lightest].append(path)
        load[lightest] += table.get(slug_of(path), 1.0)
    return groups[index - 1]


def _run_one(script: Path, args: argparse.Namespace, env: dict[str, str]) -> dict:
    """Run one example in its own process, keeping its output for the caller to print."""
    slug = slug_of(script)
    out_dir = CACHE / slug
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    command = [args.python, str(script), "--output", str(out_dir)]
    if not args.full:
        command.append("--quick")
    start = time.monotonic()
    done = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "slug": slug,
        "script": script,
        "ok": done.returncode == 0,
        "returncode": done.returncode,
        "seconds": time.monotonic() - start,
        "log": done.stdout,
    }


def cmd_run(args: argparse.Namespace) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    chosen = scripts()
    if args.shard:
        chosen = shard_of(chosen, *args.shard)
    table = weights()
    pending = []
    for script in chosen:
        slug = slug_of(script)
        key_file = CACHE / f"{slug}.key"
        if (
            not args.force
            and (CACHE / slug).exists()
            and key_file.exists()
            and key_file.read_text() == script_hash(script)
        ):
            print(f"[examples] {slug}: up to date")
        else:
            pending.append(script)
    # the longest first, so that the last to start is not the longest to finish
    pending.sort(key=lambda p: (-table.get(slug_of(p), 1.0), slug_of(p)))
    jobs = args.jobs or os.cpu_count() or 1
    jobs = max(1, min(jobs, len(pending)))
    shard = f" (shard {args.shard[0]}/{args.shard[1]})" if args.shard else ""
    print(f"[examples]{shard} {len(pending)} to run, {jobs} at a time")
    for script in pending:
        print(
            f"[examples]   {slug_of(script)} (about {table.get(slug_of(script), 1.0):g} min)"
        )
    if args.dry_run or not pending:
        return

    env = dict(os.environ)
    share = str(max(1, (os.cpu_count() or 1) // jobs))
    for name in THREAD_VARIABLES:
        env.setdefault(name, share)
    github = bool(os.environ.get("GITHUB_ACTIONS"))
    lock = threading.Lock()
    results = []

    def finish(result: dict) -> None:
        slug = result["slug"]
        with lock:  # one example's output at a time, in a folding group on GitHub
            print(
                f"::group::{slug}" if github else f"[examples] {slug}: output",
                flush=True,
            )
            print(result["log"].rstrip(), flush=True)
            if github:
                print("::endgroup::", flush=True)
            error_file = CACHE / f"{slug}.error"
            if result["ok"]:
                error_file.unlink(missing_ok=True)
                (CACHE / f"{slug}.key").write_text(script_hash(result["script"]))
                (CACHE / f"{slug}.time").write_text(f"{result['seconds']:.1f}")
                print(
                    f"[examples] {slug}: done in {result['seconds'] / 60:.1f} min",
                    flush=True,
                )
            else:
                error_file.write_text(
                    f"exit status {result['returncode']}\n{result['log'][-2000:]}"
                )
                print(
                    f"[examples] {slug}: FAILED (exit status {result['returncode']})",
                    flush=True,
                )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_one, script, args, env) for script in pending]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            finish(result)
            results.append(result)
    wall = time.monotonic() - started
    work = sum(r["seconds"] for r in results)
    print(
        f"[examples] {len(results)} examples, {work / 60:.1f} min of work in {wall / 60:.1f} min"
    )
    for r in sorted(results, key=lambda r: -r["seconds"]):
        print(
            f"[examples]   {r['slug']:<32} {r['seconds'] / 60:5.1f} min"
            f"{'' if r['ok'] else '  FAILED'}"
        )
    failed = [r["slug"] for r in results if not r["ok"]]
    if failed:
        print(f"[examples] {len(failed)} failed: {', '.join(failed)}")
        sys.exit(1)


# Conversion -------------------------------------------------------------------------------- #


def _yaml_string(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def has_frames(path: Path) -> bool:
    """Tell whether a ``.plotly.json`` figure is animated."""
    try:
        return bool(json.loads(path.read_text()).get("frames"))
    except (json.JSONDecodeError, OSError):
        return False


def convert_example(script: Path, order: int) -> list[str]:
    """Write the MDX page of one example; return problems found along the way."""
    slug = slug_of(script)
    out_dir = CACHE / slug
    problems = []
    if not out_dir.exists():
        return [f"{slug}: not run"]
    info = docstring_of(script)
    info["intro"] = rst_to_md(info["intro"])
    info["notes"] = rst_to_md(info["notes"])
    info["figures"] = [(n, rst_to_md(c)) for n, c in info["figures"]]
    missing = [
        n for n, _ in info["figures"] if not (out_dir / f"{n}.plotly.json").exists()
    ]
    for name in missing:
        problems.append(f"{slug}: missing figure {name}.plotly.json")
    present = [(n, c) for n, c in info["figures"] if n not in missing]
    animations = [
        (n, c) for n, c in present if has_frames(out_dir / f"{n}.plotly.json")
    ]
    static = [(n, c) for n, c in present if (n, c) not in animations]

    public_dir = PUBLIC / slug
    shutil.rmtree(public_dir, ignore_errors=True)
    public_dir.mkdir(parents=True)
    for name, _ in present:
        shutil.copy2(
            out_dir / f"{name}.plotly.json", public_dir / f"{name}.plotly.json"
        )

    github = (
        f"https://github.com/max-models/tree-d-3d/blob/devel/examples/{script.name}"
    )
    frontmatter = [
        "---",
        f"title: {_yaml_string(info['title'])}",
        f"description: {_yaml_string(info['intro'] or info['title'])}",
        "sidebar:",
        f"  order: {order}",
        "---",
        "",
        f"import PlotlyChart from '{COMPONENTS}/PlotlyChart.astro';",
        "",
    ]
    body = []
    if info["intro"]:
        body.append(info["intro"])
    install = 'pip install "tree-d-3d[examples]"'
    run_line = (
        f"`{info['usage']}`" if info["usage"] else f"`python examples/{script.name}`"
    )
    body.append(
        f"Source: [`examples/{script.name}`]({github}). Run it yourself with {run_line} "
        f"(needs `{install}`)."
    )

    def chart(name: str, caption: str) -> str:
        # A JSX expression (``{...}``), not a plain quoted attribute: a caption containing a
        # literal ``"`` (as some do) is invalid MDX in a bare ``title="..."`` attribute, since
        # JSX has no backslash-escape for a quote inside one -- unlike the JSON string
        # ``_yaml_string`` returns, which doubles as a valid JS string literal inside ``{...}``.
        title = _yaml_string(caption or name)
        tag = f'<PlotlyChart src="/examples/{slug}/{name}.plotly.json" title={{{title}}} />'
        return f"{tag}\n\n*{caption}*" if caption else tag

    if animations:
        body.append("## Animations")
        body += [chart(name, caption) for name, caption in animations]
    if static:
        body.append("## Figures")
        body += [chart(name, caption) for name, caption in static]
    if info["notes"]:
        body.append(info["notes"])

    CONTENT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(frontmatter) + "\n" + "\n\n".join(body) + "\n"
    (CONTENT / f"{slug}.mdx").write_text(text)
    print(f"[examples] {slug}: {len(present)} figures -> {CONTENT / slug}.mdx")
    return problems


def cmd_convert(args: argparse.Namespace) -> None:
    for stale in CONTENT.glob("*.mdx"):
        stale.unlink()
    problems = []
    for order, script in enumerate(scripts(), start=1):
        problems += convert_example(script, order)
    for problem in problems:
        print(f"[examples] warning: {problem}")
    if args.strict and problems:
        sys.exit(1)


def cmd_clean(args: argparse.Namespace) -> None:
    shutil.rmtree(CACHE, ignore_errors=True)
    shutil.rmtree(PUBLIC, ignore_errors=True)
    for page in CONTENT.glob("*.mdx"):
        page.unlink()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the example scripts whose source changed")
    run.add_argument("--force", action="store_true", help="rerun every example")
    run.add_argument(
        "--full", action="store_true", help="drop --quick (the real runtime)"
    )
    run.add_argument(
        "--python", default=sys.executable, help="interpreter to run scripts with"
    )
    run.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help="examples to run at a time (default: one per core)",
    )
    run.add_argument(
        "--shard",
        type=parse_shard,
        metavar="I/N",
        help="run only the I-th of N groups of examples, balanced by example_weights.json",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="print what would run and stop"
    )
    run.set_defaults(func=cmd_run)
    convert = sub.add_parser("convert", help="write the MDX pages and assets")
    convert.add_argument(
        "--strict", action="store_true", help="fail on missing or unrun examples"
    )
    convert.set_defaults(func=cmd_convert)
    clean = sub.add_parser("clean", help="remove the cache and generated files")
    clean.set_defaults(func=cmd_clean)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
