"""Run the scripts in ``examples/`` and convert their figures into Starlight pages.

Each figure is a Plotly figure saved with ``plasma_plots.save_figure(fig, output / name)``,
which writes ``<name>.plotly.json`` (embedded as an interactive chart), ``<name>.png`` (its
static fallback) and ``<name>.html`` (a standalone copy, not used by the docs).

``run``
    Runs each example with ``--quick --output docs/.cache/examples/<slug>``. A script is rerun
    only when its source hash changed (``--force`` reruns all of them, ``--full`` drops
    ``--quick``).
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
import hashlib
import json
import re
import shutil
import subprocess
import sys
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


def cmd_run(args: argparse.Namespace) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    failed = []
    for script in scripts():
        slug = slug_of(script)
        out_dir = CACHE / slug
        key_file = CACHE / f"{slug}.key"
        error_file = CACHE / f"{slug}.error"
        key = script_hash(script)
        if (
            not args.force
            and out_dir.exists()
            and key_file.exists()
            and key_file.read_text() == key
        ):
            print(f"[examples] {slug}: up to date")
            continue
        print(f"[examples] {slug}: running", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        command = [args.python, str(script), "--output", str(out_dir)]
        if not args.full:
            command.append("--quick")
        try:
            subprocess.run(command, cwd=REPO, check=True)
            error_file.unlink(missing_ok=True)
            key_file.write_text(key)
        except subprocess.CalledProcessError as exc:
            error_file.write_text(str(exc))
            failed.append(slug)
            print(f"[examples] {slug}: FAILED: {exc}")
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
