---
title: Building the docs
description: How this documentation site is built from Markdown pages, notebooks and docstrings.
---

The site in `docs/` is built with [Astro](https://astro.build/) and
[Starlight](https://starlight.astro.build/). It has four kinds of pages:

- **Hand-written pages** in `docs/src/content/docs/`, as Markdown or MDX.
- **Tutorials**, converted from the Jupyter notebooks in `tutorials/` by `starlight-notebooks`.
  Each notebook is executed and its outputs (text, images, tables) are written into an MDX page.
- **Examples**, converted from the scripts in `examples/` by `docs/tools/examples.py`. Each script
  is run and its figures are published as interactive Plotly charts, with captions taken from its
  own docstring.
- **API reference**, generated from the docstrings by
  [starlight-pydocs](https://ewels.github.io/starlight-pydocs/), one page per module.

The configuration (`defineDocs` in `docs/astro.config.mjs`), the look, the header, the notebook
tool and the build workflow come from the shared
[starlight-theme](https://github.com/max-models/starlight-theme) of the max-models projects;
change them there, and update the version in `docs/package.json` and
`.github/workflows/docs.yml` to pick up a new release.

## Requirements

- Node 22 or newer.
- The package installed with the `docs` extra: `pip install -e ".[docs]"` (this includes the
  `examples` extra, since the example scripts plot with Plotly through `plasma-plots`).

## Commands

```bash
make docs-install     # npm packages and the Python docs extra
make docs-notebooks   # execute tutorials/*.ipynb and convert them to pages
make docs-examples    # run examples/*.py (--quick) and convert them to pages
make docs-dev         # live preview at http://localhost:4321/tree-d-3d/
make docs-build       # the static site in docs/dist, as in CI
make docs-preview     # serve docs/dist
make docs-clean       # remove generated pages and build output
```

`make docs-notebooks` only reruns a notebook whose cells changed since the last run, and
`make docs-examples` only reruns a script whose source changed since its last run; pass `FORCE=1`
to rerun all of them.

## Writing an example

Add a script to `examples/`. Draw each figure with Plotly (`plotly.graph_objects`) and save it
with `plasma_plots.save_figure(fig, output / name)`, which writes `<name>.plotly.json` (embedded
as the interactive chart), `<name>.png` (a static fallback, not used by the docs) and `<name>.html`
(a standalone copy). An animation is a figure with `frames` and a slider/play button in its
`layout` (see `plasma_plots.heatmap_movie` for the pattern, or build `go.Frame`s by hand); the page
shows it under "Animations" instead of "Figures".

The module docstring drives the page: the first line is the title, the paragraph before the
figure list is the description, and a numbered list of `` `name`: caption `` becomes the page's
figures, in that order, `name` matching the stem passed to `save_figure`. The script must accept
`--output DIR` and write its figures there, and should accept `--quick` to run a smaller, faster
version for the docs build.

## Writing tutorials

Put a notebook in `tutorials/`. Its first `# Heading` becomes the page title and the first
paragraph its description. Cell tags control the output:

| Tag                | Effect                                   |
| ------------------ | ---------------------------------------- |
| `remove-cell`      | the cell is left out of the page         |
| `hide-input`       | the code is hidden, the outputs are kept |
| `fold-input`       | the code is in a closed "Show code" box  |
| `remove-output`    | the outputs are left out                 |
| `raises-exception` | an error output is shown, not a failure  |

Math in `$...$` and `$$...$$` is rendered with KaTeX.
