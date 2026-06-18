# Embed Jupyter Notebooks into the Sphinx Docs via nbsphinx

## Context

The project keeps 14 Jupyter notebooks under `/examples/notebooks/` (one per workflow:
object detection, distillation, semantic/panoptic/instance segmentation, exports,
YOLO/RF-DETR/ultralytics integrations, etc.). They are currently surfaced from the docs
only as external "Open in Colab" links — there is no embedded, browsable version inside
the rendered Sphinx site.

The goal is to render the notebooks as first-class pages of the docs so users can read
them top-to-bottom without leaving the site. The Colab links stay (they remain the way
to run the notebooks). nbsphinx is the chosen extension; pandoc ships as a Python dep
(`pypandoc-binary`), so no system install is needed.

The notebooks all have **cleared outputs** and contain `!pip install`, multi-GB dataset
downloads, and 10–1000-epoch GPU training, so executing them at build time is not
viable. They will be rendered as-is (code + markdown only, no outputs).

## Approach

Add nbsphinx, configure it to never execute, and copy notebooks from
`/examples/notebooks/` into the docs source tree inside the existing `prebuild.py` step
— matching how `changelog.md` and the `_auto/` config dumps are already generated. The
existing `tutorials/colab/index.md` page is repurposed as the index/toctree of rendered
notebooks.

### 1. Dependencies — `pyproject.toml` (lines 72–81)

Add two entries to the docs block:

```toml
"nbsphinx>=0.9",
"pypandoc-binary>=1.13",
```

Keep `sphinx>=7.1` constraint (nbsphinx 0.9 is compatible). No other deps change.

### 2. Sphinx config — `docs/source/conf.py`

- Add `"nbsphinx"` to `extensions` (after `"myst_parser"`).
- Add nbsphinx settings near the bottom of the file:
  ```python
  # -- nbsphinx ---------------------------------------------------------------
  nbsphinx_execute = "never"        # notebooks render as-is; outputs come from Colab
  nbsphinx_allow_errors = True      # tolerate stale error-state cells
  nbsphinx_kernel_name = "python3"  # 2 notebooks lack kernelspec — give them a default
  ```
- Add an `nbsphinx_prolog` that injects a small Colab badge at the top of every rendered
  notebook, linking back to the canonical version on GitHub/Colab (uses the
  `{{ env.docname }}` jinja variable to derive the path). This keeps the "Run this"
  affordance on the page.
- Extend `exclude_patterns` with `"**.ipynb_checkpoints"` to silence the warning class
  nbsphinx emits when checkpoints leak in.

### 3. Prebuild copy step — `docs/prebuild.py`

Add one new function and one call in `main()`:

- `copy_notebooks(dest_dir: Path) -> None` — globs
  `PROJECT_ROOT / "examples" / "notebooks" / "*.ipynb"`, copies each into `dest_dir`.
  Use the same "skip-write-if-unchanged" pattern already used by `build_changelog_html`
  (read existing file, compare, only write on diff) to avoid Sphinx rebuild loops when
  running incrementally.
- Call it from `main()` with `dest_dir = DOCS_DIR / "tutorials" / "colab"`.

Reuse the existing module-level constants (`PROJECT_ROOT`, `DOCS_DIR`). No new CLI args.

### 4. Index page — `docs/source/tutorials/colab/index.md`

Replace the current hand-maintained bullet list of 4 Colab links with a toctree that
points at every copied notebook. Page title can stay "Google Colab Notebooks" (URL is
stable) or be renamed to "Notebooks" — recommend renaming the heading but keeping the
filename and `(colab)=` anchor so existing links don't break.

````markdown
(colab)=

# Notebooks

The following notebooks demonstrate how to use LightlyTrain end-to-end. Each
page renders the notebook content; click the "Open in Colab" badge at the top
to run it interactively.

```{toctree}
:maxdepth: 1
:glob:

*
````

```

The `:glob:` pattern picks up every `.ipynb` copied in by the prebuild step automatically — no per-notebook entry to keep in sync.

### 5. Gitignore — `/.gitignore`

Add one line under the existing `docs/source/changelog.md` / `docs/**/_auto/` block:

```

docs/source/tutorials/colab/\*.ipynb

```

Keeps copied notebooks out of git (they are generated artifacts, just like `changelog.md`).

### 6. `--fail-on-warning` hygiene

The existing `make docs` runs with `--fail-on-warning --keep-going`. nbsphinx typically emits warnings for:
- Missing kernelspec — solved by `nbsphinx_kernel_name = "python3"` (step 2).
- Notebooks not in any toctree — solved by the `:glob:` toctree (step 4).

If new warning classes surface during the local build, add them to `suppress_warnings` in `conf.py` rather than removing `--fail-on-warning`. Do **not** disable the strict flag.

### 7. Styling (only if needed)

Furo + nbsphinx usually look fine out of the box. If input/output prompts render awkwardly in dark mode, add a few CSS rules to `docs/source/_static/custom.css` targeting `.nbinput`, `.nboutput`, and `.prompt`. Defer until visually inspected — don't add CSS preemptively.

## Files Touched

- `pyproject.toml` — add `nbsphinx`, `pypandoc-binary`
- `docs/source/conf.py` — add extension, nbsphinx settings, prolog, exclude
- `docs/prebuild.py` — add `copy_notebooks()` and call in `main()`
- `docs/source/tutorials/colab/index.md` — replace with toctree
- `.gitignore` — ignore copied notebooks

No changes to `docs/Makefile`, `docs/build.py`, `docs/source/index.md`, or the GitHub Actions workflow — the existing pipeline already runs `prebuild.py` then `sphinx-build` with the right flags.

## Verification

1. `uv sync --group dev` — confirms nbsphinx + pypandoc-binary install cleanly across supported Python versions.
2. `cd docs && make clean && make docs` — confirms:
   - prebuild copies all 14 notebooks into `docs/source/tutorials/colab/`
   - sphinx-build completes without warnings (so `--fail-on-warning` passes, as it does today)
3. `make serve` then open `http://localhost:1234/stable/tutorials/colab/` — visual check that:
   - the index lists all 14 notebooks
   - one representative notebook (e.g. `object_detection`) renders with markdown headings, code cells, and the prolog Colab badge
   - sidebar navigation and search still work, in both light and dark mode
4. `git status` — confirm no `.ipynb` files appear under `docs/source/tutorials/colab/` (gitignore working).
5. Push a branch and let `test_documentation.yml` run — same `make docs` invocation as local, so should pass.

## Out of Scope

- Executing notebooks at build time (rejected — heavy training + dataset downloads).
- Reducing epochs/datasets in the notebooks for fast execution (separate UX decision; would degrade the standalone Colab experience).
- Adding a Colab badge to `torchvision_embedding_model.ipynb` (it's missing one) and adding kernelspec to the two notebooks that lack it — useful but independent of this integration; can be a follow-up.
- Replacing the inline Colab badges on `index.md`, `object_detection.md`, etc. with links to the rendered notebook pages — also a follow-up after the new pages exist and we see them in practice.
```
