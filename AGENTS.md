# Working in the sfmtool Repository

Multi-language SfM toolkit: a Python CLI and pipeline orchestration layer
(`src/sfmtool/`) on top of a Rust Cargo workspace (`crates/`) that owns the
algorithms, file-format I/O, GUI viewer, and PyO3 bindings. Config lives in
`pyproject.toml` and `pixi.toml`.

## Environments and tasks

This repo uses [Pixi](https://pixi.sh). Run everything via `pixi run …`.
Environments: `default` (runtime), `test` (adds pytest, ruff, maturin,
cargo-llvm-cov), `dev` (ipython), `docs` (zensical), `cuda` (CUDA-enabled
pycolmap). See `pixi.toml` for the full task list.

```bash
pixi run test                       # Python tests (pytest)
pixi run test -- tests/test_xxx.py  # Single test module
pixi run test-rust                  # Rust tests w/ coverage (excludes sfmtool-py + sfm-explorer)
pixi run coverage-all               # Combined Python + Rust coverage (scripts/coverage.sh)
pixi run fmt && pixi run check      # Python format + lint (ruff)
pixi run cargo {fmt,clippy,test,check} --workspace
pixi run maturin develop --release  # Rebuild PyO3 bindings after Rust edits
pixi run gui [-- path.sfmr]         # Build and run the SfM Explorer (release)
pixi run docs-{build,serve}         # Zensical docs
pixi run sfm …                      # Run the CLI
```

### Task completion checks

When finishing a task, run the checks for what you changed:

- Python changes → `pixi run fmt && pixi run check`
- Rust changes → `pixi run cargo fmt && pixi run cargo clippy --workspace`
- Rust edits that touch anything re-exported through `sfmtool-py` → rerun
  `pixi run maturin develop --release` before Python tests (the `.so` does
  **not** rebuild automatically despite the editable Python install).

## Structure at a glance

- `src/sfmtool/` — Python package (~93 modules). Entry point is `cli.py`
  (Click + `_cli_group.CategoryGroup` for categorized `--help`). Subpackages:
  - `_commands/` — one module per top-level CLI subcommand
  - `feature_match/` — descriptor matching, polar/rectified sweep, flow matching, geometric filtering
  - `xform/` — reconstruction transforms (align, filter, rotate, scale, translate, bundle-adjust, …)
  - `visualization/` — colormap, heatmap, discontinuity display
- `crates/` — Cargo workspace, 7 crates:
  - `sift-format`, `matches-format`, `sfmr-format` — on-disk formats (`.sfmr` is ZIP + zstd)
  - `sfmr-colmap` — COLMAP binary + SQLite interop
  - `sfmtool-core` — algorithms: camera, alignment, distortion, epipolar, matching, frustum, optical flow, transforms, spatial indexing
  - `sfm-explorer` — native GUI viewer (winit + wgpu + egui); window title "SfM Explorer"
  - `sfmtool-py` — PyO3 bindings, compiled as `sfmtool._sfmtool`
- `tests/` — pytest, ~43 modules (top-level + `tests/xform/`). Fixtures in
  `conftest.py` — notably `isolated_seoul_bull_image` and
  `isolated_seoul_bull_17_images`. Look for `test_*_rust_bindings.py` modules
  that exercise the PyO3 surface.
- `specs/` — design specs. Read the relevant file before making non-trivial
  changes and update it when behavior diverges. Subdirs: `cli/` (per-command),
  `core/` (algorithm design), `formats/`, `gui/`, `workspace/`.
- `test-data/images/` — three checked-in datasets:
  `seoul_bull_sculpture` (17 @ 270×480), `dino_dog_toy` (85 @ 2040×1536),
  `seattle_backyard` (26 @ 360×640). Bootstrap with `scripts/init_dataset_*.sh`.
- `docs/` — Zensical site, deployed to GitHub Pages.
- `.github/workflows/` — `ci.yml` (Linux + Windows, runs `coverage-all`, uploads
  to codecov), `docs.yml`, `publish_to_pypi.yml`.

## CLI

Run `pixi run sfm --help` to list all subcommands grouped by category
(Workspace / Image Feature / Reconstruction / Visualization / Image Processing
/ COLMAP Interop). Source in `src/sfmtool/_commands/<name>.py`; specs in
`specs/cli/<name>-command.md`. `sfm cam` is a command **group** (with one
subcommand `cp` today); every other top-level command is flat. Typical
reconstruction flow:

```bash
cd workspace-dir
pixi run sfm init .
pixi run sfm sift --extract images
pixi run sfm solve -i images     # incremental SfM
pixi run sfm solve -g images     # global SfM
```

## Things that can surprise you

- `pixi run test-rust` excludes `sfmtool-py` and `sfm-explorer` (llvm-cov
  limitations). Use `pixi run cargo test --workspace` to cover those.
- The Python package is editable-installed, but the native extension
  `sfmtool._sfmtool` is not auto-rebuilt — remember `maturin develop` after
  Rust changes.
- `sfm explorer` launches the same binary as `pixi run gui`, just via the
  Python CLI through the bindings.
- Not every `specs/cli/*-command.md` maps to a top-level command
  (e.g. `scale-by-measurements-command.md` documents an `xform` sub-command);
  likewise not every CLI command has a spec yet.
- Python 3.14 and Rust 1.94 are pinned in `pixi.toml`.
- A workspace can supply per-directory camera intrinsics via
  `camera_config.json` files; resolution is closest-ancestor-wins, capped at
  the workspace root. See `src/sfmtool/_camera_config.py` and
  `specs/workspace/camera-config.md`. When such a file resolves for any image
  in a `solve` / `match` / `to-colmap-db` invocation, `--camera-model` is
  rejected up front.
