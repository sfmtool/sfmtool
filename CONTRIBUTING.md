# Contributing to sfmtool

Thanks for your interest in contributing! sfmtool is a personal project
aimed at making Structure from Motion (SfM) reconstruction fun to play
with. It is not a stable, supported tool — APIs, file formats, and CLI
shapes can all shift between commits — but if you have interesting
datasets sitting around, or you just want to poke at SfM, we hope you
enjoy trying it out.

Under the hood, sfmtool is a multi-language toolkit: a Python CLI and
pipeline (`src/sfmtool/`) on top of a Rust Cargo workspace (`crates/`).
The two are wired together by PyO3 bindings in `crates/sfmtool-py`.

## Development environment

The project uses [Pixi](https://pixi.sh) for all environments and tasks.
Once Pixi is installed, run everything via `pixi run …`.

```bash
pixi run sfm --help                 # Run the CLI
pixi run test                       # Python tests (pytest)
pixi run test -- tests/test_x.py    # Single test module
pixi run cargo test --workspace     # Full Rust test pass
pixi run test-rust                  # Rust tests with coverage
                                    # (excludes sfmtool-py + sfm-explorer)
pixi run coverage-all               # Combined Python + Rust coverage
pixi run gui                        # Launch the SfM Explorer GUI
pixi run docs-serve                 # Live-rebuild the Zensical docs site
```

Environments: `default`, `test`, `dev`, `docs`, `cuda`. The CI matrix uses
`test`. See `pixi.toml` for the full task list.

## Before you submit a PR

Run the checks that match what you changed.

| You changed…                          | Run                                             |
| ------------------------------------- | ----------------------------------------------- |
| Anything in `src/sfmtool/` or tests   | `pixi run fmt && pixi run check`                |
| Anything in `crates/`                 | `pixi run cargo fmt && pixi run cargo clippy --workspace` |
| Anything re-exported via `sfmtool-py` | `pixi run maturin develop --release` *before* re-running Python tests |

The native extension `sfmtool._sfmtool` is **not** rebuilt by the editable
Python install — `maturin develop` is required after any Rust edit that
crosses the PyO3 boundary, otherwise Python tests run against a stale `.so`.

CI runs `pixi run -e test coverage-all` on Linux and Windows, plus a fast
lint job (`ruff format --check`, `ruff check`, `cargo fmt --check`,
`cargo clippy -- -D warnings`). The lint job gates the test matrix, so a
formatting slip won't burn the full test budget.

## Specs

Design lives under `specs/` (subdirs: `cli/`, `core/`, `formats/`, `gui/`,
`workspace/`, `drafts/`). For non-trivial behavior changes, read the
relevant spec first and update it in the same PR. Not every CLI command
has a spec yet, and not every spec maps 1:1 to a top-level CLI command;
when in doubt, mention the gap in the PR description.

## Test data

Three datasets are checked in under `test-data/images/`:

- `seoul_bull_sculpture` — 17 @ 270×480 (the small fixture used in tests)
- `dino_dog_toy` — 85 @ 2040×1536
- `seattle_backyard` — 26 @ 360×640

Bootstrap a workspace with `scripts/init_dataset_*.sh`. Reuse these in bug
reports and reproductions where possible.

## Commit style

Short, imperative subject lines (e.g. "Add a spec for the spherical tile
rig", "Fix NumPy to quaternion transpose bug"). Group related changes into
one commit; avoid noisy fixup commits in the final history.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Please include
the `pixi run sfm --version` output (or commit SHA) and the exact commands
you ran.
