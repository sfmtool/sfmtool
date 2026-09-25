# Working in the sfmtool Repository

Multi-language SfM toolkit: a Python CLI and pipeline orchestration layer
(`src/sfmtool/`) on top of a Rust Cargo workspace (`crates/`) that owns the
algorithms, file-format I/O, GUI viewer, and PyO3 bindings. Config lives in
`pyproject.toml` and `pixi.toml`.

## Environments and tasks

This repo uses [Pixi](https://pixi.sh). Run everything via `pixi run …`.
Environments: `default` (runtime), `test` (adds pytest, ruff, maturin,
cargo-llvm-cov), `dev` (ipython), `docs` (zensical), `cuda` (CUDA-enabled
pycolmap). `test-ci` is `test` minus the editable `sfmtool` self-install, so
`pixi install` does no Rust build — CI uses it and compiles the extension once
via `maturin develop --release`; locally prefer `test` (imports work with no
extra step). See `pixi.toml` for the full task list.

```bash
pixi run test                       # Python tests (pytest)
pixi run test -- tests/test_xxx.py  # Single test module
pixi run test-rust                  # Rust tests w/ coverage (excludes sfmtool-py + sfm-explorer)
pixi run coverage-all               # Combined Python + Rust coverage (scripts/coverage.sh)
pixi run fmt && pixi run check      # Python format + lint (ruff)
pixi run cargo {fmt,clippy,test,check} --workspace
pixi run doc                        # Rustdoc gate (warnings are errors)
pixi run maturin develop --release  # Rebuild PyO3 bindings after Rust edits
pixi run gui [-- path.sfmr]         # Build and run the SfM Explorer (release)
pixi run gui-mcp [path.sfmr]        # …with its MCP endpoint, for an agent to drive
pixi run docs-{build,serve}         # Zensical docs
pixi run sfm …                      # Run the CLI
```

### Task completion checks

When finishing a task, run the checks for what you changed:

- Python changes → `pixi run fmt && pixi run check`
- Rust changes → `pixi run cargo fmt && pixi run cargo clippy --workspace &&
  pixi run doc`
- Rust edits that touch anything re-exported through `sfmtool-py` → rerun
  `pixi run maturin develop --release` before Python tests (the `.so` does
  **not** rebuild automatically despite the editable Python install).

### Rust visibility in the viewer

In `sfm-explorer`, keep an item private when only its module and descendants use
it. Use `pub(crate)` when sibling modules need it, including methods and fields
of types declared in private modules. Reserve bare `pub` for an item exposed
through the crate's public interface.

### Opening a pull request

**Every PR body follows `.github/PULL_REQUEST_TEMPLATE.md`** — read it before
writing the description, not after. Its sections (Summary / Changes / Testing /
Checklist / License) are what reviewers look for, and `gh pr create --body`
bypasses the template silently: GitHub only prefills it in the web editor, so a
PR opened from the CLI gets whatever body you hand it and no warning that a
template existed. Keep the checklist items and the License block verbatim,
ticking what applies and marking the rest `n/a` with the reason, rather than
deleting the lines.

### Writing style

Specs, doc comments, commit messages and PR descriptions are written in plain
language. Do not use mannered prose: no metaphors or idioms where a
straightforward literal phrase says the same thing clearly. Write "the
constellation query matched no other image", not "the constellation came up
empty-handed". Name the thing and say what it does.

## Structure at a glance

- `src/sfmtool/` — the Python package. Entry point is `cli.py`
  (Click + `_cli_group.CategoryGroup` for categorized `--help`). Subpackages:
  - `_commands/` — one module per top-level CLI subcommand
  - `align/` — alignment of multiple reconstructions (pairwise, by-cameras, by-points, multi-way)
  - `analyze/` — reconstruction analysis: summary, per-image metrics, depth, covisibility/frustum graphs
  - `camera/` — camera intrinsics, EXIF/config-based inference, `camera_config.json` resolution
  - `camrig/` — `.camrig` rig construction, copy, pattern matching, solve resolution
  - `colmap/` — COLMAP interop: DB setup for the solvers, binary/pycolmap ↔ `.sfmr` conversion, DB export
  - `feature_match/` — descriptor matching, polar/rectified sweep, flow matching, geometric filtering
  - `merge/` — merge aligned reconstructions (point correspondences + pose refinement)
  - `motion/` — camera-motion discontinuity analysis (image sequences + reconstructions)
  - `rig/` — multi-sensor rig ingestion/rendering: `rig_config.json`, frame grouping, insv2rig/pano2rig, equirect render
  - `sift/` — SIFT feature file I/O and extraction (COLMAP, OpenCV, and
    `sfmtool` Rust backends)
  - `strips/` — patch-strip montages behind `compare --strips` and
    `inspect --strips` (surfel strip rendering, NCC scoring, montage layout)
  - `xform/` — reconstruction transforms (align, filter, rotate, scale, translate, bundle-adjust, …)
  - `visualization/` — colormap, heatmap, discontinuity display
- `crates/` — Cargo workspace, 11 crates:
  - `sfmtool-progress` — the `Progress` parameter a long call reports its
    phases, messages, counts and fraction through and is cancelled by. Its own
    crate, std-only, so the format crates report through the same type
    `sfmtool-core` does; core re-exports it as `sfmtool_core::progress`
  - `sfmtool-sift-format`, `sfmtool-matches-format`, `sfmtool-sfmr-format`,
    `sfmtool-camrig-format`, `sfmtool-kdf-format` — on-disk formats (all ZIP + zstd)
  - `sfmtool-archive-io` — the ZIP + zstd container primitives those five share
    (entry read/write, XXH128 section hashing); each format crate keeps its own
    schema, validation and error type
  - `sfmtool-colmap` — COLMAP binary + SQLite interop
  - `sfmtool-core` — algorithms: camera, alignment, distortion, epipolar, matching, frustum, optical flow, transforms, spatial indexing
  - `sfm-explorer` — native GUI viewer (winit + wgpu + egui); window title
    "SfM Explorer", or "SfM Explorer - <file>.sfmr" once a file is loaded
  - `sfmtool-py` — PyO3 bindings, compiled as `sfmtool._sfmtool`
- `tests/` — pytest (top-level modules + `tests/camrig/`, `tests/matching/`,
  `tests/patch/`, `tests/rig/`, `tests/rust_bindings/`, `tests/sift/` and
  `tests/xform/`). Fixtures in
  `conftest.py` — notably `isolated_seoul_bull_image` and
  `isolated_seoul_bull_17_images`. `tests/camrig/`, `tests/patch/` and
  `tests/xform/` each add a
  `conftest.py` of shared helpers, imported as `from .conftest import …`
  (the other subpackages have none). Look for `test_*_rust_bindings.py` modules
  that exercise the PyO3 surface.
- `specs/` — design specs, indexed by `specs/README.md`. Read the relevant file
  before making non-trivial changes and update it when behavior diverges.
  **`specs/GLOSSARY.md` holds the words this project has settled on** -- read it
  before naming a function, a wire tool, a field or a user-facing string, and
  add to it when a naming question is settled by argument. A glossary entry
  outranks a count of which spelling is currently more common in the tree. Each
  area carries a `README.md` index — `cli/` (all commands, by category), each
  `core/` module subdir, `formats/`, `gui/`, `workspace/`. Add a row there when
  you add a spec. **Start a new spec from `specs/TEMPLATE.md`**, whose default
  order is: purpose, then the public Rust interface *with why it is shaped that
  way and an example call*, then the theory, then implementation notes — and
  those notes carry what the code cannot (a cross-function invariant, why this
  loop order, a numerical hazard), never a transcription of the body. It is a
  starting point, not a form: drop, merge or reorder sections where the subject
  is served better. What holds regardless is that the opening paragraph reads for
  someone who has read no other spec, that a caller can find out what to call
  without wading through a derivation, and that a spec describes what the code
  **is**, in the present tense — write a change proposal in `specs/drafts/` and
  convert it before filing, so no standing spec says "new module X" or "prior
  state (before this change)". Location encodes lifecycle, so a standing spec
  carries no `**Status:**` line and opens with its purpose paragraph; only a
  draft in `specs/drafts/` opens with `**Status:** Draft`. A specified-but-unbuilt
  part is a present-tense sentence in the standing spec plus an amendment draft it
  links both ways, never an inline "not yet implemented" marker. The pointer at
  the implementing code belongs in the spec's interface section, with repo paths
  written as relative Markdown links. The `audit-specs` skill audits a sample of specs
  against the template on each run, for those invariants rather than for
  conformance. Subdirs mirror the code they describe:
  - `cli/` — one file per command, grouped by the `--help` category the command
    is registered under in `cli.py` (`workspace/`, `image-feature/`,
    `reconstruction/`, `visualization/`, `image-processing/`,
    `colmap-interop/`); the `xform` sub-commands nest in
    `cli/reconstruction/xform/`
  - `core/` — algorithm design, one subdir per `sfmtool-core` module
    (`analysis/`, `camera/`, `features/`, `geometry/`, `patch/`,
    `reconstruction/`, `spherical/`)
  - `formats/` — on-disk formats, one per format crate
  - `gui/` — `sfm-explorer` design, flat
  - `workspace/` — workspace layout and its config files
- `test-data/images/` — four checked-in datasets:
  `seoul_bull_sculpture` (17 @ 270×480), `dino_dog_toy` (85 @ 2040×1536),
  `seattle_backyard` (26 @ 360×640), `kerry_park` (24 rig frames × 2 fisheyes
  @ 480×480, with `rig_config.json`). Bootstrap with `scripts/init_dataset_*.sh`.
  `seoul_bull_sculpture` also holds `seoul_bull_sculpture_ground_truth.sfmr`,
  a reference reconstruction of its 17 images, in metres. A
  `.sfm-workspace.json` beside the images makes that directory its workspace,
  so it opens in place (`pixi run gui --
  test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr`).
  It is a minimal file with embedded patches, so it needs no `.sift` files; its
  scale comes from three GPS fixes. Don't run
  `sfm sift` or a solve in that directory: the marker would put their output
  in `test-data`.
- `docs/` — Zensical site, deployed to GitHub Pages.
- `reports/` — dated snapshots from the audit skills (`audit-hygiene`,
  `audit-specs`, `suggest-next-steps`). See "Quality reports" below.
- `skills/` — the four project skills (`audit-hygiene`, `audit-specs`,
  `implement-random-idea`, `suggest-next-steps`), checked in here and symlinked
  into `.claude/skills/`.
- `.github/workflows/` — `ci.yml` (Linux runs `coverage-all` + codecov upload;
  Windows and macOS run the same suites without instrumentation, split across
  two matrix jobs that share nothing and so run in parallel — `test-os-rust`
  (`cargo test --workspace`) and `test-os-python` (`maturin develop --release`
  then `pytest`); the windowed `ui_basic` suite gets a job per platform —
  `ui-test-windows`, `ui-test-macos`, `ui-test-linux`; the Rust build is
  cached, saved and pruned by `main` only — the "Rust caches" comment above
  `prune-caches` says why. Two things there are easy to get wrong: a **target**
  cache is keyed on `hashFiles('Cargo.lock', 'Cargo.toml',
  'crates/*/Cargo.toml')`, not the lockfile alone, because feature resolution
  and which targets exist are declared in the *manifests* — key it on the
  lockfile and a feature change silently restores artifacts built under a
  different resolution, which is a full rebuild that the `cache-hit != 'true'`
  save guard then never replaces; and `test-os-rust` and `test-os-python` must
  keep *separate* target-cache prefixes because they build disjoint trees
  (`target/debug` vs `target/release`). A **cargo home** stays keyed on
  `Cargo.lock` alone, so the two kinds are on different generations and
  `ci_prune_caches.sh` takes one hash per call — prune them in separate
  invocations. pixi envs are not cached anywhere, in either workflow),
  `docs.yml`, `publish_to_pypi.yml`.

## CLI

Run `pixi run sfm --help` to list all subcommands grouped by category
(Workspace / Image Feature / Reconstruction / Visualization / Image Processing
/ COLMAP Interop). Source in `src/sfmtool/_commands/<name>.py`; specs in
`specs/cli/<category>/<name>-command.md`. `sfm ws` and `sfm camrig` are command **groups**
(`ws` has one subcommand, `ws init`; `camrig` has three: `camrig create`,
`camrig cp`, `camrig spherical-tiles`); every other top-level command is flat.
Typical reconstruction flow:

```bash
cd workspace-dir
pixi run sfm ws init .
pixi run sfm sift --extract images
pixi run sfm solve -i images     # incremental SfM
pixi run sfm solve -g images     # global SfM
```

## Quality reports

`reports/` holds dated read-only snapshots produced by the audit skills
(`audit-hygiene`, `audit-specs`, `suggest-next-steps`). Treat them as a living
backlog and keep them honest as findings get addressed:

- **Mark off findings in place.** Whenever you act on a recommendation from a
  report, annotate that finding inline rather than deleting it — add a dated
  status line in the established style, e.g.
  `> _Status (YYYY-MM-DD): Done — <what changed>, commit <sha>._` (use
  `Partially done` / `Not done` as appropriate). The body of a finding stays as
  the original snapshot; status accretes above or below it. This is how the
  existing reports already track progress.
- **Retire a report once it has outlived its usefulness — use judgement.** The
  bar is "is this still earning its place as a live backlog?", not "is every last
  box ticked". Retire (delete the whole file, git preserves history) when any of
  these holds:
  - Every finding is resolved or superseded.
  - The substantive findings are resolved and only minor or discussion-grade
    items remain (carry those forward — fold them into a related report, the
    next regenerated snapshot, or an issue — rather than keeping a near-empty
    report alive for them).
  - The report has gone stale against significant code movement, such that
    re-running the audit skill would supersede it more cleanly than annotating it
    item by item. In that case regenerate a fresh dated snapshot and delete the
    old one in the same commit, carrying any still-open findings into the new
    report.

  When you retire a report, say briefly in the commit message why (resolved /
  superseded / stale-and-regenerated) and where any unfinished items went. Don't
  leave fully-actioned or clearly-stale reports lying around, but don't force a
  retirement while a report is still doing useful work tracking real open items.

## Things that can surprise you

- `pixi run test-rust` excludes `sfmtool-py` and `sfm-explorer` (llvm-cov
  limitations). Use `pixi run cargo test --workspace` to cover those.
  `sfm-explorer` splits in two: its `ui_basic` integration tests need a real
  window, while its **lib** tests are headless —
  `scene_renderer/upload/tests.rs` drives real `wgpu` uploads on the `noop`
  backend and `track_view/view/tests.rs` runs whole egui frames through
  `Context::run_ui`, so they need neither a GPU nor a window and run anywhere.
  **That split is a Cargo feature, not a convention**: `ui_basic` is declared as
  an explicit `[[test]]` target with `required-features = ["ui-tests"]`, so a
  plain `cargo test --workspace` builds and runs everything *except* it, and
  `pixi run ui-test` (all three desktop platforms) passes `--features ui-tests`
  to get it. The trap in that arrangement is that cargo **silently skips** a
  `required-features` target rather than erroring, so anything meant to check
  `ui_basic` has to name the feature — which is why the `lint` job's clippy
  invocation carries `--features sfm-explorer/ui-tests` (not `--all-features`,
  which would drag in the Windows-only `directmanipulation` examples). In CI the
  lib tests execute in the `test-os-rust` (Windows/macOS) jobs; Linux compiles
  them, and `ui_basic` with them, in `lint`, but does not run them, to keep
  uninstrumented artifacts out of the coverage job's target dir. `ui_basic` has
  a job per platform instead (`ui-test-{windows,macos,linux}`).
- **`pixi run ui-test` on Linux needs an accessibility stack, and says nothing
  when it is missing.** xa11y reads the viewer's tree over AT-SPI2, which is a
  pair of D-Bus services rather than part of the OS, so a headless box needs a
  display, a session bus and those daemons before there is a tree at all — and
  a query without them returns an *empty* tree, not an error, so every widget
  assertion fails while the launch looks healthy. The Linux `ui-test` task
  routes through `scripts/a11y_env.sh`, which starts only what is missing and
  is a passthrough on a real desktop; CI uses `xa11y/setup-a11y` for the same
  thing. The viewer also needs a Vulkan ICD (`mesa-vulkan-drivers` for
  lavapipe): Vulkan is the only wgpu backend compiled in for Linux, so without
  one it panics at surface creation. See `specs/gui/architecture.md` §
  "Testing".
- Rustdoc warnings are **errors**, via `[workspace.lints.rustdoc]` in the root
  `Cargo.toml` (each crate opts in with `[lints] workspace = true`). That means
  a plain `cargo doc` fails on a broken intra-doc link, not just `pixi run doc`.
  The gate documents private items, so links written inside private modules are
  checked too. A link to a genuinely private item from a **public** item's doc
  is an error (`private_intra_doc_links`) — write those as a plain code span
  (`` `foo` ``) rather than widening visibility. Method refs inside an inherent
  impl need `Self::`; sibling private modules need `super::`.
- Every `.rs` under `crates/` and every `.py` under `src/`, `tests/` and
  `scripts/` opens with the two-line `Copyright The SfM Tool Authors` /
  `SPDX-License-Identifier: Apache-2.0` header (after the shebang, where there
  is one), and a new file without it fails
  `crates/sfmtool-core/tests/license_headers.rs`.
- The Python package is editable-installed, but the native extension
  `sfmtool._sfmtool` is not auto-rebuilt — remember `maturin develop` after
  Rust changes.
- `sfm explorer` launches the same binary as `pixi run gui`, just via the
  Python CLI through the bindings.
- **The viewer can be driven over MCP, and an agent may own its lifecycle.**
  `pixi run gui-mcp <file>.sfmr` hosts a Model Context Protocol endpoint on
  `127.0.0.1:8787` for reading the scene graph, moving the selection and the 3D
  camera, reading the Action Log back, and **screenshotting the window or any
  panel in it** — usually the fastest way to find out whether a solve is wrong
  and *where*. Off unless asked for; see `specs/gui/mcp-server.md`.

  Setup is a human's job, exactly once, because Claude Code binds its MCP
  servers when a session starts:

  ```bash
  claude mcp add --transport http sfm-explorer http://127.0.0.1:8787/mcp
  ```

  A viewer must be listening on that port at the moment a session starts, or
  the server registers as failed and its tools are absent for the whole
  session — `/mcp` says which. **After that the agent can restart the viewer
  freely**: the transport is stateless HTTP, so killing it, rebuilding, and
  relaunching on the same port leaves the registered tools working, since the
  next call is just a new POST into the new process. Launch it *detached* when
  doing that (on Windows, `cmd /c start "" target/release/sfm-explorer.exe
  --mcp 8787 <file>.sfmr`) so it outlives the session that spawned it and is
  still there for the next one. Kill it **before** rebuilding: a running
  `.exe` is locked on Windows and `cargo build` fails with `Access is denied
  (os error 5)`.

  Keep the registration at the default `local` scope. A `.mcp.json` committed
  with `--scope project` would point every contributor's session at a loopback
  port that is usually not listening, and hand them a failed server on every
  start.
- Not every CLI command has a spec yet. The `xform` sub-commands are specced
  under `specs/cli/reconstruction/xform/` rather than as top-level commands.
- Python 3.14 and Rust 1.98 are pinned in `pixi.toml`. That is the *development*
  toolchain, and it is deliberately not the same thing as the MSRV: the workspace
  declares `rust-version = "1.97"` in `[workspace.package]` (inherited by every
  crate in the workspace), because the PyPI sdist compiles this workspace on the
  user's own rustc
  and we publish wheels for Linux and Windows only. The `msrv` job in `ci.yml`
  builds against that floor; it reads the version out of `Cargo.toml`, so raise
  the MSRV there and nowhere else. The floor is deliberately kept one release
  behind the dev toolchain rather than pinned to whatever the dependency tree
  needs today, so a routine dependency bump does not immediately force an MSRV
  bump with it. Bumping a dependency can raise the floor silently —
  `libsqlite3-sys` forced 1.95 while declaring no `rust-version` — so trust that
  job over dependency metadata.
- A workspace can supply per-directory camera intrinsics via
  `camera_config.json` files; resolution is closest-ancestor-wins, capped at
  the workspace root. See `src/sfmtool/camera/config.py` and
  `specs/workspace/camera-config.md`. When such a file resolves for any image
  in a `solve` / `match` / `to-colmap-db` invocation, `--camera-model` is
  rejected up front.
