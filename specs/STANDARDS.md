# Quality Standards

The rules every library function, binding, Python module, source file, test
and spec in sfmtool is held to, whatever module it is in: the contracts of its
interfaces, how its names and modules are laid out, and what every build must
pass. Each section states one standard:
the rule, why it exists, how to check code against it and what is out of scope.
The `audit-hygiene` and `audit-specs` skills check against this file, and
every implementer and auditor agent that makes or reviews a change reads it,
so the audit that writes a finding and the agents that fix it work from the
same text. A finding or a fix
names the standard it concerns by its heading, as a link such as
`specs/STANDARDS.md#caller-input-never-panics`; keep the headings stable for
that reason.

A breach is a finding even where the whole module is consistent with itself,
so check each standard per function, not per convention split.

## Direction of a fix

When a spec and the code disagree because the code breaks one of these
standards, or another rule stated in `AGENTS.md` or
[GLOSSARY.md](GLOSSARY.md), the fix is in the code. The spec is corrected only
where the code is right and the text is wrong.

A spec or doc edit that describes the breach as how the code works (a panic
on bad input, a module that is "the exception" to a rule, a Python wrapper
over a binding) keeps the defect in place: the next reader takes the text as
the design, and the next audit acquits the code against it. When the code fix
is a large design change, the finding or the fix records that it needs a
maintainer's decision and names the code fix; it does not commit the text
that documents the breach in the meantime.

## Points at infinity are supported

A reconstruction's points are homogeneous: `positions_xyzw`, where a `w = 0`
row is a unit direction (see
[sfmr-file-format.md](formats/sfmr-file-format.md)). Every library function
that takes, returns or iterates over a reconstruction's points accepts points
at infinity in its interface and handles them correctly. A function with a
real reason to work on finite points only says so in its contract and skips or
rejects `w = 0` rows visibly. This exists because a direction treated as a
position turns into `inf` or `NaN`, or into a wrong but finite answer, far
from where it was introduced.

Look for:

- An interface that can only express finite points: a `positions_xyz` /
  `[f64; 3]` / `Vec3` parameter or return value where the data can hold
  `w = 0` rows, or a Python wrapper that slices `[:, :3]` off an xyzw array.
- A division by `w` with no `w = 0` branch.
- Wrong geometry for a direction: translation or scale applied to a `w = 0`
  row (only the rotation acts on it), a depth, distance, centroid, bounding
  box or spatial index computed over directions as if they were positions, or
  a reprojection or triangulation path that assumes a finite point.
- Points at infinity dropped silently: filtered out with no count in the
  result and no error. Converting directions to finite points is
  `materialize_points_at_infinity`'s job, for a consumer that cannot store
  `w = 0`, and is not reinvented inline.

Confirm before reporting: read the function and, where cheap, say what input
shows the bug (for example a reconstruction with one `w = 0` point passed to
the transform). Out of scope: a function that never sees reconstruction points
(image warping, SIFT, descriptor matching).

## Long-running functions take a Progress

A library function whose running time can exceed roughly a second on
realistic input takes a `&Progress` (`sfmtool_progress::Progress`, re-exported
as `sfmtool_core::progress::Progress`), reports through it, checks it for
cancellation and passes it down to long-running callees. This exists so the
CLI and the viewer can show where a long call is and stop it.

Treat as above the threshold: a loop over every image, image pair, track or
point of a reconstruction; an iterative solver or RANSAC run; reading or
writing a whole `.sfmr`, `.sift`, `.matches` or `.camrig` file; and any
function that calls one that takes a `Progress`. Also report:

- A function that takes a `Progress` and then calls a long-running callee with
  `&Progress::none()` instead of passing its own down (or a `phase` or `split`
  of it). The caller's progress display stops moving and cancellation stops
  working for that stage.
- A Python-facing binding of such a function that offers no way to pass
  progress in (`ProgressCounter`, see
  [py_progress.rs](../crates/sfmtool-py/src/py_progress.rs)).

Out of scope: short functions (a single reprojection, a per-point kernel
called from inside a loop that already reports). The threshold is about the
whole call, and a `Progress` on every helper adds nothing.

## One interface, no progress variants

A caller that wants no reporting passes `&Progress::none()`. There is one
function, not a pair such as `foo` / `foo_with_progress`, `foo` / `foo_silent`,
or a public `foo` that only forwards to a public `foo_inner(…,
&Progress::none())`. The same applies to an `Option<&Progress>` parameter and
to a `verbose: bool` or `show_progress: bool` flag next to or instead of a
`Progress`, which are a second interface in a different form. Two interfaces
drift apart, and callers pick the one that cannot be cancelled. The fix merges
them into one function that takes `&Progress` and updates the callers.

Out of scope: a private helper that takes no `Progress` because it is short.

## Caller input never panics

A library function validates what it is given and returns a `Result` with a
descriptive error enum, with a hand-written `Display` and `Error` as the
crates already write them (`MatchImagePairError` is an example). A binding
maps the error to a Python exception: `ValueError`, or `IndexError` or
`TypeError` where the existing bindings use them. A `PanicException` that
Python code can reach is a bug. `assert!`, `expect` and `unwrap` stay only for
internal invariants that valid input cannot reach. This exists because a panic
gives a Python caller no exception type to catch and no message about which
argument was wrong, and because validation in the binding alone leaves a Rust
caller of the same function to panic.

The standard covers three kinds of code:

- **Code Python can reach.** No input that reaches a function through a
  binding in `sfmtool-py` may panic, whichever crate the function is in.
- **The public Rust API of the library crates**: `sfmtool-core`,
  `sfmtool-progress`, `sfmtool-archive-io`, `sfmtool-colmap` and the five
  format crates (`sfmtool-sift-format`, `sfmtool-matches-format`,
  `sfmtool-sfmr-format`, `sfmtool-camrig-format`, `sfmtool-kdf-format`). A
  public function returns an error for bad caller input.
- **The parts of `sfm-explorer` that handle data from outside the process**:
  its command-line arguments (which `sfm explorer` passes in through a
  binding), the files it reads and the MCP requests it serves. The rest of the
  viewer,
  its `pub(crate)` and private code that only the viewer calls, is outside
  this standard.

The code does not yet meet this standard everywhere. Existing public
functions whose doc comment has a `# Panics` section about caller input are
backlog to fix: the hygiene audit lists and ranks them, and new code adds no
more. An existing panic on caller input is a defect, not a precedent.

Look for:

- `assert!`, `expect`, `unwrap`, `panic!` or unchecked indexing on argument
  data in a function the scope above covers: lengths that must match, indices
  that must be in range, a shape that must hold.
- A binding that passes unchecked arrays or lengths into code that panics on a
  mismatch (`from_row_slice`, slicing, `chunks_exact` with an assumed length,
  indexing by a caller's index).
- Validation of a core function's input done only in its binding.
- A spec or doc comment that says a function panics on bad input.

Confirm before reporting: name the argument and the value that reaches the
panic and, where cheap, the Python call that raises `PanicException`. Out of
scope: an `expect` on an invariant valid input cannot break, such as a value
the function itself just computed.

## Logic lives in Rust, not in Python wrappers

Logic lives in the Rust crates, in core or in the bindings. Python code over a
binding is not a second layer: there is no Python class or function whose job
is to re-validate, re-read or reshape what a binding already does. This exists
because two layers doing the same work disagree over time, and the Python copy
is the slower one that no Rust caller sees. Where a binding's interface is not
what Python callers need, the binding changes.

Look for a Python class or function in `src/sfmtool/` whose body re-checks
arguments a binding already checks, re-reads a file a binding already reads,
converts a binding's result into another container for no caller-visible
reason, or only forwards to a `_sfmtool` binding under another name. Also look
at the package root's `_LAZY_NAMES`, which binds Python names only
(§ [Python names mark what is public](#python-names-mark-what-is-public) and
[python-bindings.md](python-bindings.md) state the root's rules). Confirm by
reading the binding and its Rust code, and cite the duplicated check or read
on both sides.

Out of scope: a Python function that combines several bindings and other
steps into a pipeline, such as an extraction run or a solve.

## Medians and quantiles come from one module

There is one implementation of the median and one of the quantile, in
[`sfmtool_core::numeric`](../crates/sfmtool-core/src/numeric.rs), and every
crate that depends on `sfmtool-core`, `sfm-explorer` and `sfmtool-py`
included, gets its medians and quantiles there. No other function computes a
median, a quantile or a percentile. Code that needs one calls these:

- `median(values)` and `median_in_place(values)` give the median.
- `quantile(values, q)`, `quantile_in_place(values, q)` and
  `quantile_of_sorted(sorted, q)` give the `q`-quantile for `q` in `[0, 1]`.

The plain forms copy the caller's slice. The `_in_place` forms are for a
caller that owns a scratch buffer: `median_in_place` leaves it permuted in no
particular order and `quantile_in_place` leaves it sorted. `quantile_of_sorted`
takes a slice already sorted ascending by `f64::total_cmp`. All five follow
the same rules, which are the contract callers rely on:

- **Ordering is `f64::total_cmp`.** NaN sorts above every finite value (and
  `−NaN` below), so no comparison panics and a minority of NaNs leaves the
  answer finite. A NaN result means a NaN reached that rank of the population.
- **Empty input gives NaN**, never a number, so the caller's "no answer"
  path handles it.
- **The median of an even count is the mean of the two middle values**, as in
  `numpy.median`.
- **The quantile interpolates linearly between order statistics**, numpy's
  default `linear` method: the position is `q·(n − 1)`, and the interpolation
  uses numpy's own arithmetic, so whenever the gap between the two order
  statistics is finite the result equals `numpy.quantile` bit for bit. When one
  of the two is infinite the result is that infinity for any non-zero
  fraction, and only `−∞` against `+∞`, or a NaN input, gives NaN.

This exists because separate copies agreed only on clean data. Six median
copies used three NaN policies: three sorted with `f64::total_cmp`, two with
`partial_cmp().unwrap_or(Ordering::Equal)`, which is not a total order when a
NaN is present and so left the result unspecified, and one with
`partial_cmp().unwrap()`, which panicked on a NaN. For empty input they
returned `0.0`, `None` or NaN, or indexed out of bounds in a release build.
Five private quantile copies and the shared one used three different
interpolation arithmetics, and one of them returned `0.0` for an empty slice.
The same residual population was summarized differently depending on which
module computed it, and none of the differences failed to compile.

The rule is enforced by
[`numeric/tests.rs`](../crates/sfmtool-core/src/numeric/tests.rs).
`the_workspace_has_one_median` and `the_workspace_has_one_quantile` read every
`.rs` file under `crates/`, skipping `target` and `tests` directories and files
named `tests.rs`. The first fails on any `fn` whose name contains `median`, and
the second on any whose name contains `quantile` or `percentile`, ignoring
case, unless it is listed by path and name in `MEDIAN_ALLOWLIST` or
`QUANTILE_ALLOWLIST`. Each entry states why it is not a second copy. Most
entries are helpers named for the statistic they report that call the shared
function, such as `residual_quantiles_px` in `move_camera.rs`. The fix for a
failure is to call `numeric`; an entry is added, with the reason, only for such
a helper or for a genuinely different operation.

The test reads function names only, so look for what it cannot see:

- An inline median: a sort or `select_nth_unstable` followed by indexing the
  middle (`v[n / 2]`), in a function not named for a median.
- A hand-written percentile: an index computed from a fraction
  (`sorted[(0.9 * n as f64) as usize]`) or an interpolation between two order
  statistics.
- A call to a crate's own copy, or a closure or local helper that computes
  one.
- An allowlist entry whose reason says it delegates but whose body computes
  the statistic itself.

Confirm before reporting: show that the code computes an order statistic of a
population, name the `numeric` function that replaces it, and say for which
input the two differ (a NaN, an empty population, an even count, a fraction
between two ranks) or that they agree on every input.

Out of scope:

- The different operations on the allowlists, all in
  `crates/sfmtool-core/src/`: a weighted median
  (`weighted_median` in `geometry/bundle_adjust.rs`), the KD-forest split pivot
  (`median_value` in `features/kdforest/build.rs`, generic over the descriptor
  scalar type, which takes the upper middle and never averages), and
  `iteratively_trimmed_median` in `analysis/scene_scale.rs`, which trims a
  sorted prefix repeatedly and keeps an actual sample on each pass.
- The crates `sfmtool-core` depends on, such as the format crates, which
  cannot call it. `median_sorted` in `sfmtool-sfmr-format/src/depth_stats.rs`
  is listed for that reason.
- Independent reference implementations inside tests, which check the shared
  functions or other code against a plain sort.
- Python code, which uses numpy; the shared functions match `numpy.median`
  and `numpy.quantile`.

## Numeric precision is the smallest the use justifies

Every numeric type in a computation, both the one its values are stored in and
the one its arithmetic is done in, is the smallest precision that analysis and
measurement show is enough for what the result is used for, with speed and
storage weighed beside it. No precision is a default. Pixel colour arithmetic
in particular does not default to `f64`: an 8-bit sample holds under three
significant digits, and an `f32`'s 24 bits leave each rounding about
1/65,000 of a grey level, while an `f64` row holds half the values per vector
instruction and needs twice the memory. Where the choice is made, in the doc
comment of the function or module that makes it or in its spec, the reason is
stated: the range the values take, the digits the use needs, the accumulation
error, and the speed or size it gains.

A **reduction**, a sum of many values, is a different question from per-value
arithmetic, because its rounding error grows with the number of terms.
Its scheme is chosen by its error bound, as the cheapest that meets the digits
needed:

- One running sum of `n` terms in a precision of unit roundoff `u`
  (`2^-24` for `f32`) is bounded by about `n · u` of the sum of the terms'
  magnitudes. For the 130,000 terms of a 480 x 270 image in `f32` that is
  `8 · 10^-3`.
- Summing in blocks, each row and then the row sums, bounds it by about
  `(rows + columns) · u`, `5 · 10^-5` for the same image, at no extra cost,
  and splits the work by rows.
- Pairwise summation bounds it by about `log2(n) · u` at about the same
  arithmetic; compensated (Kahan or Neumaier) summation bounds it by a few `u`
  at about four times the arithmetic.
- An `f64` accumulator is chosen only when no `f32` scheme's bound meets the
  need, or when the terms themselves are `f64` for their own reasons.

A reduction whose result is stored, hashed or compared is summed in a fixed
order, so the result does not depend on the thread count.

This exists because the reading image behind a `.sift` file's sharpness and
gradient structure tensor was computed in `f64` throughout, from 8-bit pixels,
for readings compared within a capture to about three significant digits.
Measured against that, the same pipeline in `f32`, with each row summed and the
row sums totalled, moved the readings by at most `1.1 · 10^-6` of their value
on 140 photographs and made the grey conversion three times faster; one running
`f32` sum instead moved them by up to `2 · 10^-4`. The SIFT detector's input was
a second copy of the same grey conversion in `f64`.

Look for:

- `f64` in a loop over the pixels or samples of an image, a patch or a tile:
  `as f64 / 255.0`, a `Vec<f64>` or `[f64; N]` image buffer, an `f64`
  accumulator of `f32` samples.
- A precision chosen with no stated reason where it is chosen.
- A sum over a whole image or a large population kept in one running `f32`
  accumulator (`.sum::<f32>()` over every pixel), whose `n · u` bound may
  exceed what the use needs.
- A parallel reduction whose order depends on how rayon splits the work, where
  the result is stored or compared.

Confirm before reporting: state the values' range and the digits the use needs,
and show by an error bound or a measurement that the smaller precision meets
that need (or, for a reduction, that the scheme's bound does not), with the
speed or size it changes.

Out of scope: the precision an interface fixes, such as a JSON number, which is
a double, a numpy array a caller passes, a COLMAP database's columns or a
GPU texture format; the choice is then made at that interface rather than in
the code that fills it.

## Every file format has the same I/O shape

Every sfmtool on-disk format, one per format crate (`.sfmr`, `.sift`,
`.matches`, `.camrig`, `.kdf`), has its I/O in `sfmtool.fileio` in the same
shape:
`read_<fmt>`, `read_<fmt>_metadata`, `write_<fmt>` and `verify_<fmt>`, plus a
Rust file class where a reader needs one (`MatchesFile`, `SiftFile`). The
format crate validates what is read and written; the binding converts types
and the Python package adds nothing. This exists so a caller who knows one
format knows them all, and so a check made for one writer holds for every
caller of it.

Tabulate the functions per format and report a format whose I/O lacks one of
them, lives in another module, has a Python reader or writer class, or
validates in Python or in the binding rather than in the format crate. Out of
scope: a function that only one format needs, such as `read_sift_partial`, and
formats sfmtool reads or writes but does not define (COLMAP's binary model and
database, image files).

## Viewer items are no more visible than their users need

In `sfm-explorer`, an item is private when only its module and that module's
descendants use it, and `pub(crate)` (or `pub(super)`) when sibling modules
need it. That includes the methods and fields of types declared in private
modules. Bare `pub` is kept for the crate's public interface, which is what
the `sfm-explorer` binary and `sfmtool-py` call: `run`, `run_with_args` and
`RunError` with its methods in `lib.rs`, and, with the `mcp` feature,
`tool_catalog`, `validate_tool_call` and `CatalogTool` with its fields, which
`lib.rs` re-exports from `mcp/call_check.rs`. This exists because a
declaration's visibility is how a reader learns an item's reach. Every module
of the crate is private, so a bare `pub` inside one claims a reach the item
does not have: it reads as part of the interface the binding uses, and the
reader has to trace the module tree to find out that nothing outside the crate
can name it. The rule is not about dead code: rustc's `dead_code` lint warns
about an unused item in a private module whether it is `pub` or `pub(crate)`.

`lib.rs` sets `#![warn(unreachable_pub)]`, so rustc warns on a bare `pub`
function, method, type, constant, module or `use` that the crate does not
export, and the `lint` job's clippy run in CI (`-D warnings`) fails on it.
Two gaps remain, and they are what to look for:

- **Struct fields.** rustc does not apply `unreachable_pub` to fields, so a
  bare `pub` field in a `pub(crate)` struct passes every build. Grep
  `crates/sfm-explorer/src/` for `^\s*pub [a-z_][a-z0-9_]*\s*:` and for
  tuple fields (`(pub `), and leave out `CatalogTool`'s fields.
- **Windows-only code.** The `lint` job runs on Linux, so code compiled only
  on Windows, such as `platform/windows.rs`, is not checked there; the lint
  only warns in the Windows builds. Grep that code for bare `pub` by hand.

Confirm before reporting: check that the item is not part of the public
interface above and not re-exported by a bare `pub use` on the way to it. Out
of scope: the other crates, whose `pub` items are their public API, and the
integration tests in `crates/sfm-explorer/tests/`, each of which is its own
crate.

## Python names mark what is public

A leading `_` on a Python module, function, class, constant or attribute name
means the name is internal to `sfmtool`: it is not public API. A name with no
`_`, reached by a path with no `_`-prefixed component, is public API. Code
inside `src/sfmtool/` may import an internal name from any other subpackage
(`_commands/match.py` imports `feature_match._run`, and `analyze/summary.py`
imports `camera.cameras._CAMERA_PARAM_NAMES`). Code outside the package, such
as `scripts/`, the docs, spec examples and users' code, does not import
internal names. A test may import an internal name when that name is what it
tests, and otherwise uses public names. This exists so that a name says by
itself whether callers outside the package depend on it, and so whether it
can change without a compatibility cost.

The compiled extension `sfmtool._sfmtool` is internal by this rule. The public
home of its bindings is the module of the same name on the package,
`sfmtool.<submodule>` (`from sfmtool.fileio import read_sfmr`); see
[python-bindings.md](python-bindings.md). Each part of the package has one of
these shapes:

- **Re-exporting subpackage.** Its `__init__.py` imports names from its own
  modules and lists them in `__all__` (`compare/`, `feature_match/`, `strips/`,
  `visualization/`, `xform/`). Its modules are `_`-named, and its public names
  are the ones the `__init__.py` re-exports. A caller imports a re-exported
  name from the subpackage (`from ..visualization import
  render_heatmap_overlay`), not from the `_`-named module that defines it.
- **Module-path subpackage.** Its `__init__.py` re-exports nothing from its own
  modules, and callers import `sfmtool.<subpackage>.<module>` (`align/`,
  `analyze/`, `camera/`, `camrig/`, `colmap/`, `merge/`, `motion/`, `rig/`,
  `sift/`). A plain module name there is public; a module that is not meant as
  API, such as a helper for one sibling, takes `_` (`motion/_recon_console.py`).
- **The `sfmtool` package root** binds the public names of its Python
  submodules through `_LAZY_NAMES` in `__init__.py`, and the binding modules
  and `rig` through `_LAZY_SUBPACKAGES`, so its own modules are `_`-named,
  except `cli.py`, the `sfm` entry point named in `pyproject.toml`, and the
  binding modules (`analysis.py`, `geometry.py`, `fileio.py`, …), each of which
  re-exports one extension submodule. The root re-exports no binding flat.
- **`_commands/`** is internal as a whole, because of its underscore. Inside
  it, a plain name is a command module, one per row in `cli.COMMANDS`, and a
  helper shared by several commands takes `_` (`_range_options.py`) so it is
  not mistaken for a command.

Tests already check part of this: `tests/test_module_layout.py` fails on a
path into the extension from `tests/` or `scripts/` outside its allowlist, and
on a script import that names an internal module or name;
`tests/test_lazy_loading.py` fails when the root binds a binding flat and when
a command module's command has no row in `cli.COMMANDS`. Look for what they do
not check:

- An internal name imported in a spec example, a page of the docs site
  (`sfmtool/sfmtool.github.io`) or a test that does not test that name.
- A plain-named module in a re-exporting subpackage, or a caller that imports
  a re-exported name from the `_`-named module that defines it.
- A plain-named module in a module-path subpackage that only one sibling uses,
  or a plain-named helper in `_commands/` that defines no command.
- A public name of a Python submodule that the package root binds by an import
  line rather than through `_LAZY_NAMES`.

Out of scope: local names inside a function body, and the Rust crates, whose
visibility is declared by the language.

## Importing sfmtool and sfm loads only what is used

`import sfmtool` binds the extension's names and nothing else. The public names
of its Python submodules are bound on first use through the `__getattr__` in
`sfmtool/__init__.py`, from its `_LAZY_NAMES` table. A new public name from a
Python submodule is an entry in that table and in the `if TYPE_CHECKING:`
imports beside it, which type checkers read, not a top-level import. `sfm`
imports a command's module only when that command runs, and lists the commands
in `--help` from `COMMANDS` in `cli.py`. A new command is a row there, its
one-line help is the first sentence of its docstring, and a command module does
no work at import time beyond defining the command. This exists so that
`import sfmtool` and `sfm --help` stay fast: neither imports numpy, OpenCV or
pycolmap, and a command that needs none of them, such as `sfm explorer`, does
not wait for them. [cli/README.md](cli/README.md) § "How `sfm` loads its
commands" describes the mechanism.

`tests/test_lazy_loading.py` checks this: that `import sfmtool` and
`sfm --help` import none of numpy, OpenCV and pycolmap, that running a command
imports only its own module, that each `_LAZY_NAMES` entry resolves to its
submodule and matches the `TYPE_CHECKING` imports, and that `COMMANDS` and the
command modules match. Look for what it does not check: a command module that
reads files, builds large tables or calls into the extension at module level,
and a top-level import of a Python submodule in `sfmtool/__init__.py`, which
the tests notice only when that submodule imports one of the three libraries.
Out of scope: imports inside a command's function body, which run only when
the command does.

## A Rust module with submodules is a file beside its directory

A Rust module that has submodules is a `foo.rs` file beside a `foo/` directory,
not `foo/mod.rs`. A new module uses that form, and no change adds a `mod.rs`.
An existing `mod.rs` is converted when a change restructures its module
anyway. This exists so a module's file
is named after the module, which is what an editor's tab and a search result
show; a tree of files all named `mod.rs` cannot be told apart that way.

Count the two forms with, from the checkout root:

```bash
# modules in the foo.rs-beside-foo/ form
find crates -path '*/target' -prune -o -type d -print |
  while read -r d; do [ -f "$d.rs" ] && echo "$d"; done | wc -l
# remaining mod.rs files
find crates -path '*/target' -prune -o -name mod.rs -print | wc -l
```

This file gives no counts because each conversion changes them; the hygiene
audit reports them with each snapshot. Look for a `mod.rs` added since the
last snapshot (`git log --diff-filter=A -- '*mod.rs'`). Out of scope: a
module without submodules, which is a plain `foo.rs`.

## A settings struct is named *Options

A struct the caller fills with its choices of how a function or algorithm runs
is named `*Options`, usually with an `impl Default`. That holds for RANSAC
thresholds, iteration caps, seeds, gates, sizes and nested settings alike.
`*Params` names the values a model or kernel is evaluated with, computed for
each call rather than chosen: a camera's `K`, `R` and `t` (`CameraParams`), a
GPU uniform block that mirrors a WGSL `struct Params`, the per-keypoint window
geometry SIFT's histogram fills share, the variational flow settings derived
for each pyramid level. `*Config` is not a settings suffix
(`TwoViewGeometryConfig` keeps COLMAP's name for a pair's geometry type), and
the viewer's `*DisplaySettings` and `SplitSettings` are UI state, not function
arguments. The words and the reasons for them are
[GLOSSARY.md](GLOSSARY.md) § "Rust type names", which is the authority on these
suffixes; this standard is the rule that new code follows it. It exists
because a suffix that tells a caller-chosen setting from a computed value
tells a reader which structs to fill and which to leave to the code.

Public settings structs in `sfmtool-core` that still carry `*Params` or
`*Config` are backlog from before the ruling, listed in the glossary entry's
"Direction of travel"; a new settings struct uses `*Options`. Count them with
`grep -rnE 'pub struct \w+(Params|Config)\b' crates/sfmtool-core/src` and
remove from the hits the per-call values the glossary names (`CameraParams`).
Look for a new `*Params` or `*Config` struct holding caller choices, in any
crate. Out of scope: the bindings
and Python, which keep their own names.

## Every source file carries the license header

Every `.rs` file under `crates/` and every `.py` file under `src/`, `tests/`
and `scripts/` opens with the two-line header

```text
Copyright The SfM Tool Authors
SPDX-License-Identifier: Apache-2.0
```

behind the language's comment marker (`//` or `#`), after the shebang where
there is one. This exists so each file states its license where it is read,
including when it is copied out of the repository on its own.
[`crates/sfmtool-core/tests/license_headers.rs`](../crates/sfmtool-core/tests/license_headers.rs)
reads the header back out of every such file and fails with the list of files
that lack it, so there is nothing to look for by hand beyond running it. Out
of scope: other file types, such as WGSL shaders, shell scripts and specs.

## Rustdoc builds without warnings

Every doc comment in the workspace builds under `pixi run doc`
(`cargo doc --workspace --no-deps --document-private-items`) with no
warnings. `[workspace.lints.rustdoc]` in the root `Cargo.toml` denies each
warn-by-default rustdoc lint, and each crate opts in with
`[lints] workspace = true`, so a plain `cargo doc` fails the same way. This
exists because doc comments here carry invariants and links to the specs, and
a link that stops resolving is otherwise noticed by no one.

The gate documents private items, so links written inside private modules are
checked too. Three ways of writing a link keep it building:

- A link to a private item from a **public** item's doc is an error
  (`private_intra_doc_links`). Write it as a plain code span (`` `foo` ``)
  rather than widening the item's visibility.
- A method reference inside an inherent `impl` is written with `Self::`.
- A reference to a sibling private module is written with `super::`.

Look for a crate whose `Cargo.toml` lacks `[lints] workspace = true`, and for
an item made more visible than its users need only so a doc link resolves.
Out of scope: the allow-by-default rustdoc lints the workspace does not turn
on (`missing_crate_level_docs`, `private_doc_tests`, `unescaped_backticks`),
and an `#[allow(rustdoc::invalid_rust_codeblocks)]` on a binding whose doc
comment is a Python docstring, whose indented `Args:` paragraphs rustdoc would
otherwise parse as Rust.

## Code builds at the MSRV and without the viewer's MCP feature

Every crate builds with the minimum supported Rust version, the
`rust-version` in `[workspace.package]` of the root `Cargo.toml`, which every
crate inherits, and not only with the development toolchain that `pixi.toml`
pins. This exists because the PyPI sdist compiles the workspace with the
user's own rustc: wheels are published for x86-64 Linux, x86-64 Windows and
Apple-silicon macOS, and everyone else builds from the sdist. The floor is
kept one release behind the development toolchain, rather than at the highest
version the dependency tree needs today, so a routine dependency bump does not
force an MSRV bump with it. The version is raised in `Cargo.toml` and nowhere
else: the `msrv` job in `.github/workflows/ci.yml` reads it from there.

`sfm-explorer` also builds without warnings with `--no-default-features`, that
is without its `mcp` feature, so an item that only the MCP code uses is gated
on the feature (`#[cfg(feature = "mcp")]`). This exists because `sfmtool-py`
turns `mcp` on for every workspace build, so an ungated item looks used in
every local build and becomes a dead-code warning only in the build without
the feature.

The `msrv` job checks both: it runs `cargo check --workspace` with exactly the
MSRV rustc, then `cargo check -p sfm-explorer --no-default-features`, and
checks the viewer's library that way again with `-D warnings`. Look for a standard-library API
or language feature stabilized after the MSRV, and for an item that only
`mcp/` uses with no `cfg` gate. Confirm the first against the version the
standard-library docs give as "since", and the second with
`cargo check -p sfm-explorer --no-default-features`. Out of
scope: test and bench targets, which the sdist does not compile and the job
does not check at the MSRV; and a dependency's declared `rust-version`, which can be lower than what it
needs (`libsqlite3-sys` needed 1.95 while declaring none), so the job's build
is what counts, not dependency metadata.

## A convention has no documented exceptions

Code that departs from a stated rule is changed to follow it, or the rule is
changed for all code. A spec, doc comment or `AGENTS.md` does not record that
one module is "the exception" to a rule. This exists because a written
exception is read as permission: the next module copies it, and the rule stops
meaning anything.

Look for "the exception", "exception to", "except for", "the one … that does
not" and "unlike the others" next to a stated rule. Each hit where code departs from the rule
is a finding against the code: change the code to follow the rule, or change
the rule for all code, and delete the exception sentence. Out of scope: a
structural fact the rule already allows, such as a binding module being a
package's `__init__.py`, and a boundary that [GLOSSARY.md](GLOSSARY.md)
declares deliberate. A standard's stated scope is not an exception, and
neither is code that breaks a standard and is listed as backlog by an audit:
that code is still a finding, and the fix is still to change it.

## Tests assert something that can fail

Every assertion in a test can fail when the code it guards is wrong. A change
that adds or changes a test shows that it can fail. A fix to a flaky or weak
test shows it by a temporary change to the guarded code that makes the test
fail, reverted before committing. This exists because a test that cannot fail
reports a pass whatever the code does, which is worse than no test.

Look for:

- A test made deterministic by holding, pausing or ordering work in a way
  that also decides the result its key assertion checks. Cover each ordering
  with its own deterministic test instead, so each one still checks the
  guarded behaviour.
- An assertion that holds for any output: a length checked against itself, a
  value compared with the result of the same call, an `is not None` on a
  value that cannot be `None`.
- A test whose fixture already satisfies the assertion before the code under
  test runs.

Confirm before reporting: name the change to the guarded code under which the
test still passes.

## Tests cost what their assertions need

Every test takes only the time its assertions need, and a test is made faster
only in ways that keep everything it checks. The suites run on every change.
CI runs them under coverage instrumentation, which `scripts/coverage.sh`
measured at nearly three times the time of a plain run, and spreads them over
parallel workers, so the slowest tests set the wall time of the whole run. A
test that runs longer than its assertions need costs every contributor that
time on every push.

Thoroughness comes first. A test is made faster by a cheaper input or by
sharing work with other tests, never by dropping a case, weakening or removing
an assertion, or narrowing what the test covers. A change that speeds up a
test keeps every case, and each assertion can still fail as it could before
(see [Tests assert something that can fail](#tests-assert-something-that-can-fail)).
A cut can leave every assertion passing while the input no longer exercises
what the test was written to cover: a sparser synthetic scene can keep a
determinism test passing while the run no longer takes the code paths whose
output the test compares. So a test whose value is that its input exercises
particular events names those events and asserts that each one happens, and a
change that cuts its input checks that each still happens.
Do not write fewer or thinner tests to save time: when covering a behaviour
needs a slow input, the test uses it, and the pull request that adds the test
says why it needs that input. The time limits under "How to check" decide
nothing by themselves. They pick out tests for a closer look, and each test
is judged by the rule above.

Ways to cut a test's time while keeping what it checks:

- **The smallest input that exercises the behaviour:** a few points or
  images, a synthetic scene, where the assertion does not depend on the
  input's size. Size means every dimension that sets the test's cost, not only
  the count of items, and which dimensions apply depends on what the test
  runs:
  - a test that extracts features from real images: the image size, the
    feature cap (`max_num_features`) and whether the image is doubled first
    (`double_image`); in the `sfmtool` extractor the image size and the
    doubling together also set the octave count;
  - a test on synthetic features or points, such as a `.sift` capture
    written directly with no image decoded: the features per item, such as a
    synthetic scene's point density, since indexing and matching scale with
    it. The image size of such a test is a geometry parameter, not a cost;
  - any test: iteration counts (RANSAC iterations, solver rounds, refinement
    passes), sample counts, worker counts, and how many times the test
    repeats a full run.

  Lower each for every expensive step the test runs, not only for the steps
  it checks. Scale linked parameters together: in a walk over frames, raising
  the motion per frame while lowering the target step keeps the same events
  in far fewer frames.
- **One run for several tests:** a result that several tests only read is
  built once, in a module- or session-scoped fixture, and handed out read-only
  where possible. Under `pytest -n`, each worker process builds its own copy
  of such a fixture, so a session-scoped fixture is built at most once per
  worker rather than once per test.
- **The call, not the pipeline:** a test whose subject is that an argument
  reaches a step spies on that step instead of running the whole pipeline to
  see the effect.
- **Measurements outside the suite:** accuracy or speed at a realistic size is
  a measurement. It belongs in a measurements spec and the script that
  produces it, such as
  [grown-plane-normal-measurements.md](core/bench/grown-plane-normal-measurements.md),
  not in a test.

A test that hangs costs without limit and reports nothing, so a hang must
become a failure. A test of code that could hang, such as one that waits for
worker threads to stop, runs that code on another thread and waits for it
with a timeout well above its normal time, failing with a message that says
where it stopped; `workers_stopped_as_they_start_all_stop` in
[`pick_images/source/tests.rs`](../crates/sfmtool-core/src/analysis/pick_images/source/tests.rs)
does this. A test of such code that waits with no timeout is a finding,
whatever its time. A long test command run by hand or by an agent runs under
a time limit too, so a hang stops it rather than leaving it idle.

A pull request that adds a test or changes code that a test runs reports, in
its Testing section, the time of each test it added and the before and after
times of each existing test whose time it changed by more than a second, read
from the durations list that `pixi run test` prints. Whole-suite times can
differ by up to a factor of two between runs on one machine, so compare
per-test times, and repeat a comparison before reporting it.

How to check: `pyproject.toml` sets `--durations=15`, so pytest prints the 15
slowest setup, call and teardown phases at the end of every run
(`pixi run test -- --durations=0` prints all of them). A test, or a fixture's
setup, that takes more than 5 seconds in `pixi run test` on a developer
machine gets a closer look, and so does a Rust test that `cargo test` reports
as "has been running for over 60 seconds" in CI's `test-linux-rust` job. Look
for:

- Phases over 5 seconds in the durations list.
- Several tests that run the same expensive call on the same input.
- A test that lowers a size or resolution for the steps it checks but leaves
  an expensive step at its default.
- A dimension of the input, from the list above, left at a size the
  assertions do not need.

Besides a wait with no timeout on code that could hang (above), a finding is
a test over those limits whose time its assertions do not need: one where a
way above would cut its time with every case and assertion kept.
Confirm before reporting: time the test alone (`pixi run test -- <node id>`,
or `pixi run cargo test -p <crate> <test name>`), and name the way that would
cut it, how each of its cases and assertions is kept, and how each event its
input exercises still happens. A test over the limits that none of the ways
above would make faster without dropping a case or weakening an assertion
meets this standard.
