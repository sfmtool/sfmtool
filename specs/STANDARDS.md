# Quality Standards

The rules every library function, binding, Python module, test and spec in
sfmtool is held to, whatever module it is in. Each section states one standard:
the rule, why it exists, how to check code against it and what is out of scope.
Every audit skill checks against this file, and every implementer and auditor
agent that makes or reviews a change reads it, so the audit that writes a
finding and the agents that fix it work from the same text. A finding or a fix
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

Look for:

- `assert!`, `expect`, `unwrap`, `panic!` or unchecked indexing on argument
  data in a public function: lengths that must match, indices that must be in
  range, a shape that must hold.
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
(`AGENTS.md` § "Python names and privacy" and
[python-bindings.md](python-bindings.md) state the root's rules). Confirm by
reading the binding and its Rust code, and cite the duplicated check or read
on both sides.

Out of scope: a Python function that combines several bindings and other
steps into a pipeline, such as an extraction run or a solve.

## Every file format has the same I/O shape

Every on-disk format's I/O is in `sfmtool.fileio` and has the same shape:
`read_<fmt>`, `read_<fmt>_metadata`, `write_<fmt>` and `verify_<fmt>`, plus a
Rust file class where a reader needs one (`MatchesFile`, `SiftFile`). The
format crate validates what is read and written; the binding converts types
and the Python package adds nothing. This exists so a caller who knows one
format knows them all, and so a check made for one writer holds for every
caller of it.

Tabulate the functions per format and report a format whose I/O lacks one of
them, lives in another module, has a Python reader or writer class, or
validates in Python or in the binding rather than in the format crate. Out of
scope: a function that only one format needs, such as `read_sift_partial`.

## A convention has no documented exceptions

Code that departs from a stated rule is changed to follow it, or the rule is
changed for all code. A spec, doc comment or `AGENTS.md` does not record that
one module is "the exception" to a rule. This exists because a written
exception is read as permission: the next module copies it, and the rule stops
meaning anything.

Look for "the exception", "except for", "the one … that does not" and "unlike
the others" next to a stated rule. Each hit where code departs from the rule
is a finding against the code: change the code to follow the rule, or change
the rule for all code, and delete the exception sentence. Out of scope: a
structural fact the rule already allows, such as a binding module being a
package's `__init__.py`, and a boundary that [GLOSSARY.md](GLOSSARY.md)
declares deliberate.

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
