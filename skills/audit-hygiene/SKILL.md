---
name: audit-hygiene
description: Survey the codebase for organizational drift — oversized files, files that should be split or merged, misleading names, inconsistent naming conventions, duplicated doc comments, directory structures that hurt navigation — and for library functions that break the interface contracts (support for points at infinity, a Progress argument on long-running work, no with/without-progress variants). Use when the user asks to clean up, reorganize, or review codebase structure.
---

# Codebase hygiene audit

As a project grows, files get bigger, coherent designs stretch, conventions fork
at module boundaries, and names and comments drift away from contents. This skill
surveys the whole codebase (Python and Rust) for those smells and produces a
prioritized list of structural fixes.

## Scope

- Python under `src/sfmtool/` and `tests/`
- Rust crates under `crates/`
- Top-level layout (`scripts/`, `specs/`, `docs/`)

## What to look for

### A. Structure

1. **Oversized files** — modules or source files that have grown past what their
   purpose justifies. Look at line count, but more importantly at whether the file
   holds multiple distinct concerns.
2. **Files that should be combined** — small files that fragment a single concern,
   or near-duplicates.
3. **Misleading names** — files whose name no longer describes what's inside (e.g.,
   grew to cover a second topic, or the original concept was renamed but the file
   wasn't).
4. **Directory-level smells** — directories that are flat when they should be
   grouped, grouped when they should be flat, or where the grouping no longer
   matches how the code is actually used.
5. **Dead or near-dead code** — modules referenced only by tests, commented-out
   blocks, `_foo_old.py`-style leftovers.

### B. Duplication

6. **Duplication against an abstraction that already exists.** The highest-value
   finding class in this repo's history, and it does not correlate with file size.
   Two shapes:
   - A shared helper exists and callers hand-inline it anyway (`numeric::median`
     re-implemented in `resect_images.rs`; `parse_patch_window` inlined in five of
     eight bindings).
   - **Sibling files that are 50% byte-identical**, each too small to trip a
     size sweep. Compare parallel families explicitly: same-named files across
     sibling module directories, per-command modules, per-backend modules.
   Do not rely on the size overview to surface these — run a dedicated pass
   (mechanical checks 3 and 4 below).
7. **Invariants maintained only by a comment.** A doc comment saying *mirrors X* /
   *must match Y* / *kept in sync with Z* / *local copy of W* is the code admitting
   a duplication with no enforcement behind it. Each one is either a finding or a
   deliberate, justified copy — read it and decide, don't assume either way. The
   fix for a finding is usually one shared constant plus a test, not a rewrite.

   A comment names only the copies its author knew about. Before stating how
   many copies there are, search for the duplicated **content** itself (the
   variant list, the literal, the formula) across the workspace, the bindings
   included, and cite the count that search found. A function that already
   holds the same content under some name is the shared home to propose, and
   its name is the one to use. A 2026-10-08 finding that proposed
   `has_bare_focal` counted two copies of a camera-model list by following a
   *mirrors* comment; there were four, one of them already a public function
   named `focal_is_releasable`.

   **Check a proposed name against everything it would classify.** A name for
   a new shared predicate, constant or set is a claim about which items belong.
   Enumerate the whole domain (every variant of the enum, every registered
   model) and confirm the name is true of exactly the items in the list and
   false of every other. If a natural-sounding property also holds for items
   outside the list, it is the wrong name: name the reason the list exists,
   which is usually stated in the comment at the copy you are reading.
   `has_bare_focal` failed this check: `SIMPLE_RADIAL`, `RADIAL` and
   `RADIAL_FISHEYE` also have a single focal, but bundle adjustment does not
   release it.

   **Put the shared item in the module whose reasoning it encodes, not on the
   type it lists.** A set of enum variants is often a fact about an
   algorithm (which models a solver can handle, which formats a reader
   supports), not a fact about the enum. Read why the list exists. If the
   reason is in an algorithm's code (a derivative, a kernel's slots, a
   format's fields), the shared home is in that algorithm's module, and the
   enum's own module stays free of it. A method on the enum is right only when
   the property holds for the type whatever code uses it. The release list
   above is a property of the bundle adjustment kernel's analytic focal
   column, so it belongs in `geometry::bundle_adjust`, not on `CameraModel`.
   Code on the type that copies the algorithm's list (a setter that refuses
   the models the algorithm does not release) is a second finding: fix that
   code to state only the type's own fact.

### C. Naming and convention consistency

**Read [`specs/GLOSSARY.md`](../../specs/GLOSSARY.md) before running any check in
this section, and treat it as authoritative.** Where it names a preferred word,
that is the answer no matter which spelling the tree currently holds more of,
and a file still on the old word is a finding rather than evidence. The tallies
below are for splits the glossary has **not** ruled on.

This matters because a majority is wrong often enough to be dangerous, in two
ways the count itself cannot distinguish:

- **A word can win by being written first.** In the bench, `surfel` outnumbered
  `patch` 132 to a handful, and `patch` was the right answer -- it is the
  PatchMatch term and it already carries the geometry, which is an argument a
  tally cannot see. Converging on the majority would have deleted the correct
  name.
- **A migration reads exactly like drift.** The em-dash is still the majority
  across `specs/` by roughly seven to one, while the files recent work has
  rewritten carry none at all. A tally recommends putting them back. Note also
  that what replaces an em-dash is a varied sentence rather than one substitute
  character, so this split cannot be scored as a two-way spelling contest in the
  first place: some conventions are not countable, and a tally that forces them
  into a count will report something false with a number attached.

So when a tally and the glossary disagree, the glossary wins; when a tally finds
a split the glossary is silent on, the finding is *"decide this and record it"*,
and the recommendation names the candidates and the argument for each rather
than just the bigger pile. Add the ruling to the glossary as part of the fix.

8. **Conventions that fork at a module boundary.** A convention held uniformly on
   both sides of a line but differing across it: `*Options` in one module vs
   `*Params` in its siblings; underscore-private file names in one subpackage and
   plain names in the next; SPDX headers on some test files and not others; error
   messages capitalized in one crate and lowercase in another. Each half looks
   self-consistent from inside, which is why these survive review. Check the
   glossary first; failing a ruling there, state the house majority, say which
   side is newer, and recommend on the argument rather than the count alone.
   A boundary the glossary declares **deliberate** -- `patch` in the bench and
   `surfel` in the renderer -- is not a finding, and eroding it from either side
   is.
9. **Names in a parallel family that don't parallel.** Sibling commands, sibling
   transforms, sibling bindings methods: check that the family's entry point,
   suffix, and argument names follow one rule, and name the exceptions.
10. **Implementation detail leaking into a public name.** Language-boundary
    suffixes (`_py`, `_rs`), internal type names, or a private module's vocabulary
    surfacing in a user-facing API — a Python keyword argument, a CLI flag, an
    on-disk key. These are the most expensive to fix later, so flag them early
    and small.
11. **The same concept under two names in one API.** Two spellings for one thing
    (`indexes`/`indices`, `view_indices`/`member_views`, `load_`/`read_`) matter
    most where they meet the user; inside one function body they usually don't.
    Weight findings by how public the name is.

### D. Comment and doc economy

12. **Doc bloat concentrated in one layer.** Compare doc-lines-to-code-lines per
    crate, and per item for wrapper layers. A binding or wrapper whose docs
    outweigh a parameter reference is usually carrying the layer below it a second
    time.
13. **A doc comment that restates its callee's doc comment.** A `sfmtool-py`
    method re-deriving the argument its `sfmtool-core` function already makes; one
    `impl` block repeating the module doc above it. Same finding class as check 6,
    in prose. Whether a doc block restates a **spec** is `audit-specs`' finding,
    not this one — it means reading the spec, so leave it there.

Do not flag ordinary API reference prose. A long `Args:` block on a
Python-visible binding is doing a job — a REPL user cannot click into `specs/`.
The finding is rationale where reference belongs, not length by itself.

### E. Library-wide interface contracts

These are rules every library function in `sfmtool` (the Rust crates, the
`sfmtool-py` bindings and the Python package) is held to, whatever module it is
in. A violation is a finding even where the whole module is consistent with
itself, so check them per function, not per convention split.

14. **Points at infinity are supported.** A reconstruction's points are
    homogeneous (`positions_xyzw`, a `w = 0` row is a unit direction; see
    [`specs/formats/sfmr-file-format.md`](../../specs/formats/sfmr-file-format.md)),
    and every library function that takes, returns or iterates over a
    reconstruction's points must accept points at infinity in its interface and
    handle them correctly. Look for:
    - An interface that can only express finite points: a `positions_xyz` /
      `[f64; 3]` / `Vec3` parameter or return value where the data can hold
      `w = 0` rows, or a Python wrapper that slices `[:, :3]` off an xyzw array.
    - A division by `w` with no `w = 0` branch, which turns a direction into
      `inf`/`NaN` and then propagates it.
    - Wrong geometry for a direction: translation or scale applied to a `w = 0`
      row (only the rotation acts on it), a depth, distance, centroid, bounding
      box or spatial index computed over directions as if they were positions,
      or a reprojection or triangulation path that assumes a finite point.
    - Points at infinity dropped silently. A function that has a real reason to
      work on finite points only must say so in its contract and must skip or
      reject `w = 0` rows visibly (a count in its result, an error), not filter
      them out unreported. Converting directions to finite points is
      `materialize_points_at_infinity`'s job, for a consumer that cannot store
      `w = 0`, and should not be reinvented inline.
    Confirm the bug before reporting it: read the function and, where cheap,
    say what input shows it (for example a reconstruction with one `w = 0`
    point passed to the transform). A function that never sees reconstruction
    points (image warping, SIFT, descriptor matching) is out of scope.

15. **Long-running functions take a `Progress`.** A library function whose
    running time can exceed roughly a second on realistic input must take a
    `&Progress` (`sfmtool_progress::Progress`, re-exported as
    `sfmtool_core::progress::Progress`) and report and check cancellation
    through it. Treat as above the threshold: a loop over every image, image
    pair, track or point of a reconstruction; an iterative solver or RANSAC run;
    reading or writing a whole `.sfmr`, `.sift`, `.matches` or `.camrig` file;
    and any function that calls one that takes a `Progress`. Also report:
    - A function that takes a `Progress` and then calls a long-running callee
      with `&Progress::none()` instead of passing its own down (or a `phase` or
      `split` of it). The caller's bar stalls and cancellation stops working
      for that stage.
    - A Python-facing binding of such a function that offers no way to pass
      progress in (`ProgressCounter`, see `crates/sfmtool-py/src/py_progress.rs`).
    Do not report short functions (a single reprojection, a per-point kernel
    called from inside a loop that already reports): the threshold is about the
    whole call, and a `Progress` on every helper is noise.

16. **One interface, not a with-progress and a without-progress variant.** A
    caller that wants no reporting passes `&Progress::none()`. A pair such as
    `foo` / `foo_with_progress`, `foo` / `foo_silent`, or `foo` that only
    forwards to `foo_inner(…, &Progress::none())` while both are public, is a
    finding: merge them into one function that takes `&Progress` and update
    the callers. The same applies to `Option<&Progress>` parameters and to a
    `verbose: bool` or `show_progress: bool` flag next to or instead of a
    `Progress`, which are a second interface in a different form. A private
    helper that takes no `Progress` because it is short is not a variant.

## How to work

1. Get a size overview: file line counts per directory.
2. Run the mechanical checks below and keep the raw numbers — they are the
   report's evidence and the next snapshot's baseline.
3. Sample the largest files and skim their structure — count distinct top-level
   concerns.
4. Dispatch `Agent` subagents in parallel over subtrees (e.g., one for
   `src/sfmtool/feature_match/`, one for `crates/sfmtool-core/`) to get focused
   assessments. Give each one `specs/GLOSSARY.md` and the convention majorities
   from step 2, in that order of authority, so their naming findings are
   comparable and none of them recommends converging onto a word the glossary
   has already retired. Give each one the three contracts in section E as
   well, so every subtree is checked against them.
5. Consolidate findings, removing duplicates and ranking.

### Mechanical checks

Cheap, repeatable, and comparable across snapshots. Run them, cite the numbers,
and note the ones that came back clean — an acquittal with a number behind it is
worth more than a paragraph.

1. **Doc density** — doc lines vs total lines per crate, and the longest
   contiguous comment blocks in the tree. Feeds checks 12–13.
2. **Doc-to-code ratio per item** — for a wrapper layer, the doc line count
   against the function body line count. Anything over ~0.5x deserves a read.
3. **Cross-file duplicate prose and code** — normalize long lines (lowercase,
   strip markup) and bucket them; report file pairs sharing many. Catches
   sibling-file duplication and callee-restating docs in one pass. Code only;
   `audit-specs` runs the same scan across `specs/`.
4. **Parallel-family diff** — for each set of same-named files across sibling
   directories (`*/prof.rs`, `_commands/*.py`, `patches/*.rs`), count identical
   lines pairwise.
5. **Convention tallies** — count both spellings of each candidate convention and
   report the split with locations of the minority: option-bag type suffixes,
   `indexes`/`indices`, verb prefixes on public functions, private-module naming,
   license headers, error message capitalization, CLI flag shapes. Report the
   minority's **age** beside its size -- if the newer files hold it, the split is
   a migration and the majority is the thing to fix. Check each split against
   `specs/GLOSSARY.md` before recommending either way.
6. **Public-name leak scan** — every name reachable from Python or the CLI,
   checked for implementation-detail suffixes and for a spelling that disagrees
   with its siblings.
7. **Reference integrity** — every `specs/…` path cited from code resolves to a
   file, and every area's `specs/README.md` index lists the specs actually
   present. Link rot only, no reading; whether the linked spec is *right* is
   `audit-specs`' call.
8. **Glossary conformance** — for every entry in `specs/GLOSSARY.md`, count uses
   of the preferred word and of what it replaced, **inside that entry's stated
   scope only**. A word outside its scope is not a violation: the glossary's
   boundaries are entries too, so `surfel` in the renderer is a pass and
   `surfel` in the bench is a finding. Report the residue per file, and report
   the entries that came back clean -- a retired word with a zero behind it is
   how the glossary earns its keep. An entry whose residue is large and *newer*
   than the ruling is the one case where the glossary itself is the suspect:
   say so rather than filing a hundred findings.
9. **Points-at-infinity scan** — grep library code for finite-only point
   interfaces and unguarded homogeneous divisions: `positions_xyz\b`,
   `\[:, *:3\]` on an xyzw array, `/ *w\b`, and public signatures taking
   `[f64; 3]` / `Vec3` slices of reconstruction points. Report the hit count
   per pattern, then read each hit and keep only those with no `w = 0`
   handling (`is_at_infinity`, an explicit branch, a documented finite-only
   contract with a visible skip count). Feeds check 14.
10. **Progress coverage** — list public functions that loop over a whole
    reconstruction, run a solver or read or write a whole file, and report how
    many take a `&Progress`. Separately count `Progress::none()` in non-test
    library code and read each one inside a function that itself has a
    `Progress` in scope. Feeds check 15.
11. **Progress variants** — grep for `_with_progress`, `_without_progress`,
    `_no_progress`, `_silent`, `_quiet`, `Option<&Progress`, and
    `verbose: bool` / `show_progress` parameters in library code. Each hit is a
    candidate for check 16; report the count even when it is zero.

## Output

One section per recommendation:

```
**<short title>**
- Location: <file or dir path>
- Problem: <specific smell — "this 1200-line file mixes matching, filtering, and I/O">
- Proposed fix: <split into X and Y | merge with Z | rename to W | regroup under foo/>
- Effort: <low | medium | high>
- Risk: <low | medium | high> — <what could break>
```

End with a **Top 3** section: the fixes with the best effort-to-value ratio.

### The "Explicitly not flagged" list

Carry one, but treat it as a measurement, not a verdict. Every entry states the
figure and the date it was taken, and the section opens by saying so: this repo
has repeatedly grown a cleared file 30–90% within a month of clearing it. Prefer
acquitting with a mechanical number (longest function, duplicate-line count) that
the next snapshot can re-run, over a prose judgement it would have to re-derive.

## Saving the report

Write the full report to `reports/<date>-hygiene-audit.md`, where `<date>` is
today's date in `YYYY-MM-DD` form (get it with `date +%F`). Create the
`reports/` directory if it doesn't exist. Each run is a dated snapshot — do not
overwrite a prior day's report. Carry forward every still-open finding from the
report being superseded, re-measured at the current HEAD rather than copied, and
retire the old snapshot per the rules in `AGENTS.md`. After saving, tell the user
the path and give a short summary of the Top 3 in the conversation; don't paste
the whole report back.

## Guidelines

- Be specific. "X is too big" is useless; "X is 900 lines covering matching AND
  geometric filtering AND serialization — split serialization into a sibling
  file" is useful.
- Cite line counts or symbol counts when flagging size; cite both sides of the
  split when flagging a convention.
- Don't flag a file just because it's long — long is fine if the file has a single
  coherent purpose.
- Don't flag a difference just because it's a difference. Check whether the
  minority spelling is carrying meaning before calling it drift: a batched binding
  legitimately pluralizes its core counterpart's name, `estimate_` and `compute_`
  legitimately distinguish fitting from evaluation, and a suffix may be domain
  notation rather than a language tag. Read the item before reporting it.
- When a convention is real but unwritten, say so, and propose where it should be
  written down (`AGENTS.md`, the area's `specs/README.md`). Undocumented
  conventions are how the next module forks one.
- Prefer fixes that end in enforcement — a shared constant, a compile-time assert,
  a clippy `disallowed_methods` entry, a grep-based test — over fixes that end in
  a doc comment. This repo has a demonstrated pattern of re-breaking contracts
  that live only in prose.
- Do not modify any code during this skill — it is read-only analysis.
