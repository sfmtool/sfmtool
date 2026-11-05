---
name: audit-hygiene
description: Survey the codebase for organizational drift — oversized files, files that should be split or merged, misleading names, inconsistent naming conventions, duplicated doc comments, directory structures that hurt navigation — and for code that breaks the quality standards in specs/STANDARDS.md (points at infinity, Progress on long-running work, no panics on caller input, no Python wrappers over bindings, medians and quantiles from sfmtool_core::numeric, the smallest numeric precision the use justifies, one I/O shape per file format, viewer item visibility, Python names and lazy loading, Rust module files and *Options settings structs, license headers, rustdoc, MSRV and mcp-less builds, no documented exceptions, tests that can fail, tests that cost only what their assertions need). Use when the user asks to clean up, reorganize, or review codebase structure.
---

# Codebase hygiene audit

As a project grows, files get bigger, coherent designs stretch, conventions fork
at module boundaries, and names and comments drift away from contents. This skill
surveys the whole codebase (Python and Rust) for those smells and produces a
prioritized list of structural fixes.

## Scope

- Python under `src/sfmtool/` and `tests/`
- Rust crates under `crates/`
- Top-level layout (`scripts/`, `specs/`)

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
   is. Where `specs/STANDARDS.md` already states the convention, the minority
   is a breach of that standard, not a split to decide: settings-struct
   suffixes (§ "A settings struct is named *Options"), `mod.rs` against
   `foo.rs` (§ "A Rust module with submodules is a file beside its
   directory"), `_`-prefixed Python names (§ "Python names mark what is
   public") and license headers (§ "Every source file carries the license
   header").
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

### E. Quality standards

Check every standard in [`specs/STANDARDS.md`](../../specs/STANDARDS.md).
That file holds each rule, its reason, what to look for, how to confirm a
finding before reporting it and what is out of scope; this section adds only
how this audit runs them.

- **Check per function, not per convention split.** A breach is a finding
  even where the whole module is consistent with itself. Run the
  mechanical checks 9–21 below, and the standards' tallies in check 5,
  first, then read each hit against the standard it feeds.
- **Cite the standard.** Each finding names the standard by its heading
  and links it (`specs/STANDARDS.md#caller-input-never-panics`).
- **Propose the code fix.** Per STANDARDS.md § "Direction of a fix", the
  proposed fix for a breach changes the code to meet the standard. It is
  never a spec or doc edit that describes the breach. When the code fix is a
  large design change, say so under Effort and Risk and leave the choice to
  the maintainer, rather than proposing the doc edit instead.

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
   has already retired. Give each one `specs/STANDARDS.md` as well, so every
   subtree is checked against the same standards.
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
   `specs/GLOSSARY.md` before recommending either way. Always report three
   tallies with the commands STANDARDS.md gives for them: `mod.rs` files
   against modules in the `foo.rs`-beside-`foo/` form, feeding § "A Rust
   module with submodules is a file beside its directory"; public settings
   structs in `sfmtool-core` still ending in `Params` or `Config`, feeding
   § "A settings struct is named *Options"; and plain-named modules in the
   re-exporting subpackages and in `_commands/` that define no command,
   feeding § "Python names mark what is public".
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
   contract with a visible skip count). Feeds STANDARDS.md § "Points at
   infinity are supported".
10. **Progress coverage** — list public functions that loop over a whole
    reconstruction, run a solver or read or write a whole file, and report how
    many take a `&Progress`. Separately count `Progress::none()` in non-test
    library code and read each one inside a function that itself has a
    `Progress` in scope. Feeds STANDARDS.md § "Long-running functions
    take a Progress".
11. **Progress variants** — grep for `_with_progress`, `_without_progress`,
    `_no_progress`, `_silent`, `_quiet`, `Option<&Progress`, and
    `verbose: bool` / `show_progress` parameters in library code. Each hit is a
    candidate for STANDARDS.md § "One interface, no progress variants";
    report the count even when it is zero.
12. **Caller-input panics** — apply the scope in STANDARDS.md § "Caller
    input never panics". Count `assert!`, `.expect(`, `.unwrap()`, `panic!`
    and `from_row_slice` per crate, leaving out test code (`tests/`
    directories, `tests.rs` files, inline `#[cfg(test)]` modules) and
    `debug_assert!`. In `sfm-explorer` count only `cli/`, `mcp/` and the
    code that parses what it reads from a file: find that code by grepping
    non-test files for `File::open`, `fs::read`, `serde_json::from` and the
    `read_*` format readers, and follow the parsed data from each hit.
    Count `# Panics` doc sections the same way. Then read the hits in
    bindings and in public functions of the library crates that take caller
    data, and keep those a caller's argument can reach. Also grep specs for
    "panics". Report the kept hits as one ranked finding that lists the
    functions, those Python can reach through a binding first, then the rest
    of the public Rust API, each with the argument that reaches the panic.
    Report the raw counts beside it so the next snapshot can measure the
    backlog shrinking.
13. **Python over bindings** — list the Python functions and classes in
    `src/sfmtool/` that call a `_sfmtool` name, and the root `_LAZY_NAMES`
    entries, and read each for re-checked arguments, re-read files or a body
    that only forwards. Feeds STANDARDS.md § "Logic lives in Rust, not in
    Python wrappers".
14. **Format I/O table** — one row per on-disk format, one column per
    function that STANDARDS.md § "Every file format has the same I/O shape"
    lists, and the module each lives in. Report the table even when every
    row is full.
15. **Documented exceptions** — grep `AGENTS.md`, `specs/` and doc comments
    for "the exception", "exception to", "except for", "unlike the others"
    and "the one … that does not" (``the one [a-z_` ]* that (does|is) not``;
    a bare "the one" matches over a thousand lines).
    Feeds STANDARDS.md § "A convention has no documented exceptions".
16. **Viewer visibility** — the `unreachable_pub` lint checks every bare
    `pub` item in `sfm-explorer` except struct fields and code compiled only
    on Windows. Grep `crates/sfm-explorer/src/` for bare `pub` fields
    (`^\s*pub [a-z_][a-z0-9_]*\s*:` and tuple fields `(pub `), leaving out
    `CatalogTool`'s fields, which the crate's public interface exports, and
    grep the Windows-only code (`platform/windows.rs` and
    `#[cfg(target_os = "windows")]` blocks) for any bare `pub`. Report the
    count even when it is zero. Feeds STANDARDS.md § "Viewer items are no
    more visible than their users need".
17. **Build and import gates** — tests and CI check four standards, so this
    check confirms the gates are still in place rather than re-deriving
    them: every `crates/*/Cargo.toml` has `[lints] workspace = true`
    (§ "Rustdoc builds without warnings"); `pixi run cargo check -p
    sfm-explorer --no-default-features` passes with no warning (§ "Code
    builds at the MSRV and without the viewer's MCP feature");
    `tests/test_lazy_loading.py` and `tests/test_module_layout.py` pass
    (§ "Importing sfmtool and sfm loads only what is used", § "Python names
    mark what is public"); and `pixi run cargo test -p sfmtool-core --test
    license_headers` passes (§ "Every source file carries the license
    header"). Then grep the examples in `specs/` for imports of
    `_`-prefixed `sfmtool` names, which no test scans, and the command
    modules for module-level work beyond imports and defining the command.
    Report each gate as passed or failed.
18. **Medians and quantiles** — `pixi run cargo test -p sfmtool-core --lib
    numeric` runs the name scan, so report it as passed or failed. Then grep
    `crates/` for what the scan cannot see, leaving out `numeric.rs` and test
    code (`tests/` directories, `tests.rs` files, inline `#[cfg(test)]`
    modules): an inline median, `select_nth_unstable` and
    `\[\s*([a-z_.]+\.len\(\)|n|len)\s*/\s*2\s*\]`; and a hand-written
    percentile, an index from a fraction times a length
    (`(\.len\(\)|\bn)\s*(-\s*1\s*)?\)?\s*as f64[^;]*as usize`) and an
    interpolation position (`\* *\(?(\w+\.len\(\)|n) *- *1\)? *as f64`).
    Report the hit count per pattern, then read each hit and keep those that
    compute an order statistic of a population. Feeds STANDARDS.md
    § "Medians and quantiles come from one module".
19. **Numeric precision** — grep library code under `crates/` for `f64` in
    pixel and sample loops: `as f64 / 255`, `f64::from(` of a `u8` sample,
    and `vec![0f64` / `vec![0.0f64` / `Vec<f64>` buffers in image, patch,
    tile, flow and feature code, and `= 0.0f64` accumulators in those loops;
    for whole-image sums in one running `f32` accumulator (`.sum::<f32>()`
    over every pixel); and for float reductions whose order rayon chooses
    (`.sum()` or `.reduce(` on a `par_iter`) where the result is stored or
    compared. Report the hit count per
    pattern, then read each hit and keep those whose precision has no stated
    reason or whose stated reason the values' range, the digits the use needs
    and the accumulation error do not support. Feeds STANDARDS.md § "Numeric
    precision is the smallest the use justifies".
20. **Tests that cannot fail** — grep the test code with `rg -nP` for the
    shapes STANDARDS.md § "Tests assert something that can fail" lists: an
    `is not None` (`assert .* is not None` in `tests/`) or its Rust form
    (`assert!\(.*\.is_some\(\)\)` in `crates/`, keeping the hits in test
    code); a length or a value checked against itself
    (`assert len\(([\w.\[\]]+)\) == len\(\1\)` and
    `assert ([\w.\[\]]+) == \1\s*$` in `tests/`,
    `assert_eq!\(([\w.]+), \1\)` in `crates/`); and work held, paused or
    ordered in a test (`Barrier|\bsleep\(|threading\.Event|\.wait\(` in
    `tests/`, and `Barrier|thread::sleep|Condvar|\.wait\(` in `crates/` with
    `-g '**/tests/**' -g '**/tests.rs'`). Report the hit count per pattern,
    then read each hit. Keep an `is not None` or `is_some()` only where the
    value cannot be `None`, and a hold or ordering only where it also decides
    the result the test's key assertion checks. The other two shapes, a value
    compared with the result of the same call and a fixture that already
    satisfies the assertion before the code under test runs, have no
    pattern; look for them while reading these hits and the tests that
    check 21 picks out. Before reporting a test, name the change to the
    guarded code under which it still passes. The proposed fix makes the
    assertion able to fail, or covers each ordering with its own
    deterministic test; it never deletes a case. Feeds STANDARDS.md § "Tests
    assert something that can fail".
21. **Test time** — run the Python suite with every duration listed, then
    list and sum the setup and call phases over 5 seconds:

    ```bash
    out=$(mktemp)
    pixi run test -- --durations=0 > "$out" 2>&1
    awk '/slowest durations/ {f=1} f && ($2 == "setup" || $2 == "call") && $1 + 0 > 5 {n++; s += $1; print} END {printf "%d phases, %.1f s\n", n, s}' "$out"
    ```

    The `test` task in `pixi.toml` runs `pytest -n auto --dist worksteal`,
    so these are the times of a normal `pixi run test`, the run the
    standard's 5-second limit is stated for. For Rust, read the "has been
    running for over 60 seconds" lines of the `test-linux-rust` job in the
    latest successful CI run on `main`:

    ```bash
    run=$(gh run list --repo sfmtool/sfmtool --workflow ci.yml --branch main --status success --limit 1 --json databaseId --jq '.[0].databaseId')
    job=$(gh run view "$run" --repo sfmtool/sfmtool --json jobs --jq '.jobs[] | select(.name == "test-linux-rust") | .databaseId')
    gh run view --repo sfmtool/sfmtool --job "$job" --log | grep "has been running for over 60 seconds"
    ```

    Report the number of Python phases over 5 seconds with their summed
    seconds, and the number of Rust tests over 60 seconds, even when they
    are zero, so the next snapshot can compare. Whole-suite times can differ
    by up to a factor of two between runs, so compare per-test times with
    the last snapshot, not the suite's wall time. The limits only pick tests
    for a closer look; a test over them is not a finding by itself. Read
    each one: group the tests that run the same expensive call on the same
    input, and note a test that lowers a size for the steps it checks but
    leaves an expensive step at its default. Keep only a test that one of
    the standard's ways (the smallest input that exercises the behaviour,
    one run for several tests, the call rather than the pipeline, a
    measurement outside the suite) would make faster with every case and
    assertion kept. Before reporting it, time it alone (`pixi run test --
    <node id>`, or `pixi run cargo test -p <crate> <test name>`), and name
    the way and how each of its cases and assertions is kept. A proposed fix
    never drops a case, weakens or removes an assertion, or narrows what the
    test covers. A test over the limits that no way would make faster
    without doing one of those meets the standard and is not a finding.
    Feeds STANDARDS.md § "Tests cost what their assertions need".

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

For a breach of a standard in `specs/STANDARDS.md` or another rule in
`AGENTS.md`, the proposed fix changes the code and names the standard; it is
never a spec or doc edit that records the breach as how the code works.

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
