---
name: audit-hygiene
description: Survey the codebase for organizational drift — oversized files, files that should be split or merged, misleading names, inconsistent naming conventions, duplicated doc comments, directory structures that hurt navigation. Use when the user asks to clean up, reorganize, or review codebase structure.
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

### C. Naming and convention consistency

8. **Conventions that fork at a module boundary.** A convention held uniformly on
   both sides of a line but differing across it: `*Options` in one module vs
   `*Params` in its siblings; underscore-private file names in one subpackage and
   plain names in the next; SPDX headers on some test files and not others; error
   messages capitalized in one crate and lowercase in another. Each half looks
   self-consistent from inside, which is why these survive review. State which
   spelling is the house majority and recommend converging on it.
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

## How to work

1. Get a size overview: file line counts per directory.
2. Run the mechanical checks below and keep the raw numbers — they are the
   report's evidence and the next snapshot's baseline.
3. Sample the largest files and skim their structure — count distinct top-level
   concerns.
4. Dispatch `Agent` subagents in parallel over subtrees (e.g., one for
   `src/sfmtool/feature_match/`, one for `crates/sfmtool-core/`) to get focused
   assessments. Give each one the convention majorities from step 2 so their
   naming findings are comparable.
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
   license headers, error message capitalization, CLI flag shapes.
6. **Public-name leak scan** — every name reachable from Python or the CLI,
   checked for implementation-detail suffixes and for a spelling that disagrees
   with its siblings.
7. **Reference integrity** — every `specs/…` path cited from code resolves to a
   file, and every area's `specs/README.md` index lists the specs actually
   present. Link rot only, no reading; whether the linked spec is *right* is
   `audit-specs`' call.

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
