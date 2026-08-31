---
name: audit-specs
description: Audit every spec in specs/ and docs/ against the corresponding code and produce a consistency report. Use when the user asks to check spec/code drift, verify specs are still accurate, or review the state of specifications.
---

# Audit specs against code

Walk every specification, design document, and user documentation under `specs/` and `docs/`, compare each to the code that implements it, and produce a single consolidated report.

## Scope

There are ~120 specs. Reading every one against its code exhausts the context
window and produces a report nobody finishes, so the audit runs at two depths:

- **Every spec, mechanically.** The checks under "Mechanical checks" below run
  over the whole corpus. They cost almost no context, they are the report's
  evidence, and they are what makes a bounded run trustworthy: nothing is
  *unexamined*, only un-read.
- **A sample, deeply.** Around ten specs get the full spec↔code reading. How they
  are chosen is under "Choosing the sample".

The deep pass is bidirectional:

- **Spec → code.** For each sampled file under `specs/` (recursively — `specs/cli/<category>/`, `specs/core/<module>/`, `specs/formats/`, `specs/gui/`, `specs/workspace/`) and under `docs/` — locate the implementing code and compare.
- **Code → spec.** Every significant code surface — CLI commands in `src/sfmtool/_commands/`, crates in `crates/`, modules in `src/sfmtool/`, user-facing file formats — check whether a spec or doc covers it. This half stays exhaustive but shallow: it is a coverage list, built from the import/command inventory, not a per-surface essay. Unspecced code is a first-class finding, not a footnote.

### Choosing the sample

Two pools, in this order:

1. **Mechanical hits go in regardless.** A spec whose documented default
   disagrees with the code, or that shares dozens of prose lines with its
   implementation, has earned a read — it should not need luck to get one. These
   are evidence, not a sample.
2. **Random fill, from what has been audited least recently.** Take the rest of
   the budget at random from specs not covered by recent spec-audit reports.
   Prior coverage is recoverable from the reports themselves — no state file.

```bash
SEED=$RANDOM; N=10                     # record SEED in the report
find specs docs -name '*.md' ! -name 'README.md' ! -name 'TEMPLATE.md' | sort > /tmp/all.txt
grep -hoE '^### (specs|docs)/[a-zA-Z0-9/_.-]+\.md' reports/*-spec-audit.md 2>/dev/null \
  | sed 's/^### //' | sort -u > /tmp/done.txt
comm -23 /tmp/all.txt /tmp/done.txt > /tmp/pool.txt   # never-audited first
[ -s /tmp/pool.txt ] || cp /tmp/all.txt /tmp/pool.txt # all covered? reopen the field
shuf -n "$N" --random-source=<(yes "$SEED") /tmp/pool.txt
```

Report the seed, the pool size, and the selected list, so the run is reproducible
and the next one can see what it skipped. Raise `N` when the user asks for a
deeper pass, and say in the report what budget was used.

Two things override the sample: a spec the user names, and a spec covering code
that changed materially since the last audit — check `git log` over the
implementing paths and pull those in.

A spec and its implementing code are rarely the only two copies of a design. Doc
comments are a third, and they drift like any other. While you have both texts
open — which no other skill does — also check the three things below.

### Third copies: doc comments that re-derive the spec

The house style is: a doc comment says what the reader of *this code* needs — the
contract, the units, the invariant, the surprising bit — then links
`specs/<area>/<file>.md` for the design and the empirical justification. A doc
block that links the spec **and** re-derives its argument is a finding, and the
copy is what drifts, because nothing checks it.

Flag design rationale living outside `specs/`: why a threshold sits where it does,
what trade a parameter makes, the tuning population a default was fit on. Do not
flag ordinary API reference. A long `Args:` block on a Python-visible binding is
doing a job a spec cannot — a REPL user cannot click into `specs/`. The finding is
rationale where reference belongs, not length by itself.

The wrapper layers are where this concentrates, because a binding faces users who
have no access to the spec. When you flag one, say which of the three copies
should shrink; usually it is the wrapper, down to contract-plus-link.

### Documented defaults vs actual defaults

A spec's parameter table is the copy most likely to be silently wrong, and the
check is mechanical (see below). A default that disagrees is a **behavioural**
finding, not a documentation nit — report it as such, and say which side is right.

### Spec shape, against `specs/TEMPLATE.md`

`specs/TEMPLATE.md` gives the default order: **Purpose → Rust API → Theory →
Implementation notes → Parameters → Python bindings → Testing → Non-goals → Open
questions.** It is a default, and the template says so: a spec may drop, merge or
reorder sections where the subject is served better, and several of the best
specs predate the template and use their own vocabulary throughout.

**So do not audit for conformance.** A spec that departs and works is not a
finding, and reporting it as one trains everybody to reach for the form instead of
thinking about the reader. Ask what a reader needs and whether they can get it,
not whether the headings match. The five failures below are the ones that hold
whatever arrangement a spec chose — each is something a reader is *missing*, not
something out of order. Only failure 1 is absolute; for the others, a spec that
supplies the thing under a different name, in a different place, or by a means the
template did not anticipate has satisfied them.

When you find a departure that works better than the template for its kind of
subject, say so in the report — that is a proposed amendment to
`specs/TEMPLATE.md`, and it is more valuable than most findings.

1. **An opening that assumes the reader is already inside.** The first paragraph
   is for someone who has read none of the other specs — a competent engineer who
   knows roughly what SfM is and nothing about this codebase's vocabulary. Read it
   cold, deliberately forgetting the rest of the corpus, and ask: *can I say what
   this is for, and whether it is the spec I wanted?*

   Concrete failures, in rough order of how often they occur: a bare symbol name
   in the opening sentence; a formula before any words; a term this repo defined
   in another spec used as though it were common usage; a first sentence that
   needs a followed link to parse; an opening that describes a *change* ("the
   original implementation ran…", "there is no facility to…") instead of what the
   thing is; and the quiet one — a paragraph that is entirely true, entirely
   precise, and never says what the thing is **for**.

   Quote the opening in the finding and propose a replacement first sentence.
   This is the cheapest fix in the report and the one that compounds: it is what
   every future reader hits first, including the next audit.
2. **A reader cannot find out what to call.** Usually this means a `specs/core/`
   spec carries no Rust types and signatures at all, or buries them under a
   derivation — the most common shortfall in the corpus. The requirement is the
   reader's, not the format's: someone who wants to *use* this should not have to
   read the maths first. A spec that conveys the interface in prose, in an entry
   table, or as a set of flags has met it. A spec that legitimately teaches
   before it declares — a numerical hazard you cannot state an API against until
   the reader knows it exists — has met it too; check that the teaching is
   actually load-bearing rather than habit.
3. **An interface with no rationale and no usage.** Signatures alone are
   recoverable from the code and drift the moment it changes. What earns the
   section is *why it is shaped this way* — why these arguments, why one struct
   back instead of three returns, what the alternative would have cost — plus the
   shortest real call. An interface section that is only a code block is a
   transcription; say so.
4. **Implementation notes that transcribe the code.** The test: would the sentence
   need editing after a behaviour-preserving refactor? Then it is describing the
   body rather than the design. Notes should carry what the code cannot — a
   cross-function invariant, why this loop order, a numerical hazard and the
   formula chosen to dodge it, which primitive is the single source of truth.
5. **Work-order residue.** Imperative construction language ("new module X",
   "declare `pub mod` in `lib.rs`"), or sections that only make sense to someone
   who was in the room: *Prior state (before this change)*, *Implementation plan*,
   *Phase 2*, *Consumers & migration*. A standing spec is present-tense. Recommend
   converting or deleting, and check whether *Open questions* entries are still
   open — a settled one reads as live.

Weigh these by how much a reader is misled, not by how far the spec is from the
template. A spec whose interface section is missing but whose theory is exact is
in better shape than one that fills every heading with prose that no longer
matches the code.

## How to work

1. Run the mechanical checks below. They cover every spec and cost almost no
   context, and their output both seeds the sample and becomes the report's
   corpus-wide section.
2. Choose the sample, per "Choosing the sample" above. Record the seed.
3. List every significant code surface (each CLI command, each crate, each major module, each file format) and mark which have a spec. This is a table, not prose.
4. For each **sampled** spec, read it, then read the implementing code. Dispatch `Agent` subagents in parallel — one per spec, or one per small group — so the main context stays clean; each returns a structured summary in the report's per-spec shape. This is what keeps a ten-spec run affordable, so do it even when the sample is small.
5. Consolidate into a single report.

### Mechanical checks

Run these across the whole tree before reading anything, and cite the numbers.
Each one narrows 131 specs to a handful worth close attention.

1. **Documented defaults vs actual defaults.** Parse parameter tables
   (`| `name` | `value` | …`) out of every spec and defaults out of the code —
   `#[pyo3(signature = (…))]`, `impl Default`, Click `default=`. Diff them.
   Normalize numeric spellings (`0.9` / `0.90`) and expect false positives from
   table parsing, so verify each hit against the code before reporting it. Real
   hits are behavioural drift and belong in **Top priorities**.
2. **Duplicate prose between a spec and its implementing code.** Normalize long
   lines (lowercase, strip markup) from every spec and every doc comment, bucket
   them, and report file pairs sharing many. A spec↔code pair sharing dozens of
   lines is either a doc comment re-deriving the spec or a spec embedding the
   code — both are findings above, and this says which specs to open first.
3. **Spec shape.** Per `specs/core/` spec: does it have a ` ```rust ` interface
   block; how far through the document is it; does it carry a usage example in
   any language; how many lines sit inside code fences. This produces a
   **shortlist to read, not a verdict** — a spec with no ` ```rust ` block may be
   conveying its interface some other way, and the scan cannot tell. Confirm each
   hit against failure 2 before reporting it. Also grep for work-order residue:
   headings matching *prior state | implementation plan | phase N | step N |
   migration*, and imperative construction language in the body.
4. **Opening paragraphs.** Extract the first prose paragraph of every spec — the
   text between the title and the second heading — into one list and read it as a
   list. Failures are far more obvious in bulk than in place, and the whole
   corpus fits on a couple of screens. Flag mechanically first: a backticked
   identifier or `::` in the opening *sentence*, a link inside it, an opening
   character that is a formula or a symbol. Then read the rest for the failure no
   grep finds — precise, true, and never says what the thing is for.
5. **Coverage both ways.** Which specs are never cited from code, and which code
   surfaces cite no spec. Feeds the existing **Code without specs** section.

An `audit-hygiene` run may hand you file pairs from its own duplicate scan; treat
those as leads into check 2, already narrowed.

## Report format

Open with a **Sample** block, so a reader knows what this run did and did not
look at — a bounded report is only trustworthy if its bounds are stated:

```
**Sample:** N of M specs read against their code. Seed `<seed>`; pool was
<never-audited | full corpus> (P candidates). Included regardless of the draw:
<specs with mechanical hits, and why>. Corpus-wide mechanical checks below cover
all M.
```

Then the corpus-wide **Mechanical findings** section: the defaults diff, the
spec↔code duplicate-prose pairs, the shape scan, and the coverage counts. This is
tables and numbers, and it is the part that scales — every spec appears here even
though only N were read.

Then one section per **sampled** spec, in this shape:

```
### <spec path>
**Summary:** <2-3 sentence description of what the spec covers>
**Implementing code:** <file paths, ideally with key symbols>
**Inconsistencies:**
  - <concrete divergence between spec and code>
**Third copies:** <doc comments re-deriving this spec's argument, with line
  counts and which copy should shrink — or "none">
**Recommendation:** <update spec | update code | discuss> with one-line justification
**Unclear / incorrect / suspicious:** <anything that doesn't make sense, is ambiguous, or is wrong>
```

Then a **Code without specs** section. Lead with a one-line-per-surface table
(surface | user-facing? | spec) covering everything, then expand only the entries
worth arguing about:

```
### <code surface: path or CLI command>
**What it does:** <2-3 sentence description>
**Why it matters:** <user-facing | internal-but-load-bearing | small utility>
**Recommendation:** <write a spec at specs/... | add a note to existing spec X | acceptable as unspecced>
```

End the report with a **Top priorities** section listing the 3–5 most important fixes across both divergences and missing specs.

Keep the whole report under ~800 lines. If it is running longer, the fix is a
smaller sample or terser per-spec sections — not dropping the mechanical section,
which is the cheap part and the part that covers everything.

## Saving the report

Write the full report to `reports/<date>-spec-audit.md`, where `<date>` is
today's date in `YYYY-MM-DD` form (get it with `date +%F`). Create the
`reports/` directory if it doesn't exist. Each run is a dated snapshot — do not
overwrite a prior day's report. After saving, tell the user the path and give a
short summary of the Top priorities in the conversation; don't paste the whole
report back.

Because each run reads a different sample, a spec-audit report does **not**
supersede its predecessor the way a hygiene snapshot does — the older one still
holds the only reading of the specs this run skipped. Retire an old spec audit
only when its findings are resolved or its specs have since been re-read, per the
retirement rules in `AGENTS.md`. Keep the `### <spec path>` heading format
exactly: the sampler greps prior reports for it to know what has been covered.

## Guidelines

- Inconsistencies must be concrete — cite line numbers or symbol names, not vague "the spec is out of date."
- Rank a divergence by what a reader would *do* differently believing it. A wrong
  default in a parameter table outranks a paragraph of stale narrative, however
  much longer the paragraph is.
- Three copies of a design will not stay in agreement by intention. Where a fix is
  available that removes a copy or ties two together mechanically — a doc comment
  cut back to contract-plus-link, a spec table generated from or asserted against
  the code — prefer it to re-syncing the prose by hand.
- If a spec has no implementing code yet (forward-looking spec), say so explicitly rather than flagging it as missing.
- Never imply the sample was the corpus. Say "N of M read" wherever it could be
  misread, and never write that a spec is consistent because it was not sampled —
  the mechanical checks acquit a spec, the sample's silence does not.
- If code exists with no spec, note it under a **Code without specs** section at the end.
- Do not modify any specs or code during this skill — it is read-only analysis.
