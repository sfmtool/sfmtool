# <Title — the thing, not the change>

<!--
The default shape of a `specs/core/` spec — a starting point, not a form to fill
in. `specs/cli/`, `specs/formats/` and `specs/gui/` differ; see "Other areas" at
the bottom.

**Depart from it when the subject is served better by something else, and most
subjects will want some of this.** Drop a section that does not apply — an empty
heading is worse than an absent one. Merge two that keep saying the same thing.
Reorder when the argument only lands in a different order: a spec whose whole
point is a convention may need the convention stated before anything else can be
named; one describing a format may have no theory to separate from its layout;
one about a numerical hazard may have to teach the hazard before an interface
means anything. Sub-headings inside a section are free-form — use them, and let
them carry the subject's own vocabulary rather than this file's.

The order here is a claim about what a reader usually needs first, not a filing
system. If you can say why your arrangement serves the reader better, it is the
right one, and it is worth a sentence near the top saying what the spec does
instead. What survives any rearrangement is small, because it is about the reader
rather than the subject:

- **The plain-English opening** (see Purpose). Every subject can be introduced to
  someone who has not read the other specs. No exceptions to this one. It is also
  a rule about position: **the opening paragraph is the first prose in the file**,
  under the title and nothing else. No status line, no dated errata block, no
  inventory of paths, no "see also" — those either belong further down or do not
  belong at all.
- **Present tense, describing what the code is.** Not a work order: no "new
  module X", no "declare `pub mod` in lib.rs", no "phase 2", no "prior state
  (before this change)". If a section only makes sense to someone who was in the
  room when the change was proposed, it does not belong in the standing spec.
  Write the proposal in `specs/drafts/` and convert it before filing.
- **A caller can find out what to call.** Whether that is a Rust API section, an
  entry table, or a set of flags depends on what the thing is — but it should not
  be buried under a derivation. **It is also where the code pointer lives** — see
  Rust API below.
- **Implementation notes carry what the code cannot.** Never a transcription of
  the body.

**Drafts and standing specs.** Location encodes lifecycle, so nothing in the
document has to.

- **A standing spec is anything under `specs/` outside `specs/drafts/`.** It
  describes what exists, in the present tense, and **carries no `**Status:**`
  line.** Being filed is the status.
- **`specs/drafts/` holds proposals.** A draft opens with `**Status:** Draft`
  followed by what is decided and what is not; it may use future tense and
  construction language freely, because it is addressed to the people deciding.
- **Filing a draft** means three edits: delete the Status line, move the purpose
  paragraph to the top, and convert the whole document to the present tense.
  Then move the file and add its row to the area's `README.md`.

**Partial implementation is expressed by an amendment draft, never by an inline
marker.** Do not annotate a standing spec with "not yet implemented", a phase
list, or a dated note. Say in the present tense what the code does not do, and
link the proposal that would change that:

  Non-goals: GPU evaluation. The kernels are CPU-only; a GPU path is proposed in
  [foo-gpu-amendment.md](../../drafts/foo-gpu-amendment.md).

The draft opens by naming and linking the spec it amends. When it ships, its
content folds into the standing spec and the draft is deleted.

**A non-goal is not a draft.** A non-goal is a fact — one present-tense sentence,
no draft, no link ("does not support rolling shutter; use X"). A draft is an
intention someone means to build. Do not raise a non-goal into an amendment draft
just to have somewhere to point.
-->

## Purpose

<!--
**The first paragraph is for someone who has read none of the other specs.**
Assume a competent engineer who knows roughly what Structure-from-Motion is, and
nothing else: not this codebase's vocabulary, not the sibling specs, not the
change this came from. In plain English, say what the thing is and what it is
for. They should finish it able to tell whether this is the spec they wanted.

That paragraph must stand alone. No bare symbol name in the opening sentence
(`WarpMap::from_patch` means nothing yet), no formula before the words, no term
this repo defined elsewhere used as if it were common, and no link the reader has
to follow to parse the sentence — link freely, but the sentence has to work
without the click. Don't open by describing a change ("the original
implementation ran…", "there is no facility to…"): describe what the thing is.

  Good — `analysis/cluster-census.md`: "A reconstruction can be internally
  consistent and wrong."
  Good — `analysis/image-pair-graph.md`: "Several pipelines need to know which
  image pairs see the same part of the scene."
  Not this — `analysis/source-clusters.md`: "A member observation and a selection
  row both carry the image index and the feature index." True, but it starts in
  the middle of a conversation the reader has not had, and never says what the
  thing is for.

After that opening, go as deep as the subject needs: the problem in full, why the
obvious alternative is not good enough, who the consumers are.
-->

## Rust API

<!--
**Near the top, and before the theory.** The public interface as a caller sees
it: the types, the signatures, the errors. Elide bodies — `;` on a function is
right, `{ /* … */ }` is noise.

**This section leads with the code pointer** — one sentence saying where the
thing lives and what it is bound as, e.g. "The kernels live in
[foo.rs](../../../crates/sfmtool-core/src/geometry/foo.rs), bound as
`sfmtool._sfmtool.geometry.foo`." Not a stamp at the top of the file: a reader
who has not decided to call anything does not need it yet. Write every repo path
as a relative Markdown link, so it is clickable on GitHub and in an editor and so
a moved file shows up as a broken link; binding names are code spans, not links,
because they are not files. A CLI or GUI spec does the same in the section that
names its command or module.

Then the part that earns the section: **why it is shaped this way.** Why these
are the arguments; why the output is one struct and not three returns; why the
error is an enum and not a `Result<_, String>`; what an alternative shape would
have cost. This is the reasoning a reader cannot recover from the signature, and
it is what stops the next person quietly redesigning it.

Then **example usage** — the shortest real call, with the imports, showing what
a caller actually holds before and after. One example beats three signatures.
-->

```rust
/// One-line contract.
pub struct FooParams { /* … */ }

pub fn foo(input: ArrayView2<'_, f32>, params: &FooParams) -> Result<Foo, FooError>;
```

## Theory

<!--
The design: the model, the maths, the invariants, the argument that it works.
Derivations, the failure modes it is built against, the empirical evidence
behind any tuned value. Everything a doc comment should *link to* rather than
repeat lives here — this section is the reason the code can stay terse.
-->

## Implementation notes

<!--
After the theory, and only what a reader cannot get by reading the code.

Good: an invariant two functions jointly maintain; why this loop order and what
breaks if it changes; a numerical hazard and the formula chosen to dodge it; the
cost model behind a cache; which crate primitive is the single source of truth
for something spelled here.

Not this: a transcription of the body, a walk through the control flow, a list of
the private helpers. If a sentence would need editing when the code is refactored
without changing behaviour, it is the wrong sentence.
-->

## Parameters

<!--
One table: name, default, meaning. The defaults here are checked mechanically
against the code by the `audit-specs` skill, so a wrong one is a real bug —
state where the value is actually defined.
-->

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `bar` | `0.65` | … |

## Python bindings

<!--
The `sfmtool-py` surface and the Python-visible names, when there is one. Keyword
names, shapes and dtypes of the numpy arrays crossing the boundary, and how
errors map. A short Python usage example.

Argument names should match the Rust ones unless there is a reason; say the
reason. Do not let a Rust-side suffix (`_py`, `_rs`) reach a Python name.
-->

## Testing

<!--
What must be true for this to be considered working: the properties, the
degenerate inputs, the determinism guarantees. Name the test module.
-->

## Non-goals

<!--
What this deliberately does not do, and the nearest thing that does. One
present-tense sentence each.

This is also where an unbuilt part goes: state it as a fact and link the
amendment draft that proposes it (see "Drafts and standing specs" above). A
non-goal nobody intends to build needs no draft and no link.
-->

## Open questions

<!--
Genuinely undecided, with enough context to decide later. Delete entries as they
are settled — a stale open question reads as a live one.
-->

---

## Other areas

- **`specs/cli/`** — a command's spec is organized around its flags and its
  behaviour, not a Rust API. Keep Purpose, Parameters (the flags), Testing and
  Non-goals; replace "Rust API" with the invocation and its output, and point at
  the module that implements it.
- **`specs/formats/`** — the on-disk layout is the interface. The equivalent of
  "Rust API / why it is shaped this way" is the entry table plus the versioning
  rules; keep Theory for the encoding decisions.
- **`specs/gui/`** — describe the panel's behaviour and state, and the user-facing
  contract. A Rust API section is only warranted where other modules call in.

Add a row to the area's `README.md` index when you file a new spec.
