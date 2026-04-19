---
name: implement-random-idea
description: Pick one item at random from a set of candidate tasks and drive it end-to-end — proposal, spec, incremental implementation, with subagent review at each stage. Use when the user wants to pick something at random and just ship it.
---

# Implement a random idea end-to-end

Take a set of candidate tasks (typically the output of `audit-specs`, `suggest-next-steps`, or `audit-hygiene`), randomly pick one, and carry it through proposal → spec → implementation with quality gates at every stage.

## Inputs

If the current conversation already contains a set of candidate tasks (e.g., the user listed them, or a prior skill run produced them), use those.

Otherwise, run all three of `audit-specs`, `audit-hygiene`, and `suggest-next-steps` first — in parallel, via `Agent` subagents — and pool their outputs into a single candidate list covering spec/code inconsistencies, hygiene problems, and new-feature ideas. The random pick then draws from across all three sources, so any of them can be the starting point for the end-to-end implementation.

## Stage 0 — Random selection

Enumerate candidates as a numbered list, then pick one:

```bash
echo $(( RANDOM % N + 1 ))
```

where `N` is the candidate count. Announce the pick and briefly state why, from its description, it seems worth doing. Do not substitute your own preference for the RNG — the whole point is to avoid cherry-picking.

## Stage 1 — Proposal

Write a proposal covering:
- Problem statement
- Goals and non-goals
- Sketch of the approach
- Alternatives considered
- Open questions

Then dispatch an `Agent` subagent to cross-validate: hand it the proposal and ask for (a) gaps or risks, (b) anything that's hand-wavy, (c) whether the approach seems right. Incorporate feedback. Loop if needed — proposal quality is cheap to improve at this stage.

## Stage 2 — Spec

Turn the proposal into a real spec, in the style of existing files under `specs/`. Be precise: data shapes, CLI surface, error behavior, test expectations.

Dispatch a second subagent to review the spec against the code and existing specs: does it fit? Are there contradictions? Is anything underspecified? Revise.

Save the spec under `specs/` in the appropriate subdirectory.

## Stage 3 — Incremental implementation

Break the spec into small, individually testable steps. For each step:

1. Implement (in this main session, or dispatch a subagent for a well-scoped chunk).
2. Add or update tests.
3. Run the relevant format/lint/test commands per `CLAUDE.md`:
   - Python: `pixi run fmt && pixi run check && pixi run test`
   - Rust: `pixi run cargo fmt && pixi run cargo clippy --workspace && pixi run cargo test --workspace`
   - If Rust changed and Python uses it: `pixi run maturin develop --release`
4. Review the diff (or have a subagent review it): tests present and meaningful? Code readable? Matches spec? No dead code, no unjustified comments, no premature abstractions?

Between steps, re-check against the spec. If something doesn't fit, pause and decide: adjust the code, or adjust the spec (and record why). Do not paper over a spec mismatch.

## Stage 4 — Wrap

- Confirm all tests pass and lint/format are clean.
- Summarize what landed, what's left out, and what follow-ups are worth filing.
- Do NOT create a commit or PR unless the user asks for one.

## Guidelines

- **Keep the RNG honest.** If the picked task is truly unworkable (e.g., blocked by an unbuilt dependency), say so and re-roll explicitly — don't silently swap.
- **Subagents review, they don't decide.** You read their feedback and make the call.
- **Backtrack when the result isn't good.** If stage 3 reveals the spec was wrong, go back to stage 2. Better to redo than to ship something misshapen.
- **No scope creep.** Stick to the selected task. File follow-ups instead of bundling.
