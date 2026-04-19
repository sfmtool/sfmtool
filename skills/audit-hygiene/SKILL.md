---
name: audit-hygiene
description: Survey the codebase for organizational drift — oversized files, files that should be split or merged, misleading names, directory structures that hurt navigation. Use when the user asks to clean up, reorganize, or review codebase structure.
---

# Codebase hygiene audit

As a project grows, files get bigger, coherent designs stretch, and names drift away from contents. This skill surveys the whole codebase (Python and Rust) for those smells and produces a prioritized list of structural fixes.

## Scope

- Python under `src/sfmtool/` and `tests/`
- Rust crates under `crates/`
- Top-level layout (`scripts/`, `specs/`, `docs/`)

## What to look for

1. **Oversized files** — modules or source files that have grown past what their purpose justifies. Look at line count, but more importantly at whether the file holds multiple distinct concerns.
2. **Files that should be combined** — small files that fragment a single concern, or near-duplicates.
3. **Misleading names** — files whose name no longer describes what's inside (e.g., grew to cover a second topic, or the original concept was renamed but the file wasn't).
4. **Directory-level smells** — directories that are flat when they should be grouped, grouped when they should be flat, or where the grouping no longer matches how the code is actually used.
5. **Dead or near-dead code** — modules referenced only by tests, commented-out blocks, `_foo_old.py`-style leftovers.

## How to work

1. Get a size overview: file line counts per directory.
2. Sample the largest files and skim their structure — count distinct top-level concerns.
3. Dispatch `Agent` subagents in parallel over subtrees (e.g., one for `src/sfmtool/feature_match/`, one for `crates/sfmtool-core/`) to get focused assessments.
4. Consolidate findings, removing duplicates and ranking.

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

## Guidelines

- Be specific. "X is too big" is useless; "X is 900 lines covering matching AND geometric filtering AND serialization — split serialization into a sibling file" is useful.
- Cite line counts or symbol counts when flagging size.
- Don't flag a file just because it's long — long is fine if the file has a single coherent purpose.
- Do not modify any code during this skill — it is read-only analysis.
