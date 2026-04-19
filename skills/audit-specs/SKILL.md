---
name: audit-specs
description: Audit every spec in specs/ and docs/ against the corresponding code and produce a consistency report. Use when the user asks to check spec/code drift, verify specs are still accurate, or review the state of specifications.
---

# Audit specs against code

Walk every specification, design document, and user documentation under `specs/` and `docs/`, compare each to the code that implements it, and produce a single consolidated report.

## Scope

The audit is bidirectional:

- **Spec → code.** Every file under `specs/` (including `specs/cli/`, `specs/formats/`, `specs/gui/`, `specs/workspace/`) and under `docs/` — locate the implementing code and compare.
- **Code → spec.** Every significant code surface — CLI commands in `src/sfmtool/_commands/`, crates in `crates/`, modules in `src/sfmtool/`, user-facing file formats — check whether a spec or doc covers it. Unspecced code is a first-class finding, not a footnote.

## How to work

1. List every spec/doc file. Do not sample — cover all of them.
2. List every significant code surface (each CLI command, each crate, each major module, each file format).
3. For each spec, read it, then read the implementing code. For each code surface, identify whether any spec/doc covers it. For large surface areas, dispatch `Agent` subagents in parallel so the main context stays clean — each returns a structured summary.
4. Consolidate into a single report.

## Report format

One section per spec, in this shape:

```
### <spec path>
**Summary:** <2-3 sentence description of what the spec covers>
**Implementing code:** <file paths, ideally with key symbols>
**Inconsistencies:**
  - <concrete divergence between spec and code>
**Recommendation:** <update spec | update code | discuss> with one-line justification
**Unclear / incorrect / suspicious:** <anything that doesn't make sense, is ambiguous, or is wrong>
```

Then a **Code without specs** section — each entry:

```
### <code surface: path or CLI command>
**What it does:** <2-3 sentence description>
**Why it matters:** <user-facing | internal-but-load-bearing | small utility>
**Recommendation:** <write a spec at specs/... | add a note to existing spec X | acceptable as unspecced>
```

End the report with a **Top priorities** section listing the 3–5 most important fixes across both divergences and missing specs.

## Guidelines

- Inconsistencies must be concrete — cite line numbers or symbol names, not vague "the spec is out of date."
- If a spec has no implementing code yet (forward-looking spec), say so explicitly rather than flagging it as missing.
- If code exists with no spec, note it under a **Code without specs** section at the end.
- Do not modify any specs or code during this skill — it is read-only analysis.
