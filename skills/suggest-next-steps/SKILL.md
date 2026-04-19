---
name: suggest-next-steps
description: Recommend the next things to work on — concrete implementation tasks pulled from unimplemented spec sections, plus new-feature design topics worth discussing. Use when the user asks "what should we do next" or wants forward-looking direction.
---

# Suggest next steps

Produce two lists: concrete next implementation tasks, and speculative new-feature ideas worth a design discussion.

## How to work

1. Read the specs under `specs/` and `docs/` to understand intended scope.
2. Read the code to understand what is actually built.
3. Identify gaps: parts of existing specs that aren't implemented, or are partially implemented.
4. Think about adjacent features that would be natural extensions — things that aren't in any spec yet but fit the project's direction.

Dispatch `Agent` subagents in parallel for the spec-reading phase if the coverage is wide.

## Output

### ~5 implementation tasks (from existing specs)

Each item:

```
**<task title>**
- Spec reference: <file path, section if applicable>
- Current state: <what exists | nothing yet>
- Scope: <what the task involves>
- Why now: <why this is worth doing next — unblocks X, low-hanging, high-value, etc.>
```

Rank tasks by a mix of value and readiness (cheap wins first, hard-but-high-value clearly flagged).

### 2–3 design-discussion topics (new features, not yet specced)

Each item:

```
**<topic title>**
- Motivation: <what problem this would solve>
- Sketch: <1–2 paragraph rough shape — not a spec, a starting point for discussion>
- Where it would live: <new spec file | addition to existing spec X>
- Open questions: <the hard parts the discussion would need to resolve>
```

## Guidelines

- Be concrete: "implement `sfm xform crop`" is useful; "improve transforms" is not.
- Don't propose tasks you haven't verified are actually unimplemented — grep the code.
- Design-discussion topics should be things you'd actually want to build, not filler.
- Do not modify any specs or code during this skill.
