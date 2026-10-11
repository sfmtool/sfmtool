---
name: fix-report-findings
description: Work through findings in reports/ one at a time — pick and order the open findings, then put each through the implement-change skill (an implementer agent decides whether to fix it and does, an independent auditor checks the change, repeating until only simple issues remain); each finding lands on its own pushed branch. Use when the user asks for a fix round, to fix a batch of report findings, or to burn down an audit report.
---

# Fix a round of report findings

Take findings from the dated reports in `reports/` and fix them one per
branch. This skill decides which findings to fix and in what order; each
finding is then implemented with the `implement-change` skill
(`skills/implement-change/SKILL.md`), whose implementer agent evaluates the
finding and either fixes or declines it, and whose independent auditor agent
verifies the change against the code. The coordinating session (you) picks
findings, runs `implement-change` for each, and keeps the ledger; it does not
write the fixes itself.

Work on **one finding at a time**, for the reason `implement-change` runs one
agent at a time: the agents share the main checkout, and most findings add a
status line to the same report.

## Inputs

Ask the user, or take from their request:

- **Branch prefix.** Branches are named `<prefix>-##-<descriptive>`, numbered
  from 01 in the order the findings are done (for example
  `audit-fix-07-sift-format-checks`). Use a prefix that no existing local
  branch, and no branch on the upstream or fork remote, uses.
- **How many findings**, and from which reports. Default to 10 findings, the
  first 10 in queue order (step 1), drawn from every report under `reports/`.
  The user can ask for more; 25 is a reasonable larger round. Each finding is
  one branch to review and merge, and most of them add a status line to the
  same report, so a larger round means more merge conflicts.

## 0. Find the upstream and fork remotes

Do this once for the round, as `implement-change` § "1. Find the upstream and
fork remotes" describes, and pass `<base>` and `<fork>` to every
`implement-change` run so it skips that step. If the upstream cannot be
fetched, tell the user before starting the first finding that every branch in
the round is cut from `<fork>/main`. Record both names in the ledger.

## 1. Build the queue

Read the reports and list every finding that has no status line, or is marked
`Partially done` or `Not done`. Skip findings marked `Superseded`,
`Declined` (an earlier round judged them wrong or not worth the cost, and the
status line says why) or `Needs decision` (waiting on a maintainer) unless the
user asks for them. For each, record where it is (report, section heading,
item) and one line on what is still open. Order the queue so the findings that
fix wrong statements or wrong code come first, then shape and wording, then
refactors. Leave out items that only a maintainer can decide (naming a public
API, a format change, moving a spec to another area) unless the user asks for
them; the implementer would mark them `Needs decision` anyway.

Write the queue and a ledger to the scratchpad. The ledger gets one line per
branch: its name, the outcome, and anything worth telling the user that falls
outside the finding.

## 2. For each finding

Run `implement-change` with:

- **the change**: the finding's location (report, section heading, and which
  items are still open), and the instruction to resolve it;
- **the branch**: a new branch `<prefix>-##-<descriptive>` cut from
  `<base>/main`;
- **push**: yes, to `<fork>`;
- **task-specific instructions**: the [implementer
  additions](#implementer-additions) and [auditor
  additions](#auditor-additions) below, with the branch name filled in.

With these additions a decline is committed like a fix: the implementer
annotates the finding with the reason, and when the finding was wrong it also
clarifies the text that misled the report. `implement-change` audits and
pushes a decline branch the same way as a fix branch, so the reason reaches
the report on the default branch and later rounds do not pick the finding
again.

When `implement-change` finishes, add the ledger line from its report, then
start the next finding from `<base>/main`.

## 3. Finish

Switch back to the session's own branch. Report to the user:

- which remote's `main` the branches were cut from, repeating the warning
  when it was the fork's because upstream could not be fetched;
- the branches pushed, each with a one-line summary and its audit outcome
  (clean, number of simple audit fixes, second audit round);
- any finding declined or marked `Needs decision`, why, and what the
  implementer clarified for each finding that was wrong;
- the behaviour changes, separately from the spec and doc edits;
- the out-of-scope issues the agents noticed (the ledger notes);
- which branches will conflict when merged. Almost all of them add a status
  line to the same report, and some touch the same spec.

Do not open pull requests unless the user asks.

## Practical notes

- **Status lines name the branch, not a commit.** The repository squash-merges,
  so a commit SHA in a status line stops pointing at anything after the merge.
  Write `> _Status (YYYY-MM-DD): **Done** — <what changed>, branch
  \`<branch>\`._`.
- **Status words** are listed in `AGENTS.md` under "Quality reports". The
  implementer uses `Done`, `Partially done`, `Declined` (the finding is wrong,
  or not worth its cost) and `Needs decision` (it waits on a maintainer's
  choice). Step 1 skips `Declined`, `Needs decision` and `Superseded`.

## Implementer additions

> The change is ONE finding from `reports/`. Read the finding in full and
> verify that it still holds against the current code and specs before
> deciding.
>
> **If you decline**, always annotate the finding in its report, below it, and
> commit:
>
> - Already fixed: `**Done** — already fixed by <what>, <PR or branch if
>   known>`.
> - Needs a maintainer: `**Needs decision** — <the choice to be made>`.
> - Not worth the cost: `**Declined** — <why>`.
> - Wrong: `**Declined** — <why it is wrong>`. A wrong finding usually means
>   the code, comment, spec or doc it points at was ambiguous or easy to
>   misread, since a careful reader got it wrong. Find what misled the report
>   and make a small change near the finding's target that removes the
>   ambiguity: a clearer sentence, a missing qualifier, a comment stating an
>   invariant the reader could not see, a link to where the fact is defined.
>   Then research the surrounding area: read the related specs, docs, doc
>   comments and code that describe the same behaviour, and check them against
>   the code for consistency and accuracy. Fix any small inaccuracy you find
>   there that comes from the same confusion; list anything larger under
>   "noticed outside the change" instead of fixing it. Name the clarification
>   in the status line, for example `**Declined** — the spec is right that
>   <fact>; clarified <where>, branch \`<branch>\``.
>
> **If you fix it**, a report's proposed sentence is a draft claim like any
> other. Annotate the finding in its report in place, below it:
> `> _Status (<today>): **Done** — <what changed>, branch \`<branch>\`._` (or
> **Partially done**, naming what is still open).
>
> In your final answer, say which of the four kinds a decline is, and for a
> wrong finding what you clarified and what the related-area research checked.

## Auditor additions

> The change addresses ONE finding from `reports/`. Read the finding as it
> stands on the default branch (`git show <base>/main:<report>`). For a
> decline of a wrong finding, check that the clarification addresses what
> misled the report and that the related specs and docs the implementer
> checked now agree with the code. Check that the change resolves the finding
> and that the report's status line is present and accurate.
