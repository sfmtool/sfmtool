---
name: fix-report-findings
description: Work through findings in reports/ one at a time — for each, a fixer agent decides whether to fix it and does, an independent auditor checks the change, and the loop repeats until only simple issues remain; each finding lands on its own pushed branch. Use when the user asks for a fix round, to fix a batch of report findings, or to burn down an audit report.
---

# Fix a round of report findings

Take findings from the dated reports in `reports/` and fix them one per
branch. Each finding goes through a fixer agent, which evaluates it and either
fixes or declines it, and then an independent auditor agent, which verifies the
change against the code. The loop repeats until the auditor finds nothing that
needs rework. The coordinating session (you) picks findings, launches the
agents, and pushes the branches; it does not write the fixes itself.

Run **one agent at a time**. Agents share the main checkout, build Rust, run
`maturin develop`, and edit the same report file; running two at once makes
their builds and edits collide. Launch every agent without a model override,
so it runs on the session's model.

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

Branches are created from the upstream repository's `main` (or the fork's,
when upstream cannot be fetched) and pushed to the user's fork, never to
upstream. The remote names differ between checkouts, so
identify both from their URLs before anything else:

```bash
git remote -v
gh api user --jq .login   # the user's GitHub login, if gh is signed in
```

- **Upstream** is the remote whose URL points at the `sfmtool/sfmtool`
  repository on GitHub, in either form: `git@github.com:sfmtool/sfmtool.git`
  or `https://github.com/sfmtool/sfmtool(.git)`. It is often named `origin`
  or `upstream`.
- **Fork** is the remote whose URL points at a GitHub repository owned by
  the user, usually `<login>/sfmtool`. It is often named `fork`, but it can be
  named `origin`, the user's login, or anything else. When `gh` gives a login,
  the fork is the remote whose URL owner matches it. Without a login, a single
  GitHub remote other than upstream is the fork.

A common layout is a clone of the fork with no upstream remote at all, so
`origin` is the fork. When no remote points at `sfmtool/sfmtool`, add one
for fetching, and tell the user you added it. Use the URL form the fork
remote uses (`git@github.com:sfmtool/sfmtool.git` when the fork uses SSH):

```bash
git remote add upstream https://github.com/sfmtool/sfmtool.git
git fetch upstream main
```

If a remote named `upstream` already exists and points somewhere else, pick
another name such as `sfmtool-upstream` rather than changing it.

If the upstream cannot be fetched (no network access to it, no permission,
or the user declines adding the remote), do not stop. Use the fork's `main`
as the base instead, and tell the user before starting the first finding:
the branches are cut from `<fork>/main`, so if the fork is behind upstream
they will be missing upstream's newer commits and may conflict with them.

If the fork is missing or ambiguous (two candidates, or none), stop and ask
the user which remote to push to. Do not assume `origin` is the fork just
because it exists, and never push to the upstream remote.

Record in the ledger the fork's name as `<fork>`, and the remote whose
`main` is the base as `<base>`: the upstream remote normally, the fork when
the upstream could not be fetched. Every command below uses these names, and
every agent brief must be given the base remote's name so that `<base>/main`
refers to the right ref.

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
them; the fixer would mark them `Needs decision` anyway.

Write the queue and a ledger to the scratchpad. The ledger gets one line per
branch: its name, the outcome, and anything worth telling the user that falls
outside the finding.

## 2. For each finding

Create the branch from the current default branch:

```bash
git fetch <base> main
git checkout -b <prefix>-##-<descriptive> <base>/main
```

### 2a. Fixer

Launch one agent with the [fixer brief](#fixer-brief) below, the branch name,
the base remote's name (in place of `<base>` in the brief), and the
finding's location (report, section heading, and which items are still
open). Wait for it to finish before doing anything else.

The fixer returns FIX or DECLINE. A decline is a valid outcome, and it is
committed like a fix: the fixer annotates the finding with the reason, and
when the finding was wrong it also clarifies the text that misled the report
(see the brief). Audit and push a decline branch the same way as a fix
branch, so the reason reaches the report on the default branch and later
rounds do not pick the finding again.

### 2b. Auditor

Launch a fresh agent with the [auditor brief](#auditor-brief) below, the branch
name, the base remote's name and the same finding location. Tell it what
the fixer said it changed and what it deliberately left out, and name the
claims most worth checking (a new code path, a moved section, numbers,
anything the fixer says it inferred). For a decline, tell it the fixer's
decision and stated reason, so it can check that the finding really is wrong,
or really needs a maintainer. Give it no other part of the fixer's reasoning;
it should verify, not agree.

The auditor returns CLEAN or NEEDS-FIX.

- **NEEDS-FIX**: send the auditor's list back to a fixer (resume the original
  fixer agent if it is still available, otherwise a fresh one with the fixer
  brief's "later rounds" instruction), then audit again.
- **CLEAN, but the auditor changed more than its brief allows** (its
  "Audit fix:" commits changed how code works, rewrote a section, or restored
  a removed fact in several places, instead of SIMPLE sentence-level fixes):
  run one more audit round on those commits before accepting.
- **CLEAN** otherwise: go to 2c.

### 2c. Push

```bash
test -z "$(git status --porcelain)"   # the agents commit their own work
git push -u <fork> <prefix>-##-<descriptive>
```

Add the ledger line, then start the next finding from `<base>/main`.

Between agents, check that no build or test is still running. On Linux and
macOS use `ps aux | grep -E 'cargo|rustc|pytest|maturin'`. On Windows,
`ps aux` in Git Bash lists only MSYS processes and misses `cargo.exe` and the
rest, so use `tasklist | grep -iE 'cargo|rustc|python|maturin'` instead
(`python` because pytest runs as `python.exe`). An agent that left a
background command can report back late, after the next agent has started;
when that happens, check that the earlier branch on the remote is unchanged
and the current checkout holds only the current agent's work.

## 3. Finish

Switch back to the session's own branch. Report to the user:

- which remote's `main` the branches were cut from, repeating the warning
  when it was the fork's because upstream could not be fetched;
- the branches pushed, each with a one-line summary and its audit outcome
  (clean, number of simple audit fixes, second audit round);
- any finding declined or marked `Needs decision`, why, and what the fixer
  clarified for each finding that was wrong;
- the behaviour changes, separately from the spec and doc edits;
- the out-of-scope issues the agents noticed (the ledger notes);
- which branches will conflict when merged. Almost all of them add a status
  line to the same report, and some touch the same spec.

Do not open pull requests unless the user asks.

## Practical notes

- **Disk.** A full round of Rust builds can fill the disk. If a link fails for
  lack of space, deleting `target/debug/incremental` frees the most for the
  least rebuild cost. Pass this to the agents.
- **Toolchain.** Use `pixi run cargo …`. A system `cargo` may be an older
  Rust and report clippy warnings the project's toolchain does not.
- **Status lines name the branch, not a commit.** The repository squash-merges,
  so a commit SHA in a status line stops pointing at anything after the merge.
  Write `> _Status (YYYY-MM-DD): **Done** — <what changed>, branch
  \`<branch>\`._`.
- **Status words** are listed in `AGENTS.md` under "Quality reports". The
  fixer uses `Done`, `Partially done`, `Declined` (the finding is wrong, or
  not worth its cost) and `Needs decision` (it waits on a maintainer's
  choice). Step 1 skips `Declined`, `Needs decision` and `Superseded`.

## Fixer brief

> You are working on ONE finding from `reports/` in this repository, on the
> branch already checked out. You are the only agent running, so building and
> testing are allowed. Read `AGENTS.md` first (writing style, spec rules,
> "Quality reports"); read `specs/GLOSSARY.md` before naming anything; read
> `specs/TEMPLATE.md` when changing a spec's opening or adding a spec.
>
> **Evaluate, then fix or decline.** Read the finding in full. Verify that it
> still holds against the current code and specs. Decide FIX or DECLINE.
> Decline when the finding is wrong, already fixed, needs a maintainer's design
> decision (a public API or wire rename, a format change, a large refactor), or
> would cost more than it is worth, and explain why.
>
> **If DECLINE:** always annotate the finding in its report, below it, and
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
>   "noticed outside the finding" instead of fixing it. Name the clarification
>   in the status line, for example `**Declined** — the spec is right that
>   <fact>; clarified <where>, branch \`<branch>\``.
>
> **If FIX:** make a minimal change for this finding only. A report's proposed
> sentence is a draft claim: verify every word against the code before using
> it. Before shrinking a comment that repeats a spec, move any fact found only
> in the comment into the spec. When moving or renaming a section, fix every
> link and `#anchor` that pointed at it, in specs, docs and code comments (the
> spec-link test does not check anchors). Add a test when code behaviour
> changes, and check every writer and checked-in file before making a reader
> stricter. Annotate the finding in its report in place, below it:
> `> _Status (<today>): **Done** — <what changed>, branch \`<branch>\`._` (or
> **Partially done**, naming what is still open). Commit with a plain-language
> message. Do not push.
>
> **Later rounds:** if you are given an auditor's findings, fix each one or
> explain precisely why it is wrong, commit, and change nothing else.
>
> **Checks** (all must pass): the spec-link test
> (`pixi run test -- tests/test_spec_links.py`) when specs or source citations
> change; `pixi run fmt && pixi run check` plus the relevant tests for Python;
> `pixi run cargo fmt --all`, the relevant `pixi run cargo test -p <crate>`,
> `pixi run cargo clippy -p <crate> --all-targets` (add
> `--features sfm-explorer/ui-tests` for `sfm-explorer`, or its `ui_basic`
> tests are skipped) and `pixi run doc` for Rust;
> `pixi run maturin develop --release` first when Rust changes and Python tests
> need it. Leave no background processes running.
>
> **Final answer:** DECISION (FIX or DECLINE, and for a decline which of the
> four kinds), a short rationale, what you clarified and what the related-area
> research checked (for a wrong finding), files
> changed, commits (`git log --oneline <base>/main..HEAD`), checks run and
> their results, and anything you noticed outside the finding.

## Auditor brief

> You are independently auditing a change, committed on the branch checked
> out, that addresses ONE finding from `reports/`. You did not write it. You
> are the only agent running, so building and testing are allowed. Read
> `AGENTS.md` first (writing style, spec rules, "Quality reports").
>
> 1. Read the finding as it stands on the default branch
>    (`git show <base>/main:<report>`), the diff
>    (`git diff <base>/main...HEAD`) and the commit messages.
> 2. If the fixer declined the finding, check the decline itself: that the
>    finding is really wrong, already fixed, or needs a maintainer, as the
>    status line claims. For a wrong finding, check that the clarification
>    addresses what misled the report and that the related specs and docs the
>    fixer checked now agree with the code. A decline you can show is
>    mistaken is SUBSTANTIVE.
> 3. Verify every factual claim in the change against the code yourself:
>    names, numbers, defaults, line references, behaviour, links and anchors.
>    For removed or moved text, compare it with the original and confirm no
>    fact that is still true was lost. Check that the change resolves the
>    finding, stays in scope, follows the writing style (plain, literal,
>    present tense), and that the report's status line is present and
>    accurate.
> 4. Run the checks that apply (the spec-link test; ruff and tests for Python;
>    `pixi run cargo` fmt, test, clippy and `pixi run doc` for Rust). Never
>    push.
> 5. Classify each issue:
>    - **SIMPLE**: a wrong count or number, a wording or style slip, a broken
>      link, a wrap, a small factual correction in a sentence, a missing
>      trivial test assertion. Fix these yourself and commit with a message
>      starting "Audit fix:".
>    - **SUBSTANTIVE**: the change is wrong in a way that needs rework (a claim
>      that misdescribes behaviour across a section, a code bug, a missing test
>      for changed behaviour, scope creep, a fix that does not resolve the
>      finding). Do not fix these; describe each precisely, with evidence.
>
> Leave no background processes running. **Final answer:** VERDICT = CLEAN (no
> issues, or only SIMPLE ones that you fixed) or NEEDS-FIX (one or more
> SUBSTANTIVE issues, listed). Then the issues, what you fixed, and the check
> results. Be concise.
