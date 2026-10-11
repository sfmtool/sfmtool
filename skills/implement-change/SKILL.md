---
name: implement-change
description: Implement one change on a branch — an implementer agent evaluates the change and makes it, an independent auditor checks it against the code, and the loop repeats until only simple issues remain. Use whenever you are about to implement a change in this repository (a bug fix, a feature or one step of one, a spec or doc correction, a refactor), and when another skill hands you a change to implement.
---

# Implement a change

Make one change on one branch. An implementer agent evaluates the change and
either makes it or declines it, and then an independent auditor agent verifies
the result against the code. The loop repeats until the auditor finds nothing
that needs rework. The coordinating session (you) sets up the branch, launches
the agents, and pushes when asked; it does not write the change itself.

The implementer and auditor agents launched by this skill do their work
directly. They do not invoke this skill again.

Run **one agent at a time**. Agents share the main checkout, build Rust, run
`maturin develop`, and edit the same files; running two at once makes their
builds and edits collide. Launch every agent without a model override, so it
runs on the session's model.

## Inputs

Take these from the user's request, or from the skill that called this one:

- **The change.** What to change and why, where it applies (files, specs,
  report findings, an issue), and what counts as done. A change that is too
  large for one agent to finish and check in one pass is split into steps
  first, and each step goes through this skill on its own.
- **The branch.** Either the branch already checked out, or a new branch to
  cut from `<base>/main`. When the checkout is on `main`, always cut a new
  branch. Pick a short descriptive name that no local branch and no branch on
  the upstream or fork remote uses.
- **Whether to push.** Push only when the user or the calling skill asks.
- **Task-specific instructions** (optional). Extra rules from the calling
  skill for the implementer and the auditor, such as how to record the outcome
  in a report. Pass them into both briefs in place of the
  `<task-specific instructions>` placeholder, or delete the placeholder when
  there are none.

## 1. Find the upstream and fork remotes

Do this when a new branch is cut or the branch is pushed. Skip it when the
calling skill already found the remotes and gave you `<base>` and `<fork>`.

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
as the base instead, and tell the user before launching the first agent:
the branch is cut from `<fork>/main`, so if the fork is behind upstream
it will be missing upstream's newer commits and may conflict with them.

If the fork is missing or ambiguous (two candidates, or none) and the branch
is to be pushed, stop and ask the user which remote to push to. Do not assume
`origin` is the fork just because it exists, and never push to the upstream
remote.

Call the fork's name `<fork>`, and the remote whose `main` is the base
`<base>`: the upstream remote normally, the fork when the upstream could not
be fetched. Every command below uses these names, and every agent brief must
be given the base remote's name so that `<base>/main` refers to the right ref.
When working on an existing branch that was not cut from `<base>/main`, give
the agents the commit the branch's own work starts from instead, and use it in
place of `<base>/main` in the briefs.

## 2. Set up the branch

For a new branch:

```bash
git fetch <base> main
git checkout -b <branch> <base>/main
```

For the current branch, check that the working tree is clean
(`git status --porcelain` prints nothing). If it is not, ask the user what to
do with the uncommitted changes rather than letting an agent commit them.

## 3. Implementer

Launch one agent with the [implementer brief](#implementer-brief) below, the
branch name, the base (in place of `<base>` in the brief), the change, and any
task-specific instructions. Wait for it to finish before doing anything else.

The implementer returns DONE or DECLINE. A decline is a valid outcome: the
change was wrong, already made, needs a maintainer's decision, or would cost
more than it is worth. When the task-specific instructions say how to record a
decline (for example a status line in a report), the decline is committed and
goes through the audit and push like any other change. Otherwise the
implementer commits nothing for a decline; skip the audit and report the
reason to the user.

## 4. Auditor

Launch a fresh agent with the [auditor brief](#auditor-brief) below, the
branch name, the base, the change and the same task-specific instructions.
Tell it what the implementer said it changed and what it deliberately left
out, and name the claims most worth checking (a new code path, a moved
section, numbers, anything the implementer says it inferred). For a decline,
tell it the implementer's decision and stated reason, so it can check that the
change really is wrong, or really needs a maintainer. Give it no other part of
the implementer's reasoning; it should verify, not agree.

The auditor returns CLEAN or NEEDS-FIX.

- **NEEDS-FIX**: send the auditor's list back to an implementer (resume the
  original implementer agent if it is still available, otherwise a fresh one
  with the implementer brief's "later rounds" instruction), then audit again.
- **CLEAN, but the auditor changed more than its brief allows** (its
  "Audit fix:" commits changed how code works, rewrote a section, or restored
  a removed fact in several places, instead of SIMPLE sentence-level fixes):
  run one more audit round on those commits before accepting.
- **CLEAN** otherwise: go to step 5.

Between agents, check that no build or test is still running. On Linux and
macOS use `ps aux | grep -E 'cargo|rustc|pytest|maturin'`. On Windows,
`ps aux` in Git Bash lists only MSYS processes and misses `cargo.exe` and the
rest, so use `tasklist | grep -iE 'cargo|rustc|python|maturin'` instead
(`python` because pytest runs as `python.exe`). An agent that left a
background command can report back late, after the next agent has started;
when that happens, check that the checkout holds only the current agent's
work, and that any branch already pushed is unchanged on the remote.

## 5. Finish

```bash
test -z "$(git status --porcelain)"   # the agents commit their own work
git push -u <fork> <branch>           # only when asked to push
```

Report to the user, or return to the calling skill:

- the branch, and which remote's `main` it was cut from, repeating the warning
  when it was the fork's because upstream could not be fetched;
- the outcome (done or declined, and why), with the audit outcome (clean,
  number of simple audit fixes, a second audit round);
- the behaviour changes, separately from the spec and doc edits;
- the checks run and their results;
- anything the agents noticed outside the change.

Do not open a pull request unless the user asks. When asked, write the body
from `.github/PULL_REQUEST_TEMPLATE.md` as `AGENTS.md` § "Opening a pull
request" describes.

## Practical notes

- **Disk.** Several rounds of Rust builds can fill the disk. If a link fails
  for lack of space, deleting `target/debug/incremental` frees the most for
  the least rebuild cost. Pass this to the agents.
- **Toolchain.** Use `pixi run cargo …`. A system `cargo` may be an older
  Rust and report clippy warnings the project's toolchain does not.
- **Sign-off.** Every commit carries a DCO `Signed-off-by:` trailer
  (`git commit -s`); see `CONTRIBUTING.md`. Both briefs say so.

## Implementer brief

> You are implementing ONE change in this repository, on the branch already
> checked out. You are the only agent running, so building and testing are
> allowed. Do not invoke the `implement-change` skill; you are the
> implementer it launched. Read `AGENTS.md` first (writing style, spec rules,
> library interface contracts, "Quality reports"); read `specs/GLOSSARY.md`
> before naming anything; read the specs for the area you change; read
> `specs/TEMPLATE.md` when changing a spec's opening or adding a spec.
>
> **The change:** <the change>
>
> <task-specific instructions>
>
> **Evaluate, then implement or decline.** Check the change against the
> current code and specs: that the problem it addresses exists, and that the
> approach fits the code around it. Decide DONE or DECLINE. Decline when the
> change is wrong, already made, needs a maintainer's design decision (a
> public API or wire rename, a format change, a large refactor nobody asked
> for), or would cost more than it is worth, and explain why. Record a decline
> as the task-specific instructions say; when they say nothing, commit nothing
> and explain.
>
> **If implementing:** make a minimal change, for this change only. A proposed
> sentence or code snippet you were given is a draft: verify every word
> against the code before using it. Update the spec for the area when
> behaviour diverges from it. Before shrinking a comment that repeats a spec,
> move any fact found only in the comment into the spec. When moving or
> renaming a section, fix every link and `#anchor` that pointed at it, in
> specs, docs and code comments (the spec-link test does not check anchors).
> Add a test when code behaviour changes, and check every writer and
> checked-in file before making a reader stricter. Commit with a
> plain-language message, signed off (`git commit -s`), with no attribution
> lines. Do not push.
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
> need it. If a link fails for lack of disk space, delete
> `target/debug/incremental`. Leave no background processes running.
>
> **Final answer:** DECISION (DONE or DECLINE, and for a decline which kind),
> a short rationale, files changed, commits
> (`git log --oneline <base>/main..HEAD`), checks run and their results, and
> anything you noticed outside the change.

## Auditor brief

> You are independently auditing a change committed on the branch checked
> out. You did not write it. You are the only agent running, so building and
> testing are allowed. Do not invoke the `implement-change` skill; you are the
> auditor it launched. Read `AGENTS.md` first (writing style, spec rules,
> library interface contracts, "Quality reports").
>
> **The change that was asked for:** <the change>
>
> <task-specific instructions>
>
> 1. Read the diff (`git diff <base>/main...HEAD`) and the commit messages,
>    and the files the change was about as they stand on `<base>/main`.
> 2. If the implementer declined the change, check the decline itself: that
>    the change really is wrong, already made, or needs a maintainer, as
>    claimed. A decline you can show is mistaken is SUBSTANTIVE.
> 3. Verify every factual claim in the change against the code yourself:
>    names, numbers, defaults, line references, behaviour, links and anchors.
>    For removed or moved text, compare it with the original and confirm no
>    fact that is still true was lost. Check that the change does what was
>    asked, stays in scope, has tests for changed behaviour, keeps the specs
>    in step with the code, and follows the writing style (plain, literal,
>    present tense).
> 4. Run the checks that apply (the spec-link test; ruff and tests for Python;
>    `pixi run cargo` fmt, test, clippy and `pixi run doc` for Rust). Never
>    push.
> 5. Classify each issue:
>    - **SIMPLE**: a wrong count or number, a wording or style slip, a broken
>      link, a wrap, a small factual correction in a sentence, a missing
>      trivial test assertion. Fix these yourself and commit, signed off
>      (`git commit -s`), with a message starting "Audit fix:".
>    - **SUBSTANTIVE**: the change is wrong in a way that needs rework (a claim
>      that misdescribes behaviour across a section, a code bug, a missing test
>      for changed behaviour, scope creep, a change that does not do what was
>      asked). Do not fix these; describe each precisely, with evidence.
>
> Leave no background processes running. **Final answer:** VERDICT = CLEAN (no
> issues, or only SIMPLE ones that you fixed) or NEEDS-FIX (one or more
> SUBSTANTIVE issues, listed). Then the issues, what you fixed, and the check
> results. Be concise.
