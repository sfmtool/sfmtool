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
out, whether it said the change is visible and what it said to screenshot,
and name the claims most worth checking (a new code path, a moved
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

Push only once the suites the change requires (the implementer brief's
**Checks** list) have passed in full on the branch's last commit. The auditor
of the last round runs them before it returns CLEAN. One failure is excused:
on Windows, the `ui_basic` real-mouse test
`a_real_right_click_opens_the_reconstruction_rows_context_menu` failing with
"could not aim at the viewer's own window", as it does on a locked desktop.
When that is the only failure and every other `ui_basic` test passed, it
does not block the push or the PR; name the test and the reason in your
report and in the PR's Testing section, and say that CI's `ui-test-windows`
job runs it.
Any other suite that did not run or failed blocks the push.

Report to the user, or return to the calling skill:

- the branch, and which remote's `main` it was cut from, repeating the warning
  when it was the fork's because upstream could not be fetched;
- the outcome (done or declined, and why), with the audit outcome (clean,
  number of simple audit fixes, a second audit round);
- the behaviour changes, separately from the spec and doc edits;
- whether the change is visible, as the implementer and auditor agreed, and
  the screenshot links once they are taken;
- the checks run and their results;
- anything the agents noticed outside the change.

Do not open a pull request unless the user asks. When asked, write the body
from `.github/PULL_REQUEST_TEMPLATE.md` as `AGENTS.md` § "Opening a pull
request" describes. For a visible change, the screenshots come first: launch
one more agent with the [screenshot brief](#screenshot-brief), after the
audit and on its own like the others, and open the pull request only once it
has returned the image links. A visible change is anything the viewer draws,
an image a CLI command writes, or a docs page, however small: a new menu entry
or a renamed label counts. The Screenshots section says `n/a` only when the
implementer and auditor agree nothing is visible, and then says what was
checked.

## Screenshots

How the screenshot agent, or anyone opening a PR for a visible change, takes
and hosts the pictures:

1. **Data.** Copy the dataset directory you need from `test-data/images/`
   (the images, the `.sfmr` and its `.sfm-workspace.json`) into the
   scratchpad, and work on that copy. `viewer.py launch --copy` puts its copy
   beside the source, so pointing it at `test-data` directly writes into
   `test-data`.
2. **After.** `pixi run cargo build --release -p sfm-explorer`, then
   `pixi run python skills/gui-bug-bash/viewer.py launch --session
   <scratchpad>/shots-after --copy <copy>.sfmr`. It picks a random free port
   and prints it. Drive the viewer with
   `pixi run python skills/gui-bug-bash/mcp.py --port P TOOL '{json}'`
   (`skills/gui-bug-bash/SKILL.md` lists the tools) to the state to show, and
   take each picture with `screenshot` and `--out-dir`/`--name`.
3. **Before**, when something that already existed changed. Make a worktree
   of `<base>/main` in the scratchpad (`git worktree add <scratchpad>/before
   <base>/main`) and build it with `pixi run cargo build --release -p
   sfm-explorer --manifest-path <scratchpad>/before/Cargo.toml`, which builds
   into the worktree's own `target/` and leaves the checkout's build alone.
   Launch the worktree's copy of the launcher
   (`pixi run python <scratchpad>/before/skills/gui-bug-bash/viewer.py
   launch …`), which runs the binary built beside it, and shoot the same state
   as the after picture. Stop that viewer and `git worktree remove` the
   worktree afterwards.
4. **Privacy.** Crop each picture to what changed: a panel (`panel_name`), and
   with `crop_px` the menu, row or widget inside it; for a menu that reaches
   outside its panel, crop the window instead. Look at every PNG before using
   it: no local path, username or personal data may show. The Background Tasks
   and Action Log panels show full paths in their "Open" rows, so crop those
   rows out or clear the panel first.

   For an image a CLI command writes, run the command on the scratchpad copy
   and use its output; for a docs page, screenshot the page from
   `pixi run docs-serve`. The same privacy check and hosting apply.
5. **Hosting.** Put the PNGs in one orphan commit "Screenshots for <branch>"
   on a `pr-<branch>-screenshots` branch of the fork, built without touching
   the checkout:

   ```bash
   tree=$(for f in <dir>/*.png; do
     printf '100644 blob %s\t%s\n' "$(git hash-object -w "$f")" "$(basename "$f")"
   done | git mktree)
   sha=$(git commit-tree "$tree" -m "Screenshots for <branch>" \
     -m "Signed-off-by: $(git config user.name) <$(git config user.email)>")
   git push <fork> "$sha:refs/heads/pr-<branch>-screenshots"
   ```

   Link each one as
   `https://raw.githubusercontent.com/<fork owner>/sfmtool/<sha>/<file>.png`.
   The Screenshots sections of PRs #1038 and #1041 are examples.
6. Stop every viewer started (`viewer.py stop --session <dir>`).

## Practical notes

- **Disk.** Several rounds of Rust builds can fill the disk. If a link fails
  for lack of space, deleting `target/debug/incremental` frees the most for
  the least rebuild cost. Pass this to the agents.
- **Toolchain.** Use `pixi run cargo …`. A system `cargo` may be an older
  Rust and report clippy warnings the project's toolchain does not.
- **Sign-off.** Every commit carries a DCO `Signed-off-by:` trailer
  (`git commit -s`); see `CONTRIBUTING.md`. The implementer and auditor
  briefs say so, and the screenshot commit in § "Screenshots" carries one.

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
> **Checks** (all must pass). Choose them by what the change touches, not by
> which test files you edited: some tests check every menu entry or every
> binding file, so they fail on a change that never touched them.
>
> - Specs or source citations: `pixi run test -- tests/test_spec_links.py`.
> - Python: `pixi run fmt && pixi run check` and the relevant tests.
> - Any Rust: `pixi run cargo fmt --all`,
>   `pixi run cargo clippy --workspace --all-targets --features
>   sfm-explorer/ui-tests -- -D warnings` (the feature, or clippy skips the
>   `ui_basic` tests), `pixi run cargo test --workspace` and `pixi run doc`.
> - `crates/sfmtool-py/`, or anything re-exported through it:
>   `pixi run maturin develop --release`, then the full `pixi run test`. The
>   layout and registration tests (`tests/test_module_layout.py`, the
>   `test_*_registration.py` modules) check invariants across files that a
>   subset misses.
> - `sfm-explorer` (anything the viewer draws, any MCP tool):
>   `pixi run -e ui ui-test` as well. On Windows its real-mouse test
>   `a_real_right_click_opens_the_reconstruction_rows_context_menu` fails with
>   "could not aim at the viewer's own window" when the desktop is locked or
>   another window covers the viewer. That failure alone does not block the
>   push, because CI's `ui-test-windows` job runs the test on a real desktop,
>   but every other `ui_basic` test must pass. Report it as not run, not as a
>   pass.
>
> Narrower runs are fine while you work, but the suites above must pass on
> your final commit. Report any suite you did not run in
> full, and why. If a link fails for lack of disk space, delete
> `target/debug/incremental`. Leave no background processes running.
>
> **Final answer:** DECISION (DONE or DECLINE, and for a decline which kind),
> a short rationale, VISIBLE (yes or no: does the change alter anything the
> viewer draws, such as a menu entry, label, tooltip, panel, layout, colour or
> 3D-view rendering, an image a CLI command writes, or a docs page; no change
> is too small to count; for yes, what to screenshot, in what state, and
> whether a before picture is needed; for no, what you checked), files
> changed, commits
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
>    present tense). Check the implementer's VISIBLE claim against the diff:
>    a change to anything the viewer draws, an image a CLI command writes, or
>    a docs page is visible however small, such as a new menu entry or a
>    renamed label.
> 4. Run the checks that the **Checks** list in the implementer brief
>    (`skills/implement-change/SKILL.md`) requires for what the diff touches,
>    in full, on the branch's last commit, after any "Audit fix:" commits of
>    your own. Choose them by what the diff touches, not by which test files
>    it edits: a full `pixi run test` after a `sfmtool-py` change,
>    `pixi run -e ui ui-test` after a `sfm-explorer` change, and
>    `cargo test` and `cargo clippy` with `--workspace` after any Rust change.
>    A failing suite is SUBSTANTIVE unless the fix is SIMPLE, with one
>    exception: on Windows, the real-mouse `ui_basic` test
>    `a_real_right_click_opens_the_reconstruction_rows_context_menu` failing
>    with "could not aim at the viewer's own window" (a locked desktop) is
>    not SUBSTANTIVE and does not stop you returning CLEAN, as long as every
>    other `ui_basic` test passed. Report it as not run, not as a pass; CI's
>    `ui-test-windows` job runs it. If you already have a SUBSTANTIVE
>    finding, you may skip the long suites, since the next round's auditor
>    runs them; say which you skipped. Never push.
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
> SUBSTANTIVE issues, listed). Then VISIBLE (yes or no, and what to
> screenshot, correcting the implementer's answer if it was wrong), the
> issues, what you fixed, and the check results. Be concise.

## Screenshot brief

> You are taking the screenshots for a visible change committed on the branch
> checked out, for its pull request. You are the only agent running, so
> building is allowed. Do not invoke the `implement-change` skill. Read
> `AGENTS.md` § "Opening a pull request" and `skills/implement-change/SKILL.md`
> § "Screenshots", and follow the latter.
>
> **The change:** <the change>
>
> **What to show:** <the VISIBLE answers of the implementer and auditor>
>
> The base is `<base>/main` and the fork remote is `<fork>`. Change nothing
> on the branch and leave the checkout as you found it.
>
> **Final answer:** for each picture, its link and a one-sentence caption
> saying what it shows and in what state, the dataset copied, and the
> `pr-<branch>-screenshots` commit. Leave no viewer or worktree behind.
