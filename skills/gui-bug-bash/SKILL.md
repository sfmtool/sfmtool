---
name: gui-bug-bash
description: Find bugs in the SfM Explorer by driving a live viewer over its MCP endpoint on copies of real .sfmr files, with the emphasis on recently merged GUI work, and write a dated report of verified findings. Use when the user asks for a bug bash, exploratory testing or a bug hunt of the viewer, or asks to exercise recent GUI features on real data.
---

# GUI bug bash

Start an SfM Explorer on copies of real reconstructions, drive it through its
MCP endpoint the way a person and an agent would, and report the bugs found, each
with steps to reproduce, the evidence, and the spec text or code it contradicts.
The output is `reports/YYYY-MM-DD-gui-bugbash.md`. A run aims for **10–20
verified findings**; stop sooner if the chosen scope is covered.

What makes this work, from the first run (2026-09-29):

- **Real data, not the fixtures.** Most bugs appeared where a recent feature
  met a reconstruction property its tests did not have: `.sift`-file features
  instead of embedded patches, a scene that is mostly points at infinity, a
  fisheye rig, an image deleted under a bench track.
- **A verification gate before anything is written down.** About a third of the
  first run's candidates were specified behaviour. Each candidate is checked
  against `specs/gui/` and the tool's own description before it is reported.

## Tools in this directory

Run them with `pixi run python skills/gui-bug-bash/<script>` from the repo root.
They use only the standard library.

| script | what it does |
|---|---|
| `viewer.py launch` | Picks a **random free port**, copies each `--copy` source into a `bugbash-<date>-<port>/` folder beside it, copies the viewer binary into a session directory, starts it detached with `--mcp <port> --no-default-layout`, waits for the endpoint, prints the session (port, pid, copies, labels) and saves it as `session.json` |
| `viewer.py status` / `stop` | Read or end the session a `--session <dir>` names |
| `mcp.py` | Call one tool: `mcp.py --port P TOOL '{json}'`. Prints text whole as UTF-8, writes images to `--out-dir` as `--name`.png, `--print EXPR` evaluates a Python expression over the parsed reply `d`. `mcp.py --port P wait` polls until the background task ends. `mcp.py --port P tools [--schema NAME]` lists tools. Exit status 2 is a refusal, 3 is no endpoint |
| `mcp.Client` | The same from Python, for loops: `Client(port).call(tool, args)` returns the parsed reply or raises `McpError` |
| `probes/schema_lint.py` | Checks `tools/list` for schemas that contradict their descriptions (alternatives marked required, required names that are not properties) |
| `probes/sample.py` | Calls one read tool over a range or a random sample of indexes into a JSON-lines file, and groups the errors |

**Always pass `--port`.** Shell state does not carry between commands, and the
default (8787) is the port a person's own viewer uses.

The endpoint is plain JSON-RPC over HTTP, so none of this needs the
`sfm-explorer` MCP server to be registered in the session. If it is, its tools
talk to port 8787 only, which is a different viewer from the one this skill
starts.

## 1. Setup

1. **Find the data.** Ask where the user's datasets are, or use the location
   they named, and check that it exists before relying on it: a named drive
   or directory may not be on this machine. A recursive `find` over a large
   collection can run for minutes, so run it in the background, bound it with
   `-maxdepth`, or list the `sfmr/` directories you need.
2. **Build.** `pixi run cargo build --release -p sfm-explorer`. The launcher
   runs a copy of the binary, so a viewer that is running does not block the
   build, and several sessions can run side by side.
3. **Launch** one viewer with the first files of the dataset matrix (section 3):

   ```bash
   pixi run python skills/gui-bug-bash/viewer.py launch \
     --session <scratchpad>/bugbash-<name> \
     --copy <datasets>/Foo/sfmr/solve.sfmr=foo \
     --copy <datasets>/Bar/sfmr/pan.sfmr=pan
   ```

   `SRC=NAME` sets the copy's stem, which is the node's label in the viewer.
   More files can be opened later with `open_reconstruction` on further copies.
   Make those with the same folder name (`bugbash-<date>-<port>/`) so the
   report can list every copy in one place.
4. **Save the tool list** for the report and the probes:
   `mcp.py --port P tools` and `probes/schema_lint.py --port P`. Schema lint
   findings go through the gate like any other.

On Linux the viewer needs a display and a Vulkan driver
(`scripts/display_env.sh`; see AGENTS.md).

## 2. Data safety

- **Never open an original for editing.** `viewer.py launch --copy` makes the
  copy; `--open` opens a path as it is and is meant for the corrupt-file probes
  in the scratchpad.
- The copy sits in a folder **beside** the original, inside its workspace,
  because a `.sfmr` resolves its images through its workspace. A copy in the
  scratchpad is a separate test case: the orphaned file.
- **Save only with an explicit `path`**, into the same `bugbash-<date>-<port>/`
  folder. `save_reconstruction` with no path writes over the file the node came
  from. Save-as relabels the node to the new file's stem (specified), so read
  `get_scene` after a save before naming the node again.
- Index files (`build_index_files`) are written beside the copy, which is
  where they belong.
- List every folder of copies at the top of the report. Do not delete them
  unasked: they are the evidence a reviewer reproduces from.

## 3. Choosing the scope

Two pools, as in `audit-specs`:

1. **Recent work.** `git log --since=<date of the last gui-bugbash report> --
   crates/sfm-explorer specs/gui` (or the last week if there is none). Read the
   titles, and for each PR note the tools, panels and gestures it touched.
   These get the most attention.
2. **Least recently exercised.** Every earlier `reports/*-gui-bugbash.md` ends
   with a "Checked and found working" list and a bug table. Compare them with
   `mcp.py tools` and give some of the budget to tools and panels no report
   mentions. Prior coverage is read from the reports, so there is no state
   file.

Write the chosen areas down before starting; the report's opening paragraph
names them.

### The dataset matrix

Choose files by property, not by name, and cover as many rows as the scope
touches:

| property | why it finds bugs |
|---|---|
| features in `.sift` files (`feature_source: "sift_files"`) | no keypoint pixels in the `.sfmr`, so any code reading `keypoint_xy` gets nothing |
| embedded patches | the bench, fits and patch tools need them |
| the same file after `convert_to_embedded_patches` | state made before the conversion meets data made after it |
| a fisheye rig | incidence angles past 90°, two cameras, rig frames |
| mostly points at infinity (a rotation pan) | directions stored where positions usually are |
| a large reconstruction (tens of thousands of points, 80+ images) | timing, sampling, dense Track View tables |
| a copy outside its workspace | image and index resolution |
| truncated, non-ZIP and checksum-corrupt files, and a missing path | load errors |

`get_scene` reports `feature_source`, `has_patch_data` and the counts
(including `points_at_infinity`) for each node.

## 4. Techniques

Each of these found at least one real bug in the first run.

- **Interleave state changes.** Put something on the bench, select it, or set
  a view. Then make a structural edit: delete a camera image, convert to
  embedded patches, bake a transform, switch a camera model, undo past it.
  Then read the first thing back and use it (fit, commit, screenshot). Index
  renumbering and stale snapshots show up here.
- **Follow every refusal's suggested remedy**, and check that the remedy is
  accepted and actually helps.
- **Compare a reply with the read that follows it.** A write's reply states
  what it did; `get_bench_track`, `get_point` or `get_scene` says what
  happened. For example, a clamped tilt reported the requested normal.
- **Compare two tools that report the same quantity**: `get_point` against
  `get_bench_track` pixels, `get_camera_intrinsics` against a refusal's angle,
  a version label against the reply's counts.
- **Edge inputs**: 0, a negative number, 1e-300, 1e300, 1e308, a zero vector,
  empty and duplicate lists, every observation of a track, a label with a
  newline or NUL, a path that does not exist.
- **Screenshot every panel a feature draws in**, and read the text in the
  image: headers, button captions, numbers, units. `screenshot` with
  `panel_name` (use `show_panel` first if it is behind a tab) and
  `max_dimension` around 1000.
- **Drive the real gestures.** `get_widgets` gives ids and rectangles, and
  `click` (with `count: 2`, `mouse_button: "right"`, `modifiers`), `hover`,
  `press_key` and `type_text` reach the same code paths as a person. The
  context menus of Image Detail and the Scene tree are listed in `click`'s
  reply as soon as they open.
- **Change the layout.** `get_window_layout`, edit the document, and
  `set_window_layout`: put two panels in one dock node, or split ones that
  share a node. The stock layout (from `--no-default-layout`) is only one
  arrangement.
- **Concurrency**: start a background operation (`bundle_adjust`,
  `build_index_files`, `find_nearby_tracks`), then close the node, edit it,
  undo, or `cancel_background_task`, and check what is left on disk and on
  the bench.
- **Sample instead of guessing.** For "how often" questions (patch sizes,
  missing placements, error spread) use `probes/sample.py` over a few hundred
  indexes and look at the distribution.

## 5. Harness pitfalls

Each of these cost the first run a wrong conclusion or a repeated step.

- **Bare indexes follow the selection.** `get_point {point: 12}` reads the
  *selected* reconstruction, and opening a file selects it. Use
  `pt3d_<hash>_<index>` ids or `select_reconstruction` first. A run of "no live
  point" errors from `sample.py` usually means this.
- **Long operations return `{"running": true}`.** Run `mcp.py wait` before
  reading the result. `get_background_task`'s `text` is the *last* task's,
  which may be an older one when the new call was refused or ran
  synchronously.
- **Version serials are global across nodes**, so a node's history can jump
  from v13 to v18. An edit after an undo discards the versions ahead of the
  cursor. Read `get_history` rather than remembering serials.
- **Bench steps do not make a node dirty**; only document edits do (specified
  in `specs/gui/bench.md`).
- **`get_action_log` returns the oldest entries first.** Pass
  `since_revision` and `actors: ["user"]` to see what the clicks did.
- A panel behind another tab refuses a screenshot and says to `show_panel`. A
  3D view change made while the 3D viewer is hidden is applied when it is next
  drawn, so `get_scene` may still show the old view.
- Staging a point that is already on the bench activates the existing item.
  A double-click on a point in the 3D view or in Image Detail stages it.
- Opening a file ends a solo (specified).

## 6. The verification gate

A candidate becomes a finding only when all of these hold:

1. **It reproduces** from a fresh state with the steps you will write down.
2. **It is not specified behaviour.** Search the tool's description
   (`mcp.py tools --schema NAME`) and `specs/gui/` (`grep -rn -i` for the
   gesture, tool or word) for the behaviour. If a spec says it is intended,
   list it under "Ruled out as by design" instead. Ruled out in the first run:
   the Points size slider reading 0.0 (it is log2), point ids that do not
   match the index after a commit (stable ids), save-as relabelling the node,
   the FOV kept after camera view, Find Nearby skipping points at infinity.
3. **The evidence is in hand**: the calls, the reply values, and what a
   following read or screenshot showed. Say what was observed, and next to it
   what was expected and where that expectation comes from.
4. **Where it is quick, the cause is located**: a file and line, or the
   function whose behaviour explains it. Not required, but a finding with a
   code pointer is much quicker to act on.

Rank by severity: **high** loses or corrupts data, **medium** blocks a feature
or gives a wrong answer, **low** is a wrong message, display or schema detail.

## 7. The report

`reports/YYYY-MM-DD-gui-bugbash.md`, in plain language (AGENTS.md "Writing
style"). The 2026-09-29 report is the model; it was retired once its findings
were fixed, so read it from history with `git show
cb88b98a:reports/2026-09-29-gui-bugbash.md`. Sections:

1. **Opening paragraph**: the build (`git rev-parse --short HEAD`), how the
   viewer was driven, and the scope with the PR numbers.
2. **Files**: a table of each node label, the original it copies (as a path
   relative to the local dataset collection, never an absolute path on this
   machine) and its properties. Then the `bugbash-*` folder name used.
3. **Summary table**: number, severity, area, one-line bug, ordered by
   severity.
4. **One section per bug**: what goes wrong, numbered repro steps as MCP calls
   with the values seen, the expected behaviour and its source, and the code
   location if found. Keep to what was observed.
5. **Ruled out as by design**: one line each, with the spec that settles it.
6. **Checked and found working**: the tools, gestures and data combinations
   exercised without a finding. The next run reads this to choose its scope.

Status annotations and retirement follow AGENTS.md "Quality reports".

## 8. Parallel runs

A viewer's selection, bench and history are shared by everyone calling it, so
**two agents must not drive one viewer**. For a parallel run, each agent
launches its own viewer (a random port each, and a `bugbash-<date>-<port>/`
folder each, so copies never collide) with its own share of the scope or the
dataset matrix. Each writes its findings into its own section or file, and one
agent merges them into the report, removing duplicates and running the
verification gate once more over the merged list. Launch the viewers one
after another, not all at once, so each launch's port check sees the ports
the earlier ones took.

## 9. Cleanup

- `viewer.py stop --session <dir>` for each viewer started, unless the user
  wants to look at one. If it is left running, say so, with its port.
- Leave the copies in place and list them in the report (section 2).
- Do not commit the copies or the session directories. The report and any
  skill or test changes are the only files for version control.
