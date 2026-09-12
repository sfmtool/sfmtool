# The Background panel: an operation that outlives a frame

**Status:** Draft

Amends [gui/edits/bundle-adjust.md](../gui/edits/bundle-adjust.md) § "Non-goals",
[gui/document-model.md](../gui/document-model.md) § "Two kinds of edit",
[gui/panel-layout.md](../gui/panel-layout.md) § "Home positions",
[gui/action-log.md](../gui/action-log.md),
[gui/mcp-server.md](../gui/mcp-server.md) and
[gui/operation-progress.md](../gui/operation-progress.md), whose collector keeps
a status, a count and a fraction that nothing draws until this is built.

Its case is measured. A bundle adjustment of `dino_dog_toy-embedded` (85 images,
21 009 points, 392 489 observations) holds the GUI thread for 95 seconds: the
window is frozen for all of it, and every MCP call in that window fails on the
10 second apply timeout with "The viewer did not answer within 10 seconds. It
may be showing a modal dialog, or be mid-drag", which is the wrong reason. The
breakdown the frozen operation recorded is readable afterwards and says the
damping ladder was 87 of those 95 seconds.

## The problem

Some of the viewer's operations legitimately take a long time. A bundle
adjustment of a real reconstruction is minutes of solving; a resection in place
is a second; opening a large file with embedded patches is several. Every one of
them runs on the GUI thread, inside the frame that asked for it, so for its
whole duration the window does not redraw, does not orbit, does not answer a
keystroke, and does not answer an agent's call. The person at the window cannot
tell a solve that is working from one that has hung, cannot see how far along it
is, and cannot do anything else with the scene while they wait. An agent that
calls the tool gets a timeout, with no way to find out whether the work is still
running or was abandoned.

This proposes running those operations on a worker thread, and a **Background**
panel under the Scene tree that says what is running, on which node, how far
along it is, and what it has spent its time on so far. The panel shows the phase
table live while the operation runs, and that same table is what the Action Log
entry carries once it is done, so watching a long operation and reading about it
afterwards are the same view of the same data
([../gui/operation-progress.md](../gui/operation-progress.md)).

## Why this is safe here

The document model is value semantics
([gui/document-model.md](../gui/document-model.md)). A version is an immutable
reconstruction behind an `Arc` plus an overlay that only that version owns; the
base is never written through, `Arc::make_mut` appears nowhere, and there is no
interior mutability in the document. A bulk edit is already a **pure function
from the value at the cursor to the next value**.

That is exactly the precondition for moving one off the GUI thread. The worker
takes a clone of the `Arc` and reads it; the GUI thread keeps drawing the same
allocation and cannot disturb it; and **the document itself needs no lock,
because no part of it is shared mutably**. No part of this proposal wraps
`AppState` in a lock or hands a worker a reference into the scene, and the
reason it does not have to is that the value model was built to make this
possible.

Two small things are shared mutably, deliberately and under a lock, and neither
is the document: the collector the worker reports into and the panel reads
(§ "Reporting progress"), and the cancel flag. Both are a few words of state
that exist to be written by one thread and read by another, which is a different
thing from sharing a reconstruction.

The same argument says what is **not** safe: the GPU upload that follows a new
base. Uploads run against the device and the queue on the GUI thread, and moving
them is a different problem with a different answer. So this proposal shortens
the freeze to the upload, and makes what remains of it legible rather than
invisible. The undo that costs 2.4 s today is mostly upload, and it will still
cost most of that; what changes is that the panel and the log can say so.

## What the user sees

### Placement

The left column splits top to bottom. Scene keeps the top of it and a ninth tab,
`Tab::Background`, titled **Background**, takes the bottom at a default share of
0.28:

```
┌────────┬──────────────────┬───────────────┐
│ Scene  │    3D Viewer     │ Image Detail  │
├────────┼──────────────────┴───────────────┤
│Backgr. │  Image Browser │ Action Log      │
└────────┴──────────────────────────────────┘
```

It is a panel like any other: draggable, closeable, ticked in **Panels ▸
Background**, and given a home position of the left edge at 0.18 with Scene as
its default group-mate, so re-opening it from the menu puts it back beside the
tree.

### Idle

The panel is not blank when nothing is running. It shows the last operation of
the session, greyed: its name, its node, what it cost, and its phases, collapsed
under a toggle that works as the Action Log's does. A session that has run
nothing says `Nothing running` and no more.

### Running

```
┌ Background ──────────────────┐
│ Bundle adjust                │
│ perf-eval-embedded-guard     │
│ ████████████░░░░░░  iter 7/20│
│ normal equations, block 214  │
│ 12.4 s elapsed       [Cancel]│
│                              │
│ materialise          412 ms  │
│ solve              ▸ 11.9 s  │
│   linearise           4.1 s  │
│   normal equations    6.8 s  │
└──────────────────────────────┘
```

- **The name and the node** are the operation and the label it runs on, so a
  glance says what the window is busy with.
- **Progress** is a bar with the kernel's own count and unit when anything
  underneath reports one, and a spinner with the open phase's name when nothing
  does. The bar is a statement about how much of the work is behind you, never a
  prediction of when it will end: it steps at every phase boundary, because
  something did finish, and moves smoothly only across the stages that actually
  report counts ([../gui/operation-progress.md](../gui/operation-progress.md) § "Nesting").
  Nothing interpolates across a silent stage, and there is never a synthesised
  percentage, because a bar moving at a rate nobody measured makes a promise
  about the finish.
- **The status line** is the one thing the operation says it is doing right now,
  replaced as often as it likes: a file name inside a loop over images, the
  member being refined. It is live state and is never kept in the Action Log
  entry, because once the entry exists the answer is "finished"
  ([../gui/operation-progress.md](../gui/operation-progress.md) § "Status is not a message").
  It sits under the bar, because it is the words for the same thing the numbers
  beside the bar count.
- **Elapsed** counts up from the instant the operation started, which is the
  number the person is actually watching.
- **Cancel** is present always and enabled only when the operation can be
  cancelled, with a tooltip saying so when it cannot. The alternative, hiding
  the button, leaves the reader wondering whether they missed it.
- **The phase table** is the live form of what the Action Log entry will hold.
  Completed phases show their cost, the open one is marked and shows the time it
  has been open so far, and phases that have not started are not shown, because
  the panel does not know they are coming. It shows whichever level of timing is
  running, so ticking **Detailed timing** before starting a long operation is how
  somebody watches a kernel's internals
  ([../gui/operation-progress.md](../gui/operation-progress.md) § "Two levels").

### What the rest of the viewer does meanwhile

Everything that reads the scene keeps working. The node shows the version at its
cursor, which is the value the worker was handed and which nothing is mutating,
so orbiting, selecting, opening the Point Track panel, taking a screenshot and
opening a second file all behave exactly as they do when nothing is running.

What is refused is anything that would change the node the operation is running
on. An edit, an undo, a redo, a history jump, a save or a close of that node is
refused with *"{label} is busy: {operation} is still running."* Other nodes are
untouched and remain fully editable. The busy node's Edit menu items are greyed
with the same sentence as their tooltip, so the refusal is visible before it is
provoked.

**One operation at a time, viewer-wide.** Starting a second is refused with
*"{operation} is still running on {label}."* These operations saturate the
machine, so running two would make both slower and neither would finish sooner;
a queue is state with no demand behind it, and is a non-goal below.

Quitting while an operation runs abandons it. The result would have nowhere to
land, and a viewer that refuses to close is worse than a solve that has to be
run again.

### When it finishes

The finished value arrives on the channel with its row map and the sentence the
edit would have recorded. On the next frame the GUI thread pushes the version,
moves the selection forward through the map, and writes exactly the Action Log
entry the synchronous edit writes today, with the worker's phases attached to
it. The panel drops back to its idle form showing that same operation.

The entry's actor is whoever asked for the operation, not the viewer: an agent
that started a solve is the actor of the version it produced, however many
minutes later it lands.

Its **cost covers the whole operation**, not the frame that pushed it, and its
detail is what the worker reported. Both arrive through one call,
`ActionLog::record_done(kind, started, text, detail)`
([../gui/operation-progress.md](../gui/operation-progress.md), "The Action Log's side"): the
entry is written now, its cost is measured from an instant already past, and it
settles on the frame that draws the result as every other entry does. Without
the start instant a two-minute solve would report the six milliseconds of the
frame that installed it, which is the sort of number that discredits a whole
column.

A **cancelled** operation writes a failed entry, *"{operation} of {label}
cancelled after {elapsed}"*, and pushes no version. A **failed** one writes the
refusal the kernel returned, as the synchronous edit does.

Nothing is logged when an operation *starts*. The log records outcomes, not
intentions ([gui/action-log.md](../gui/action-log.md)), and what is running is
what the panel is for.

## Which operations go to the background

The rule is cost, not kind. A point edit is microseconds of work; sending it to
a worker would add a thread hop and a frame of latency to an operation that has
neither today, in exchange for nothing. So:

- **Point edits stay synchronous.** Delete point, add observation, remove
  observation, create point, move camera.
- **Bulk edits go to the background.** Bundle adjust first, since it is the one
  that freezes the window for minutes and the one the wire already warns about.
  Resect in place and delete image follow, once the first has settled.

Two further adopters are obvious and are not in the first commit: **opening a
file**, which is seconds of decode and is the other thing that freezes a fresh
session, and the **materialisation** an edit performs before calling a kernel.
Both are pure functions over values that nothing else holds, so both fit the
same mechanism without extending it.

## Reporting progress

The channel from a worker to this panel is
[../gui/operation-progress.md](../gui/operation-progress.md)'s `Progress`: one parameter the
kernel takes, carrying phases, messages, progress counts and the cancel flag.
The worker builds it over a collector the GUI thread shares, so the panel reads
what has been reported by locking that collector each frame rather than by
receiving it. Nothing about a phase or a message crosses the channel below; the
channel carries only the fact that something changed, and at the end the value.

**How much a given kernel reports is its own business**, and a kernel that
reports nothing still works: it is one phase, named by the caller, with a
spinner under it. `bundle_adjust` starts that way, so the first version of this
panel names `materialise`, `solve` and `row map` and spins through the solve,
which is already the difference between a frozen window and a window that says
what it is doing. Threading `Progress` into the solve's own iteration loop then
turns the spinner into a bar and the disabled Cancel into a live one, without
touching this panel, because the panel draws whatever it is told.

That is the shape of every later adopter too. The panel is finished when it can
draw phases, messages, a count and a spinner; each kernel then decides how much
of that it fills in.

## Rust API

`crates/sfm-explorer/src/background/`: `mod.rs` owns the process and the
channel, `panel.rs` the egui view, `tests.rs` the tests. The process is a field
of `AppState`, so the busy check is where every method that would need it
already is.

```rust
/// A long operation running off the GUI thread.
pub(crate) struct BackgroundProcess {
    /// What it is, for the panel and the refusals, and what it claims about
    /// itself.
    pub operation: Operation,
    /// The node it will install its answer into, and the node it locks.
    pub node: ReconId,
    pub label: String,
    pub started: std::time::Instant,
    /// Where the worker reports, and where the panel reads. Shared, taken by
    /// `&`, never borrowed mutably
    /// ([../gui/operation-progress.md](../gui/operation-progress.md) § "In the viewer").
    pub collector: Arc<progress::Collector>,
    /// Set to ask the operation to stop. The kernel polls it through the
    /// `Progress` built over `collector`; one that never polls is not
    /// cancellable, and the panel says so.
    pub cancel: Arc<AtomicBool>,
}

/// One kind of background operation: what to run, and what it claims about
/// itself.
///
/// `cancellable` is a **declaration**, not something discovered: a kernel that
/// never polls the flag is indistinguishable from one that has not reached a
/// poll yet, so nothing can infer this. The wrapper knows which kernel it
/// calls, so the wrapper states it, and a test holds it to the claim by
/// cancelling each operation that says `true` and asserting it stops.
pub(crate) struct Operation {
    pub name: &'static str,
    pub cancellable: bool,
}

/// What crosses the channel, worker to GUI thread. Phases, messages and counts
/// do not: they are in the collector both sides hold.
pub(crate) enum Report {
    /// Something was reported. Carries nothing; it exists to wake the loop.
    Progressed,
    /// The operation is over, one way or another.
    Done(Box<Finished>),
}

pub(crate) struct Finished {
    /// The next value, and the map from the input's rows to its own.
    pub outcome: Result<(SfmrReconstruction, PointMap), String>,
    /// The version's label, and the Action Log sentence up to the serials,
    /// which only the GUI thread can know.
    pub version_label: String,
    pub text: String,
}
```

and on `AppState`:

```rust
impl AppState {
    pub fn background(&self) -> Option<&BackgroundProcess>;

    /// Why an edit of `id` is refused right now, or `None`.
    ///
    /// Every method that gives a node a new version calls this first, so one
    /// sentence covers the menu's greying, the method's refusal and the wire's.
    pub fn busy_refusal(&self, id: ReconId) -> Option<String>;

    /// Start `operation` on `id`. Refuses when anything is already running.
    pub fn start_background(&mut self, operation: Operation, id: ReconId)
        -> Result<(), String>;

    /// Apply every report the worker has sent. Returns whether anything
    /// changed, so the frame knows to redraw.
    pub fn poll_background(&mut self) -> bool;

    /// Ask the operation to stop. Silently does nothing when it cannot.
    pub fn cancel_background(&mut self);
}
```

`poll_background` runs **in phase 0, before the MCP drain**. A completed
operation's version is then on screen in the frame it landed, and an agent's
call in that same frame reads the new value rather than the old one. The worker
wakes an idle event loop the way the MCP server does, with
`EventLoopProxy::send_event`; the `UserEvent` variant is unconditional rather
than behind the `mcp` feature, because the background panel is not an MCP
feature.

**Reports carry nothing, so they cannot pile up.** A worker that reports per
iteration sends a `Progressed` with no payload; the GUI thread drains whatever
is waiting, learns only that it should repaint, and reads the current state out
of the collector once. A thousand reports in a frame cost one lock rather than a
thousand repaints, and no event is ever in two places.

## On the wire

A tool that starts a background operation **returns when the operation starts**,
not when it finishes:

```json
{"started": "Bundle adjust", "label": "guard"}
```

The agent then polls, or reads the outcome out of the log. This is a break with
what `bundle_adjust` does today, and it is the right break: the alternative is a
call that reliably exceeds the 10 s apply timeout on any reconstruction worth
adjusting, so the agent gets an error for a solve that is going fine and has no
way to tell that from one that failed. Returning immediately makes the two
distinguishable, and the wire already has the vocabulary to follow up.

A new read tool, **`get_background_process`**, reports what is running:
the operation, the label, the seconds elapsed, the progress as `done`, `total`
and `unit` when there is one, the open phase, and the completed phases in the
shape [../gui/operation-progress.md](../gui/operation-progress.md) gives them. With
nothing running it reports the last operation of the session, marked `finished`,
so one call answers both "is it done" and "what did it cost".

`get_scene` gains a `background` block of the same shape, beside
`status_message`, so an agent that already polls `get_scene` learns that the
viewer is busy without a second call, and knows not to send an edit that would
be refused.

A **`cancel_background`** tool cancels what is running, and refuses when the
operation cannot be cancelled, with the same sentence the button's tooltip
carries.

The MCP spec's warning that a long `bundle_adjust` will time out while the solve
goes on is deleted, because the behaviour it describes is gone.

## Testing

`crates/sfm-explorer/src/background/tests.rs`, headless, driving a fake worker
over the real channel so the tests are about the seam and not about a solver:

- A report moves the panel's progress, appends its phases, and asks for a
  redraw.
- `Done` with a value pushes exactly one version, moves the selection through
  the map, and writes one Action Log entry whose cost spans the whole operation
  rather than the frame that pushed it.
- `Done` with an error writes one failed entry and pushes nothing.
- A cancel writes one failed entry naming the elapsed time, and pushes nothing.
- Every edit, undo, redo, jump, save and close of the busy node is refused, with
  the same sentence in each case; the same operations on another node are
  allowed.
- Starting a second operation is refused while one runs.
- **The value was not copied to run in the background**: the base `Arc` the
  worker was handed is pointer-equal to the one the node is still drawing.

Panel, through `test_support::run_frame_headless`:

- The three states paint what they should: idle with nothing run, idle with a
  finished operation, and running with a bar and with a spinner.
- Cancel is disabled, with its tooltip, for an operation whose kernels do not
  poll the flag, and live for one that does.
- A status replaces the previous one rather than adding a line, and the panel
  shows nothing there for an operation that has never set one.
- **Every `Operation` that declares `cancellable` really is**: cancelling each
  one stops it and writes the cancelled entry. A declaration nothing checks is a
  declaration that rots.

`crates/sfm-explorer/tests/ui_basic.rs`: the Background panel is in the
accessibility tree, and **Panels ▸ Background** ticks it.

`crates/sfm-explorer/src/layout/tests.rs` and `dock/tests.rs`: the default
layout's left column is a top-bottom split of Scene over Background at 0.28, and
Background's home position is the left edge with Scene as its group-mate.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| left column split (`Layout::default`) | `0.72` | Scene's share of the left column; Background takes the rest |
| Background home edge / share | left / `0.18` | Same edge and share as Scene, whose group-mate it is |

## Non-goals

- **A queue.** One operation at a time, and a second is refused rather than
  queued. These saturate the machine; two at once finish no sooner.
- **Backgrounding point edits.** They cost microseconds, and a frame of latency
  is a worse deal than the freeze it avoids.
- **Moving GPU uploads off the frame.** They run against the device and the
  queue on the GUI thread. What remains of the freeze after this proposal is
  mostly upload, and shortening it is a separate piece of work with a different
  mechanism.
- **Editing a node while an operation runs on it.** The operation is a function
  of the value it was handed, and an edit underneath it would produce a version
  whose parent is not the version it was computed from.
- **Persisting a running operation.** Quitting abandons it.
- **Progress from a kernel that does not report it.** The panel shows a spinner
  and the open phase's name, and says nothing it cannot measure.

## Open questions

- Whether an operation should be allowed per node rather than one viewer-wide.
  Per node is the natural generalisation and costs little in the state; it is
  held back because nothing has asked, and because two solves at once is a
  slower way to do one thing.
- Whether the idle panel should keep more than the last operation. A short
  history of what was run and what it cost is the Action Log's job, and the
  panel would be duplicating it; but the panel is where a reader looks for cost,
  and three rows there may save a scroll.
- Whether a background operation should be allowed to start while a modal dialog
  is open. Today the dialog is what starts it, so the question is only about the
  file dialogs, which stop the GUI thread anyway.
