# Background task queues

**Status:** Draft. Amends [gui/background-tasks.md](../gui/background-tasks.md)
(the rule "one operation at a time, viewer-wide", the non-goals "A queue" and
"One operation per viewer rather than per node", the panel and the wire),
[gui/bench.md](../gui/bench.md) § "Live evaluation",
[gui/edits/switch-camera-model.md](../gui/edits/switch-camera-model.md)
§ "The proposal's job", [gui/mcp-server.md](../gui/mcp-server.md)
(`get_background_task` and `cancel_background_task`, which become
`get_background_tasks` and `cancel_background_tasks`, `get_scene`'s
`background_task` block and the handle) and
[gui/panel-layout.md](../gui/panel-layout.md) (the panel's title and wire
name). The
[Track Table](track-table.md) puts its column job and its reference picks on
the fast queue that this draft adds.

Decided:

- There are two queues, **slow** and **fast**. Each runs one task at a time,
  and the two run at the same time.
- Each operation declares its queue. Nothing moves between queues while it
  runs.
- A task started while its queue is busy **waits** in order, and it can be
  cancelled while it waits.
- **Open** goes in the fast queue, so a file can be opened while a bundle
  adjustment runs.
- The bench's **live evaluation** and the **camera model proposal job** become
  fast-queue tasks. They are **quiet** tasks: they write no Action Log row for
  running, and they are cancelled and queued again when someone asks for a
  task. Their dedicated threads are retired.
- The panel's wire name becomes `background_tasks`, and the two tools become
  `get_background_tasks` and `cancel_background_tasks`. None of the old names
  is kept as an alias.

Not decided: see § "Open questions".

## Purpose

SfM Explorer runs long operations, such as a bundle adjustment or a
retriangulation of every point, on a worker thread, so the window keeps drawing
while they run. Today only one such operation can run at a time, across the
whole viewer, so a 95-second bundle adjustment of one reconstruction also stops
a person from opening another file, fitting a track on the bench of another
reconstruction, or sorting the Track Table. This draft gives the viewer two
queues:

- The **slow queue** holds the operations whose cost grows with the whole
  reconstruction and that use every core: the whole-reconstruction edits.
- The **fast queue** holds the rest: opening files, bench steps, and the quiet
  work the viewer does by itself to keep its panels current.

A long solve then holds up only other slow operations, and the rest of the
viewer keeps working beside it.

The reason the standing spec gives for running one task at a time still holds
for the slow queue. Those operations saturate the machine, so two at once
finish no sooner than one after the other. It holds much less for a fit of one
track or a computation for one panel. Keeping them in their own queue costs
the slow task some competition for the cores while they run, and saves the
person a wait of minutes.

## What a task declares

`background::Operation` gains three fields beside `name`, `cancellable` and
`kind`:

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Queue { Slow, Fast }

pub(crate) struct Operation {
    pub(crate) name: &'static str,
    pub(crate) cancellable: bool,
    pub(crate) kind: Kind,
    /// The queue it waits in and runs from. Fixed per operation.
    pub(crate) queue: Queue,
    /// Whether it locks its node from the moment it is queued until it lands
    /// (§ "The rules"). Every operation that pushes a version locks.
    pub(crate) locks: bool,
    /// Whether it is work the viewer starts by itself rather than work
    /// someone asked for (§ "Quiet tasks").
    pub(crate) quiet: bool,
}
```

| Operation | Queue | Locks its node | Quiet |
|---|---|---|---|
| Bundle adjust | slow | yes | no |
| Convert to embedded patches | slow | yes | no |
| Retriangulate all points | slow | yes | no |
| Refit all points | slow | yes | no |
| Switch camera model | slow | yes | no |
| Prune covered observations | slow | yes | no |
| Add image to tracks | slow | yes | no |
| Build index files | slow | yes | no |
| Open | fast | no (it makes a new node) | no |
| Fit track, Set track stage, Fit normal, Finite difference normal, Grid plane normal, Search descriptors, Geometry search, Create track at pixel, Find nearby tracks | fast | yes | no |
| Live evaluation | fast | no; it does not start on a locked node (§ "Quiet tasks") | yes |
| Camera model proposal | fast | no; it pushes no version (Apply installs its answer), and an open proposal already refuses every task on its node | yes |
| Track Table columns, Track Table reference picks | fast | no | yes |

`Operation::ALL` gains the four quiet operations, and a test checks the table
above against each operation's declared values. The queue is fixed rather
than estimated from the reconstruction's size. That keeps it predictable: the
same operation always waits behind the same work, a test can name the queue,
and nothing has to be measured before the queue is known.

## The rules

**Each queue runs one task at a time.** A task started while its queue is
running another waits behind any tasks already waiting there, in the order
they were started, with the exceptions for quiet tasks below. The two queues
are independent. A fast task never waits for a slow one, except through the
node lock below.

**A locking task locks its node from the moment it is queued.** Today a node
is busy while a task runs on it. Under the queues a node is busy while a
locking task is running or waiting for it, in either queue. Every edit, undo,
redo, history jump, save and close of a busy node is refused with today's
sentence, *"{label} is busy: {operation} is still running."*, or, for a
waiting task, *"{label} is busy: {operation} is waiting to run."* Starting a
second locking task on a busy node is refused with the same sentence, and so
is starting one in the other queue. So a node has at most one locking task at
any time, and what the standing spec says about an edit made under a running
operation still holds: nothing moves the node's cursor between the moment the
task is queued and the moment it lands.

**A task's job is built when it is queued.** The job builders check the node's
lock themselves (`resolve_switch`, `retriangulate_refusal` and the others
check `busy_refusal`), so they run before the task is queued and the lock is
taken. A builder that refuses refuses the start, as it does today. The job
captures the value at the cursor then, and because the node is locked from
that moment, that value is still the one at the cursor when the task starts.

**A task that locks nothing runs beside a locking one.** Open makes a new
node. The Track Table's two tasks read the version at the cursor and keep
their answer in the table. They run on a busy node against the version on
screen, which is the value the slow task was handed. When the slow task lands
its version, the document serial moves, and each drops its stale answer and is
queued again, as each already does after an edit. The camera model proposal
is not in this group in practice: an open proposal refuses every task start on
its node (`proposal_refusal`), and a proposal cannot be opened on a busy node
(`switch_model_refusal`), so the two never meet.

**What stops being refused.** `running_refusal` goes. Starting a task is
refused by its node's lock, and in Step 1, before tasks can wait, by its own
queue being busy (*"The {slow | fast} queue is busy: {operation} is still
running on {label}."*). `File > Open...` (`app/menu.rs`) is greyed only while
the fast queue refuses starts in Step 1, and not at all once tasks can wait.

**Quitting** abandons every running and waiting task, as it abandons the one
task today.

## Quiet tasks

A quiet task is work the viewer starts by itself to keep a panel current.
Nobody asked for it, so nothing that someone asked for waits behind it:

- **It is cancelled and queued again when someone asks for a task.** When a
  task that is not quiet is queued in the fast queue while a quiet one is
  running, the quiet task's cancel flag is set. The asked-for task starts once
  the quiet task's thread reports that it has stopped, because a queue still
  runs one task at a time. How soon that is depends on how often the quiet
  task's kernels check the flag: an evaluation checks between its rounds and
  between the views it renders, and the proposal's camera switch does not
  check at all (§ "Open questions"). The quiet task is then queued again
  behind the asked-for task. Each quiet task already drops a cancelled run and
  repeats it, so this loses at most the work the cancelled run had done.
  Waiting tasks that are not quiet always start ahead of waiting quiet ones.
- **Each source keeps at most one waiting entry.** A source is one producer of
  quiet work: the live evaluation, the proposal, the table's columns or the
  table's picks. A new request replaces the source's waiting entry rather than
  adding a second, because a quiet task's input is the state of a panel, and
  only the latest state matters. That is how the evaluation driver and the
  proposal job decide what to run next today.
- **A waiting quiet task is checked again when it would start.** One whose
  node has become locked since it was queued is dropped, and its source queues
  it again when the node is free. Today's evaluation driver checks this only
  when it picks a subject (`next_unevaluated`).
- **No evaluation lands into a locked node.** A locking task queued on a node
  cancels a running evaluation of that node. Cancelling is cooperative, so an
  evaluation that finished before it read the flag still reports an answer;
  `land_evaluation` drops that answer, and writes no *Evaluated* row, when
  the node is locked at the moment it would land. So a bench step built at
  queue time from the track on the bench still finds that track there when it
  starts.
- **A dropped or replaced entry is not a failure.** A waiting quiet entry that
  is dropped at start, or replaced by a newer request from its source, never
  runs. The queue tells its source so with a `Dropped` message on the
  source's channel, sent before the job and its sender are dropped. The
  source treats it as not run: the evaluation driver does not record a
  failure, and it picks the subject again when the node is free. Today a
  disconnected channel reads as *"The evaluation stopped without an answer."*
  and settles the track as failed, which is wrong for a job that never ran.
- **Among quiet tasks** the order is fixed: the camera model proposal (its
  preview is on screen while a person decides), the viewed track's
  evaluation, the focused item's evaluation, the Track Table's columns, the
  Track Table's reference picks, and then the evaluations of the other bench
  items in the order the evaluation driver uses today.
- **A quiet task that a wire call waits on is treated as asked for.** A
  `find_points` call that needs a computed column defers its reply until the
  column job has filled it ([track-table.md](track-table.md) § "find_points").
  While a call waits on it, the column job is not cancelled for other tasks
  and is reported in `get_scene`'s list. A waiting column job moves to the end
  of the asked-for tasks waiting in the fast queue, as if it had just been
  queued by the call. A running one whose cancel flag was already set before
  the call arrived is let finish, its answer is kept, and its source queues
  the next run as asked for. The call's reply carries the column values, so
  nothing has to look the task up afterwards, and it is not kept in the
  record of finished tasks. It is quiet again once the call has its answer.
  This rule is built with the deferred reply, in Step 5 of
  [track-table.md](track-table.md).
- **It writes no Action Log row for running, and leaves no finished record.**
  It does not go into the record of finished tasks (§ "What a task
  remembers"), so a burst of evaluations during a slider drag cannot push out
  the record of a task someone is waiting on. The live evaluation keeps its
  own row, *Evaluated {label}: the bars turned …*, written when a verdict
  moves, because that row reports a change to the bench and not that the task
  ran.
- **It reports its answer through its source's own channel.** A quiet task's
  job is a `QuietJob`, a `FnOnce(&Progress) + Send` that sends its answers to
  its source, as the evaluation and the proposal send theirs today. That
  covers a task with more than one answer: the proposal sends its
  *Switched* message before its *Answered* one, and the Track Table's column
  job publishes its stored columns before its computed ones. The queue
  carries only that the task has finished, so `Finished` gains no variant per
  quiet task.
- **The panel shows it in one line** (§ "The panel"). It is not reported in
  `get_scene`'s list or in the busy notice, except while a wire call waits on
  it, so a non-empty list still means "someone's work is running".

**Live evaluation as a quiet task.** The evaluation driver
(`bench::live::drive_bench_evaluation`) keeps its choice of what to evaluate
next and its rule that a result whose inputs have moved is dropped. It stops
owning a thread: it queues a *Live evaluation* task for its next subject and
reads the answer from its channel as it does today. The doc comment in
`bench/live.rs` that says it writes no Action Log row is corrected at the same
time.

**The camera model proposal as a quiet task.** The proposal's job keeps its
generation, its early *Switched* message and its keep rule
(`keep_proposal_message`). Apply still installs a held answer on the GUI
thread with no task, and starts *Switch camera model* on the slow queue when
there is none. Its job runs `switch_and_retriangulate`, the same computation
the slow operation runs, so it can hold the fast queue for seconds on a large
reconstruction, and each time it is cancelled for an asked-for task it starts
over. The glossary entry that calls it "not a background task"
(`specs/GLOSSARY.md`) and the module doc of `camera_model_proposal/job.rs` are
rewritten.

## The panel

The panel's title becomes **Background Tasks**, and its wire name becomes
`background_tasks` (`Tab::BackgroundTask` becomes `Tab::BackgroundTasks`).
The old name is not kept as an alias. This is the rule
[panel-layout.md](../gui/panel-layout.md) already states for a renamed panel,
as when `point_track` became `track_view`:

- **A layout file** that names `background_task` is refused as a whole with
  the unknown-panel message, which lists the new name. Reset Layout is the way
  back.
- **A default layout file** that names it starts the viewer on the stock grid,
  as any default layout file the viewer cannot read does.
- **On the wire**, `panel_name: "background_task"` is refused by every tool
  that takes a panel name, with the same message.

`panel-layout.md`'s list of panel names, its default-layout JSON and its
validation examples, the `DEFAULT_JSON` test in `layout/tests.rs`, and the
backticked panel names that `mcp/tests/catalog.rs` checks in
[mcp-server.md](../gui/mcp-server.md) all change with it.

```
┌ Background Tasks ──────────────────────────────┐
│ Slow                                           │
│ Bundle adjust   dino            ▓▓▓▓░░  41 s  ✕│
│   waiting: Retriangulate all points  bull     ✕│
│ Fast                                           │
│ Fit track       bull: track_3       done 0.4 s │
│   quiet: Live evaluation (viewed pt3d_…_830)   │
├────────────────────────────────────────────────┤
│ Bundle adjust: phases                          │
│   linearise              4.4 s                 │
│ ▶ damping ladder        36.1 s                 │
└────────────────────────────────────────────────┘
```

- **One section per queue.** Each section shows the queue's running task in
  today's running form (name, label, bar or spinner, status line, elapsed
  time, Cancel), then its waiting tasks, one line each with a cancel button.
  An idle queue shows the last task that finished from it, read from the
  record of finished tasks, in today's idle form.
- **Quiet tasks** show as one dim line in the fast section: the running quiet
  task's operation and subject, with no bar and no Cancel. A quiet task runs
  only while no asked-for task runs in the fast queue, so the section then
  shows the fast queue's last finished task above the quiet line, as in the
  picture.
- **The phases** below the sections are those of one task: the one a person
  last clicked, and otherwise the slow queue's running task, the fast queue's,
  or the last task to finish, in that order. The panel still shows one phase
  list, so it stays the height it is.

The title is written in `Tab::title` (`dock.rs`), which the MCP input tools
read through `panel.title()`. It is also written in
[panel-layout.md](../gui/panel-layout.md),
[background-tasks.md](../gui/background-tasks.md),
[mcp-server.md](../gui/mcp-server.md), [action-log.md](../gui/action-log.md),
[multi-panel-image-browser.md](../gui/multi-panel-image-browser.md), a comment
in `lib.rs` and the module doc of `camera_model_proposal/job.rs`. All of them
change with it.

## What a task remembers

`AppState::background_task: Option<BackgroundTask>` and
`last_background_task: Option<FinishedTask>` become one state per queue and a
record of finished tasks:

```rust
pub(crate) struct QueueState {
    pub(crate) running: Option<BackgroundTask>,
    pub(crate) waiting: VecDeque<WaitingTask>,
}

pub(crate) struct WaitingTask {
    pub(crate) id: u64,
    pub(crate) operation: Operation,
    pub(crate) node: Option<ReconId>,
    pub(crate) label: String,
    pub(crate) queued: Instant,
    pub(crate) actor: Actor,
    /// Built when the task was queued (§ "The rules").
    pub(crate) job: WaitingJob,
}

pub(crate) enum WaitingJob {
    /// A task someone asked for, which lands through `finish`.
    Asked(Job),
    /// A quiet task, which answers on its source's channel. `rank` is its
    /// place in the fixed order among quiet tasks (§ "Quiet tasks"), which
    /// for the live evaluation depends on its subject.
    Quiet { source: QuietSource, rank: u8, job: QuietJob },
}
```

- **The start functions return the task's id.** Today a caller reads the task
  it has just started with `state.background_task().expect(…)`, in the MCP
  edit and bench tools, in `mcp/write.rs` and in Surface Tiling. Two readers,
  `started_or` in `mcp/bench.rs` and Surface Tiling, also test whether
  `background_task()` is set to tell a started task from a step that had
  nothing to do (`start_bench_stage_at` returns `Ok(())` for a stage change
  with no effect, and starts nothing). Both read an unrelated task once a task
  can wait or run in the other queue. So every start function returns
  `Result<Option<u64>, String>`: the id it gave the task, waiting or running,
  or `None` when it started nothing.
- **Tasks are found by id.** `AppState::task(id)` answers whether a task is
  running, waiting or finished, and with its record. It replaces the readers
  that decide "running" from `background_task()` alone and "finished" from
  the single `last_background_task` slot:
  - the deferred MCP reply (`mcp::edit::background_reply`);
  - Surface Tiling's wait (`bench/surface_tiling.rs`);
  - the Scene tree's busy marks (`scene_graph/mod.rs`) and the index-files
    status (`index_files.rs`), which read the one running task.

  With two queues finishing independently, a fast task could land between a
  slow task's landing and the poll that reads it, and push the slow task out
  of the single slot. Both queues are polled in the same step of the frame,
  before the MCP replies are resolved, so two tasks can land in one frame.
- **Finished tasks are kept in a ring** of the last `FINISHED_KEPT` (16) tasks
  that were not quiet, including tasks cancelled before they started. Quiet
  tasks are not kept (§ "Quiet tasks").
- **`BusyNotice`** holds every running task that is not quiet, so the MCP
  timeout message names each running task and its node, as it names the one
  today.
- **Threads are unchanged in kind.** A task still runs on a named
  `std::thread` of its own, spawned when it starts, and its kernels still use
  rayon's global pool. The queue only decides when a task starts. A fast task
  running beside a bundle adjustment shares the global pool with it, which
  slows the adjustment while the fast task runs (§ "Open questions").

## The wire

- **The handle** gains `queue` (`"slow"` or `"fast"`) and `waiting: true`
  when the task has not started yet. `REPLY_DIRECTLY_WITHIN` is measured from
  when the task was queued, so a call whose task waits past the 200 ms
  threshold gets the handle, with `waiting: true`, rather than an answer that
  takes longer than 200 ms to arrive.
- **The two tools are renamed** to `get_background_tasks` and
  `cancel_background_tasks`, matching the panel's new wire name. The old names
  are not kept as aliases: a call to either is refused as an unknown tool.
- **`get_background_tasks`** takes an optional `operation_id`. With one, it
  answers about that task in today's `get_background_task` shape, whether it
  is running, waiting (`waiting: true`, no phases) or finished (from the
  ring). Without one, it answers about both queues:

  ```jsonc
  {
    "slow": {
      "running": { /* today's running shape, plus "queue" */ },  // or null
      "waiting": [ { "operation": "Retriangulate all points",
                     "reconstruction_label": "bull", "operation_id": 9 } ],
      "last_finished": { /* today's finished shape */ }           // or null
    },
    "fast": { /* the same three fields */ }
  }
  ```

  Quiet tasks are not reported. Each running task carries its phases, capped
  as today, so one call answers what both queues are doing.
- **`cancel_background_tasks`** takes an optional `operation_ids` list, naming
  running or waiting tasks. Without it, it cancels every running and waiting
  task that is not quiet. A task that cannot be cancelled is listed in the
  reply's `refused` with the sentence the panel's tooltip carries, and the
  others are cancelled. Cancelling a waiting task removes it from its queue and
  writes *"{operation} of {label} cancelled before it started"*. The reply
  lists `cancelling` (running tasks whose flag was set), `removed` (waiting
  tasks) and `refused`, each by `operation_id` and `reconstruction_label`.
- **`get_scene`'s block** becomes `background_tasks`, a list of the running and
  waiting tasks that are not quiet, each in today's summary shape plus `queue`
  and `waiting`. It is empty when nothing a person or an agent asked for is
  running. The key is renamed because the value changes from an object or
  `null` to a list. A reader that tested `background_task != null` now tests
  that the list is not empty.
- **Texts and tests that change with it:**
  - the tool descriptions that say "one operation runs at a time, viewer-wide"
    or "Refused while another background operation runs"
    (`mcp/tools/catalog/background.rs`, `mcp/tools/catalog/viewer.rs`,
    `mcp/mod.rs`);
  - [mcp-server.md](../gui/mcp-server.md) § "get_background_task" and
    § "cancel_background_task", renamed for the new tools, the tool table, the
    `get_scene` example, every mention of the old tool names (the handle's
    "poll with", the timeout message's pointer) and the sentences that say a
    start is refused while another operation runs;
  - the tool counts and read-tool list in `mcp/tests/catalog.rs` and the
    tool names in `mcp/logged.rs`;
  - the tests that pin today's shapes and refusals: the `get_scene` key set
    and the handle, cancel and timeout tests in `mcp/tests/edit.rs`,
    `a_second_operation_is_refused_while_one_runs` in `background/tests.rs`,
    and the refused open in `state/open/tests.rs`.

## Action Log

Queueing a task writes no row, as starting one writes none today. A task that
is not quiet writes its row when it lands, as today, and the time in that row
runs from when the task started, not from when it was queued, so it is the
cost of the work and not of the wait. A waiting task that is
cancelled writes the *cancelled before it started* row. A refused start
writes what its caller writes today.

## Steps

Each step is one PR through `implement-change`:

1. **Two queues with no waiting.**
   - `Queue` and `locks` on `Operation`.
   - Two `QueueState`s in `AppState`.
   - `running_refusal` replaced by the node lock and the busy-queue refusal.
   - Start functions that return the id, the lookup by id and the ring of
     finished tasks, with the MCP reply, `started_or`, Surface Tiling, the
     Scene tree and the index-files status moved to them.
   - The test seams moved to ids: `finish_background_task`, which drives the
     one running task, takes the id it waits for, and `open_now`
     (`state/open.rs`) reads its open's record by id rather than from
     `last_background_task`.
   - The panel's two sections, its title and its wire name
     (`background_tasks`).
   - The wire changes for two running tasks, with the two tools renamed.
2. **Waiting.**
   - `WaitingTask`.
   - Starting the next task when a queue's task lands.
   - The node lock from the moment a task is queued.
   - Cancelling a waiting task.
   - `waiting` on the wire.
3. **Quiet tasks, and the live evaluation moved onto the fast queue.**
   - The `quiet` flag and `QuietJob`.
   - Cancelling and requeueing for asked-for tasks, one waiting entry per
     source, the check at start, and the fixed order.
   - The panel's quiet line.
   - The evaluation's own thread is retired.
4. **The camera model proposal moved onto the fast queue.** Its own thread is
   retired.

The Track Table's two quiet tasks come with its own steps (Steps 2, 4 and 5 of
[track-table.md](track-table.md)), after Step 3 here.

## Testing

- The declared queue, lock and quiet values of every operation match the
  table in § "What a task declares".
- Two operations in different queues run at the same time, and two in the
  same queue run one after the other, in the order they were started.
- A locking task in either queue makes its node busy from the moment it is
  queued: an edit, a second locking task and a close are each refused with the
  waiting sentence, and accepted again once the task lands. The task itself
  starts and lands.
- A task that locks nothing runs on a busy node, and its answer is dropped
  when the slow task's version lands.
- A running quiet task is cancelled when an asked-for task is queued; the
  asked-for task starts after the quiet one reports; the quiet one runs again
  after it; and its source never has more than one waiting entry.
- A locking task queued on a node cancels a running evaluation of that node,
  and the evaluation does not land, including one that finished before it
  read the flag.
- A waiting evaluation that is dropped at start or replaced is not recorded as
  failed, and is evaluated once its node is free.
- A start that starts nothing returns `None`, and its caller does not read
  another task as its own.
- A deferred MCP reply finds its task by id after another task has landed in
  the same frame. This is the case the single slot gets wrong.
- A start function's id names the task it started, whether that task waits or
  runs.
- Cancelling a waiting task removes it, writes its row, puts it in the ring and
  leaves the running task alone.
- `get_scene`'s `background_tasks` is empty while only quiet tasks run, and
  lists a column job while a `find_points` call waits on it.
- `get_background_tasks` with no argument reports both queues' running,
  waiting and last finished tasks and no quiet task; with an `operation_id` it
  reports that task whether it is running, waiting or finished.
- `cancel_background_tasks` with no argument cancels every running and waiting
  task that is not quiet, lists one that cannot be cancelled under `refused`,
  and leaves quiet tasks alone. The old tool names are refused as unknown.
- A layout file, a default layout file and a wire call that name
  `background_task` are refused with the unknown-panel message naming
  `background_tasks`.

## Non-goals

- **More than one task per queue at a time.** The slow queue's operations
  saturate the machine, and running the fast queue's one after another is
  simpler to reason about than scheduling several at once.
- **Two locking tasks on one node.** A node has one cursor, and two tasks
  computed from it would make two versions with the same parent.
- **Moving a task between queues.** A queue is fixed per operation.
- **Restoring waiting tasks after a restart.** Quitting abandons them, as it
  abandons running ones.

## Open questions

- **Competition for the cores.** A fast task beside a bundle adjustment shares
  rayon's global pool with it. How much a Fit or a Track Table column job slows
  the adjustment, and how much the adjustment slows them, should be measured
  on `dino_dog_toy-embedded`. If the fast tasks wait too long for the pool, a
  small dedicated rayon pool for the fast queue is the alternative. It costs a
  second set of threads, and it means a fast task's kernels have to be
  installed into that pool with `ThreadPool::install`.
- **Long tasks in the fast queue.** Some fast-queue operations are not reliably
  short: Open renders display patch bitmaps by decoding every photograph,
  Geometry search decodes every image that could see its patch, and Find nearby tracks and
  Create track at pixel read every `.sift` file and the cluster patches. An
  open can therefore wait behind one of them for seconds. Measure them, and
  decide whether Open should go ahead of waiting bench steps, or whether the
  longest of them belong in the slow queue.
- **Add image to tracks and Build index files** are put in the slow queue
  because their cost grows with the reconstruction. Both take about a second
  on `seoul_bull_sculpture`. If they are usually that short, the fast queue
  may suit them better.
- **The proposal's cancellation.** Its camera switch takes no `Progress`, so a
  cancel takes effect only after the switch, and the proposal starts over
  after every asked-for task. If that makes the preview slow to appear while
  bench work runs, the proposal could be exempt from cancellation once its
  switch has started.
- **The quiet order.** Whether the Track Table's columns should go ahead of
  the focused item's evaluation when the table is the panel in view.
- **A bench step on the node a slow task holds** is refused, because the bench
  step pushes a version into the same history. The bench could be given a
  history of its own, but that is a document-model change and out of scope
  here.
