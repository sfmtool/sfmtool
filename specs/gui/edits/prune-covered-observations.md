# Prune Covered Observations

One action that hands a reconstruction's coarse evidence over to the finer
features that supersede it: every observation another one covers in the same
photograph is retired, and the points left standing on fewer than two go with
them. It is chosen from the reconstruction's row in the Scene Graph and installs
the answer as that node's next **version**, which an undo steps back out of.

Nothing is re-solved. No point moves, no camera is re-posed, no lens is touched
and no observation is added. A surviving point comes back with the position,
patch frame, bitmap, colour and constraint it had, and only its observation list
is shorter. What is decided is which sightings a reconstruction should still be
standing on, and nothing else.

Related specs:
[../../core/reconstruction/prune-covered-observations.md](../../core/reconstruction/prune-covered-observations.md)
(the operation underneath),
[../../core/analysis/covered-by-finer.md](../../core/analysis/covered-by-finer.md)
(the rule it applies), [../scene-graph.md](../scene-graph.md) (the
reconstruction row's menu), [../background-tasks.md](../background-tasks.md)
(the worker it runs on), [README.md](README.md) (the other edit families).

---

## Purpose

A reconstruction solved from features at many scales holds, in the same patch of
the same photograph, a wide feature and a narrow one describing the same
surface. The wide one is the worse evidence: its support spans many image
pixels, so it averages over whatever detail lies under it, and where that detail
sits at more than one depth its triangulated depth is a blend of them. A
reviewer looking at a solve whose coarse structure is smeared wants a way to
retire that evidence in favour of what the finer features already say, without
moving anything.

It is deliberately **not** a filter on quality. Nothing is judged by its
residual, its track length or its colour; what decides is whether a finer
feature already claims the same pixels. So a reader can say what the version
did in one sentence, and the count of what went is a statement about scale
rather than about a threshold somebody picked.

It is also **not** an adjustment and not a retriangulation. Those move the
geometry; this changes what the geometry is fit against, and leaves the fitting
for a reviewer to ask for afterwards
([bundle-adjust.md](bundle-adjust.md),
[retriangulate-point.md](retriangulate-point.md)).

---

## Invocation

Right-click a reconstruction row in the Scene Graph tree and choose
`Prune Covered Observations`. It sits directly under
`Retriangulate All Points`, because the two are the pair a reviewer reaches for
together: one re-reads what each track says, the other decides which tracks
should still be saying it.

The entry is greyed, with the reason as hover text, when the node cannot be
pruned: its points carry no patch frame to read a footprint off, its
observations carry no pixel for one to sit at, no image carries a pose, or an
operation is already running on it. One function answers that question --
`state::edits::prune_covered_refusal` -- and the menu entry, the wire's tool and
the operation itself all ask it, so the greyed entry and a call that asks anyway
give one answer.

Over MCP the same action is `prune_covered_observations`
([../mcp-server.md](../mcp-server.md)), which additionally takes the three
thresholds; the menu passes the operation's defaults.

---

## Mechanism

Everything below lives in `sfmtool-core` as one function,
`reconstruction::prune_covered_observations` in
[prune_covered.rs](../../../crates/sfmtool-core/src/reconstruction/prune_covered.rs).
The viewer wraps it once, as a background job, in
[edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

Per observation the operation projects its point's patch frame into the
observing camera and reads two lengths off that one projection: the **radius**,
the mean of the projected frame's two column norms, and the **footprint**, half
of it. An observation is retired where another observation in the same image, on
another point, sits inside that footprint with a radius at least twice smaller.
The coarse side goes, never the fine one. A point left with fewer than two
survivors is dropped and its survivors with it.

Two populations are spared by name. A point the value **ranges or holds** is a
statement a reviewer made by hand, so none of its observations is retired
however well covered -- and it goes on covering other points' observations,
because the reason it was pinned says nothing about the evidence it offers its
neighbours. An observation whose frame does not project to a usable radius is
neither retired nor able to retire anything, and is counted.

It is a **bulk edit**: the overlay is folded in, the retired rows and the
dropped points are taken out, and the next version's base is the value that is
left. Points are renumbered, so the panels' caches for the node are dropped and
the selection follows the map, exactly as after an image deletion.

### On a worker

The action is a background operation, `Prune covered observations`
([../background-tasks.md](../background-tasks.md)). It reports the core
function's four stages -- folding the overlay, measuring the footprints, reading
the rule, writing the value back -- into the Background panel, and it is
**cancellable**: the flag is read at each of those boundaries, and a cancelled
run pushes no version and writes a failed entry carrying how far it got.

---

## The version

| | |
|---|---|
| Version label | `Pruned covered observations in seoul_bull` |
| Entry | that label, then the counts, then the serials |

The entry's tail is what the run did, in the same shape every other bulk edit
reports in: how many of how many observations were retired, how many points were
dropped, how many pinned rows were spared, and how many rows carried no usable
footprint. Each clause appears only when its count is non-zero, so a run that
spared nothing does not say so.

**A prune that retires nothing pushes no version.** The operation ran and found
nothing to do, which is a success and not a refusal, so the row reads
`Pruned covered observations in seoul_bull: no effect, no observation is covered
by a finer one` and the history is untouched. A version that changed nothing
would be a row a reader cannot tell from one that did something, and it would
cost an undo to get past.

A refusal pushes no version and writes one failed entry, in the state's own
words, which are the words the greyed menu entry carries.

---

## Testing

The core function's own tests are
[../../core/reconstruction/prune-covered-observations.md](../../core/reconstruction/prune-covered-observations.md)
§ "Testing"; what is tested here is the wrapping.

- **The edit pushes one version with a new base**, shorter than its input, under
  the label `Pruned covered observations in <node>`, and the entry names the
  counts.
- **A prune that retires nothing pushes no version** and writes one successful
  row saying so.
- **A cancelled run pushes no version**, leaves the observations where they
  were, and writes one failed entry saying it was cancelled.
- **A node with no patch frame is refused** in the sentence the greyed entry
  carries, and the refusal quotes it.
- **The Scene tree entry** is live on a node with patch frames, drawn directly
  under `Retriangulate All Points`, drawn and dead without them, and drawn and
  dead while an operation runs on the node.
- **The operation's cancellability claim** is held to by the same test that
  holds every other background operation to its claim.
- **The wire**: `prune_covered_observations` defers to the worker and comes back
  with the version, and the three thresholds cross it, a call that names none
  taking the operation's own defaults.

---

## Non-goals

- **Re-solving what is left.** The points keep the geometry they had. A reviewer
  who wants the structure re-read at the shortened tracks asks for that
  separately ([retriangulate-point.md](retriangulate-point.md)), which is what
  makes it possible to say which of the two moved a point.
- **Judging an observation by its residual.** What decides is scale and
  position, not fit. A residual filter is a different operation and would read
  as one.
- **Adding an observation.** The edit only subtracts. Deciding which sightings a
  track should have is the bench's work ([../track-edit.md](../track-edit.md)).
- **A dialog for the thresholds.** The menu passes the operation's defaults; the
  wire is where the three are named. A dialog is a thing to add when somebody
  wants one, not before.
