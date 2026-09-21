# Glossary

The words this project has settled on, what each one means, and what it was
chosen over. Read it before naming a function, a wire tool, a field or a
user-facing string, and add to it whenever a naming question is settled by
argument rather than by accident.

It exists because the cheap way to settle a naming split, counting both
spellings and converging on the bigger pile, gets the answer wrong often enough
to be dangerous. A majority cannot tell a word that won on merit from a word
that won by being written first, and it cannot tell a migration in progress from
drift. Both failures have happened here, and both are recorded below.

## How to read an entry

**A glossary entry outranks a count.** Where this file names a preferred word,
that is the answer regardless of which spelling is currently more common in the
tree. The `audit-hygiene` skill's convention tallies are for finding splits this
file has not yet ruled on; they do not overturn one it has.

**Scope is part of the entry.** A word can be right in one layer and wrong in
the next, and an entry that does not say where it applies will be
over-applied. Each entry below names its scope.

**Direction of travel is part of the entry.** Some entries describe a convention
being migrated to, where the old form is still the majority. New prose follows
the entry, and a file converges as its paragraphs are rewritten for other
reasons. Nobody is expected to sweep the tree, and a one-line edit to a long
file is not an obligation to convert the rest of it.

## The bench

The vocabulary of `crates/sfmtool-core/src/bench/`, the viewer's bench layers
and the wire's bench tools.

| Word | Means | Not | Why |
|------|-------|-----|-----|
| **patch** | the oriented square standing in the world, carrying the content the photographs see there | `surfel`, `frame` | the sense the PatchMatch family uses. The word already holds the geometry, so no second noun is needed for it |
| **placement** | the geometry alone: centre, axes, half-extent | `frame` | elsewhere in this tree `frame` means a video or rig frame, and a UI frame, so it cannot also mean this |
| **track** | the set of observations across images that are views of one point | | |
| **sighting** | one image's view of the patch, interchangeable with **observation** | | |
| **keypoint** | where one photograph sees the patch's content | | not the projection of the centre. The gap between the two is that sighting's in-plane offset |
| **bench** | where an item is held and judged before it is committed | | |

**The four gestures are named for the axis they act on**, so that no two are
synonyms: **translate** moves the centre, **resize** changes the half-length,
**spin** turns the square about its normal, and **tilt** turns the normal
itself. The pairs this replaced were unusable. `translate` and `offset` are
English synonyms that named perpendicular motions, and `rotate` is the generic
word for turning, which cannot stand beside a specific one.

**A wire tool is named for the part it acts on**, not for the item it is
addressed through. `translate_bench_patch` acts on the patch,
`sight_bench_observation` on one sighting, `commit_bench_track` on the track,
and `duplicate_bench_item` on either kind. The name that made this rule
necessary was `move_bench_track_observation`, which moves an observation and not
a track.

**Each word is said once along the path**, the enclosing namespace carrying what
it already states. A core step sits in `sfmtool_core::bench` and takes an
`&EditableTrack`, so it is `tilt_patch`. A viewer edit sits in `PatchEdit`, so
it is `PatchEdit::Tilt`. Only the wire, whose namespace is flat, spells all
three, as `tilt_bench_patch`. A name like `bench_track_frame` is three words
where the context has already supplied two.

## Words with a boundary

**patch** (the bench) and **surfel** (the scene renderer). Both name an
`OrientedPatch`, and the split is deliberate rather than settled by a count: the
renderer is doing surfel splatting, which is what that word means in graphics,
while the bench is editing one patch in the PatchMatch sense. `surfel` is
correct in `scene_renderer/`, `point_track_detail/`, `state*` and the specs that
describe rendering. Inside the bench it is wrong. *If this boundary is ever
removed it should be removed deliberately and in one pass, not eroded from
either side.*

## Contrast pairs that are not synonyms

Two words that look interchangeable and are not. Using either for the other
loses a distinction the code depends on.

| Pair | The distinction |
|------|-----------------|
| **retriangulate** / **direction re-estimation** | retriangulation settles a finite point from its observations. A direction re-estimation settles the **bearing** of a point at infinity, which has no depth to solve for. The specs name both in one breath, so neither word is available as a loose synonym for the other |
| **SIFT index** / descriptor index | the user-facing name for the `.kdf` structure is *SIFT index*. *Descriptor index* is not a second name for it |
| **spin** / **tilt** | a spin turns the square about its own normal and moves no sighting. A tilt turns the normal itself and rebuilds every sighting on the turned axes. In axis-angle form they look like one operation about two axes; they are not, because they do opposite things to the keypoints |
| **translate** / **offset** | not a pair but a collision. `offset` names a sighting's in-plane displacement from the patch's middle, `(a_i, b_i)`. It does **not** name a motion of the patch, which is `translate` |

## Prose

**Avoid the em-dash, and do not swap it for ` -- `.** The em-dash is a
construction rather than a character: a parenthetical break dropped into the
middle of a sentence. Replacing the character keeps the habit and changes only
its spelling, which is how prose ends up with a dashed aside every second
paragraph in a file that never once typed an em-dash.

What the sentence wants is usually something else, and usually something
different from the last time. A comma carries a light aside. A colon introduces
the thing that explains what came before. A semicolon balances two clauses that
could each stand alone. A full stop is right when the aside was really a second
thought, and parentheses when it is genuinely an aside. Often the break was not
earning its place and the sentence reads better rewritten without it. Vary these
rather than settling on one, because a single substitution repeated is the tic
again under a new name. A real dash remains available where a sentence genuinely
calls for one; it should be occasional.

*Direction of travel:* em-dash is still the majority across `specs/` by roughly
seven to one. A convention tally on this reports the wrong majority, which is
the point of recording it here. Where an em-dash is the separator in a list
whose every sibling item uses one, match the list; the entry is about prose, not
about a file's formatting furniture.

Specs are **present tense** and describe what the code *is*. A change proposal
lives in [drafts/](drafts/README.md) and is converted before filing. That rule
and the rest of the spec contract are in [README.md](README.md), not here.

## Adding an entry

Add one when a naming question is settled by argument. Record the word, what it
means, what it was chosen over, and above all **why**, since the reason is the
part that lets the next person decide a case this table does not list. Where the
answer differs by layer, say so. Where the tree has not caught up yet, say that
too, so an audit does not read the lag as the answer.
