# The bench

Working out whether a photograph really shows the same speck of surface as
another one is not a single decisive act. It means assembling a set of
sightings, measuring each against the rest, taking some and leaving others,
comparing what you have with something else that is also unsettled, and only
then writing the result down. A reconstruction has no room for work in that
state: everything in it is a point that exists. The **bench** is the place
beside it where things that are not settled yet are held. It is a list of
labelled **items**, in the order they were put there, each with an ID that
stays the same when its label changes, and nothing on it is part of the
reconstruction: a save does
not write an item, the point count does not include one, and exactly one step
crosses from an item to a point.

The bench is a value. Putting something on, taking something off and renaming
an item are each a function from one bench to the next, and every item the operation did not touch is the same shared pointer in
both. That is what lets a caller keep a run of benches and go back to any of
them, and it is why the viewer can hold the bench as the second half of a
version and undo the reconstruction and the bench with one gesture.

The first and, today, only kind of item is the **editable track**
([editable-track.md](editable-track.md)). The enum is what makes it the first
rather than the only: a second kind joins by adding a variant, without touching the list, the labels or the steps that work on a track.

Related specs: [editable-track.md](editable-track.md) (the item),
[`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)
(the reconstruction value a commit writes into),
[`../../gui/bench.md`](../../gui/bench.md) (the viewer's half: the version each
step is pushed as, the Scene tree group and the row each writes), and
[`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md)
(the proposal for what remains of it: the searches, the overlays and the wire).

## Rust API

The bench lives in
[bench/mod.rs](../../../crates/sfmtool-core/src/bench/mod.rs), bound as
`sfmtool._sfmtool.bench.Bench`.

```rust
pub struct Bench { /* … */ }

pub enum BenchItem {
    Track(Arc<EditableTrack>),
}

pub enum ItemKind {
    Track,
}

/// Minted by `put` from one counter shared by the whole process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ItemId(u64);

impl ItemId {
    pub fn get(self) -> u64;
}
// Display writes `#<n>`.

pub struct BenchEntry {
    pub id: ItemId,
    pub label: String,
    pub item: BenchItem,
}

pub enum BenchError {
    NoSuchItem(String),
    LabelTaken(String),
    EmptyLabel,
    ControlCharacter(char),
}

pub fn check_label(label: &str) -> Result<(), BenchError>;

impl Bench {
    pub fn new() -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn entries(&self) -> &[BenchEntry];
    pub fn labels(&self) -> impl Iterator<Item = &str>;
    pub fn position(&self, label: &str) -> Option<usize>;
    pub fn get(&self, label: &str) -> Option<&BenchItem>;
    pub fn track(&self, label: &str) -> Option<&Arc<EditableTrack>>;
    pub fn id(&self, label: &str) -> Option<ItemId>;
    pub fn label_of(&self, id: ItemId) -> Option<&str>;

    pub fn mint_label(&self, base: &str) -> String;
    pub fn put(&self, base: &str, item: BenchItem) -> (Bench, String);
    pub fn replace(&self, label: &str, item: BenchItem) -> Result<Bench, BenchError>;
    pub fn discard(&self, label: &str) -> Result<Bench, BenchError>;
    pub fn rename(&self, label: &str, to: &str) -> Result<Bench, BenchError>;

    pub fn delete_image(&self, image: u32) -> (Bench, ImageDeletion);
}

/// What `Bench::delete_image` did.
pub struct ImageDeletion {
    /// Items still on the bench whose observations moved, with where each
    /// observation went (`None` for one in the deleted image).
    pub renumbered: Vec<(ItemId, Vec<Option<usize>>)>,
    /// Observations that were in the deleted image, over every item.
    pub dropped: usize,
    /// Items discarded because every observation was in the deleted image.
    pub discarded: Vec<String>,
}

impl ImageDeletion {
    pub fn changed(&self) -> bool;
    pub fn observation_map(&self, id: ItemId) -> Option<&[Option<usize>]>;
}
```

### Why it is shaped this way

**The bench is only its list of items.** It does not record which item a caller
is working on: that is the caller's to hold. In the viewer it is the focused
item ([`../../gui/bench.md`](../../gui/bench.md) § "The focused item"), held
beside the selection and outside the version; in a script it is the script's
own variable. A second record of it inside the bench could disagree with the
caller's, so there is none. Every step that puts an item on (`create_track`,
`create_cluster`, `split`, `duplicate`, and `put` itself) returns the label it
minted, which is what a caller goes on to work on. Two benches are equal when
their entries are.

**The label is the bench's, not the item's.** Uniqueness is a property of a
particular bench: two benches may hold items minted from the same origin and
neither knows about the other, so a label stored inside the value being worked
on would either be unenforceable or would make the item responsible for
something it cannot see. `BenchEntry` pairs the two, and the item is left as the
thing it is.

**An item has an ID as well as a label.** The label is what a log row, a tab
and a wire call name an item by, but it changes on `rename`, and an undo across
a rename changes it back. A caller that holds on to an item between frames (a
selection, a list of recent items, a cached evaluation) holds its `ItemId`
instead, and reads the label it has now with `label_of`. `put` mints the ID;
`replace` and `rename` keep it; `discard` takes it off with the item. `split`
and `duplicate` go through `put`, so the new item gets a new ID and the original
keeps its own. IDs come from one counter for the whole process rather than one
per bench, so no two items ever share one: a caller that keeps benches as
versions can undo past a put and make a new put that drops the redo versions,
and the new item still does not take the ID an item in those dropped versions
had. The ID is part of a bench's equality, so two benches built separately with
the same labels and items are not equal.

**Every operation returns a new bench.** There is no `&mut`, so a refusal cannot
leave a half-applied change behind, and a caller holding the bench from before
an operation still holds exactly that. The cost is one `Vec` clone of pointer
pairs plus the IDs and label strings, and nothing at all of the items themselves, which
are behind `Arc`.

**`put` mints and `replace` does not.** Putting an item on is where a label is
chosen; installing the result of a step on an item that is already on the bench
must not change it. Splitting them into two
functions is what keeps a hundred verdicts from renaming anything.

**Every refusal names its subject.** The caller is a menu entry or a wire tool
that has to say in one sentence why nothing happened, so `BenchError`'s
`Display` writes that sentence: *"nothing on the bench is called `bull-nose`"*.

### Example

```rust
use sfmtool_core::bench::{create_cluster, Bench, ClusterSeed};

let bench = Bench::new();
let seed = ClusterSeed::from_pixel(4, "IMG_0042", [142.0, 197.5], 7.5);
let (bench, report) = create_cluster(&bench, &seed)?;
assert_eq!(report.label, "IMG_0042@142,198");

let bench = bench.rename(&report.label, "bull-nose")?;
let track = bench.track("bull-nose").expect("just renamed");
```

## Labels

A label is minted from **what the item was made from**, so a log line says what
the thing is without a lookup. The minting is the creating step's, because only
it knows the origin; the bench's part is the collision suffix and the
uniqueness.

| Made from | Label | Minted by |
|---|---|---|
| A committed point | the portable point id, `pt3d_a1b2c3d4_1207` | `create_track` |
| A pixel | the image stem and the rounded pixel, `IMG_0042@142,198` | `ClusterSeed::label` |
| A `.sift` feature | the image stem and the feature index, `IMG_0042#847` | `ClusterSeed::label` |
| A split | the parent's label with `-split` appended | `split` |

A collision takes ` (2)`, ` (3)`, the same disambiguation a scene node's label
takes, so a reader who knows one knows the other. A label is stable until
renamed, and **a discarded item's label is free to be minted again**, because it
names nothing then: the minting reads the bench as it stands and nothing else.

A caller can name the label instead, through `CreateTrackOptions::label` for a
track and `ClusterSeed::label` for a cluster; the name it gives is a base like
any minted one, so a taken label still takes the collision suffix. For a track
put on from a point, the viewer gives the point's portable id unless its caller
named a label: that id names the content a point is a
row of and the version graph that content sits in, and core has neither. What
core does when a caller names none is fall back to what it can see: the first
eight hex digits of the base's own content hash and the point's index there for
a point that is a row of that base, and `point_<index>` for a point an edit
added, which is a row of no content at all.

**A label a caller names is checked, and one that is not a label is refused.**
`check_label` refuses a label that is empty or all whitespace
(`BenchError::EmptyLabel`) and one that holds a control character, such as a
newline, a tab or a NUL (`BenchError::ControlCharacter`, by `char::is_control`).
A label is something a person reads in the Scene tree and an agent types on the
wire, and it is carried into the Action Log and version labels: a newline draws
one item on two rows, and a NUL or a tab cannot be typed back. Every step that
takes a label from its caller checks it in this one function: `rename`, and
`create_track`, `create_cluster` and `find_nearby_tracks` for a caller-named
label, which each refuse with a `Label(BenchError)` of their own error type
before doing any work. `put` stays infallible and checks nothing: a label a
step mints itself, from an image stem, a portable id or a label already on the
bench, is not a caller's, and the suffixes it appends hold no control
character. The viewer's wire tools check a `label` argument with the
same function before a create call starts.

## Deleting an image

An item names its reconstruction's images by index, and deleting an image from
the reconstruction moves every later image down by one. So whatever deletes an
image has to give the bench the same renumbering, or every observation past the
deleted image would name the photograph after the one it was sighted in.
`Bench::delete_image(image)` is the bench as it reads after the delete: each
track goes through `EditableTrack::delete_image`
([editable-track.md](editable-track.md) § "When an image is deleted"), which
drops its observations in `image` and moves those in later images down by one.

**A track left with no observations is discarded.** When every observation a
track had was in the deleted image, nothing it was made of is left in the
reconstruction, and a track with no observations cannot be evaluated, fitted or
committed; keeping it would leave an item on the bench that names nothing and
can only be discarded by hand. So it goes with the image, and the report names
it. A caller that keeps benches as versions, as the viewer does, gets it back
by undoing the delete, which brings back the image too. A track that had no
observations before the delete, such as an empty cluster, observed nothing in
the deleted image and is left alone.

The alternative, refusing the delete while the bench holds an observation in
the image or after it, was not taken: it would make a bench item a lock on the
reconstruction, and a person deleting an image has usually decided that image
is wrong, which is a decision about the tracks that sighted it too.

Every item the delete does not reach (one whose observations are all in
earlier images) is the same `Arc` in both benches, and every item keeps its
`ItemId`. The report carries, per item it renumbered, where each observation
went, so a caller holding observation indexes can follow them; how many
observations were dropped; and the labels it discarded. `changed()` is false
when no item observed the deleted image or any after it.

## Implementation notes

**A rename that changes nothing is not a collision.** Renaming an item to the
label it already holds is allowed, because the label it collides with is itself;
`rename` compares positions rather than strings for exactly that case.

## Python bindings

`sfmtool._sfmtool.bench.Bench`. Construct one with `Bench()`; read it with
`len(bench)`, `bench.labels`, `bench.track(label)`, `bench.id(label)` (the
item's ID as an `int`, or `None` when nothing has that label); change it with
`bench.discard`, `bench.rename` and `bench.replace`, each of which returns the next bench and
leaves the object it was called on as it was. A refusal is a `ValueError`
carrying the core sentence.

The `bench` submodule is deliberately **not** re-exported into the flat
`sfmtool.*` namespace the way the others are: its steps are named for what they
do to a track (`add_observation`, `commit`, `split`), which are words that
surface already spends on other things. Import the module and read the calls
through it.

```python
from sfmtool._sfmtool import bench as bench_module

bench = bench_module.Bench()
bench, track = bench_module.create_cluster(
    bench, 4, "IMG_0042", (142.0, 197.5), radius_px=7.5
)
assert bench.labels == ["IMG_0042@142,198"]
bench = bench.rename("IMG_0042@142,198", "bull-nose")
```

## Testing

[bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs) covers the
labels (each origin's form, the collision suffix, a rename freeing the old
label, `check_label` refusing an empty or whitespace label and one holding a
control character, and `rename`, `create_track` and `create_cluster` refusing
such a label), the item IDs (each put minting a distinct one, on the same bench or
another; `replace` and `rename` keeping it; a put after a discard of the same
label getting a new one; `duplicate` and `split` giving the new item a new ID
and leaving the original's; `id` and `label_of` answering each other and giving
`None` for a label or an ID the bench does not hold), that the bench is only
its list (a put adds exactly one entry at the end, and a put followed by its
discard, or a rename followed by its reverse, gives back an equal bench), and
the sharing: a step on one item leaves every other item the same `Arc`, which is
the property the viewer's per-version budget rests on.
[bench/tests/delete_image.rs](../../../crates/sfmtool-core/src/bench/tests/delete_image.rs)
covers `Bench::delete_image`: a track whose observations were all in the
deleted image discarded and named in the report, an empty track and a track
in earlier images kept as the same `Arc`, every kept item keeping its ID, the
dropped count and the per-item observation map, and a delete no item reaches
changing nothing.
[tests/rust_bindings/bench/test_bench_rust_bindings.py](../../../tests/rust_bindings/bench/test_bench_rust_bindings.py)
covers the same through the bindings, `bench.id` (an `int`, kept by a rename,
`None` for an unknown label), a discard taking off only its item, and the
refusal of a label that names nothing, and that a step leaves the Python object it was called
on unchanged.

## Non-goals

- **A second kind of item.** The bench is shaped so one can be added; only the
  editable track exists, and a pose being re-fitted or a set of images judged
  together is proposed in
  [`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md).
- **A bench across reconstructions.** A bench belongs to one reconstruction, and
  an item names that reconstruction's images.
- **Persisting the bench.** A save writes the reconstruction; a commit is how
  bench work reaches a file.
- **A history.** The bench is a value; the version list that walks a run of them
  is the viewer's, in [`../../gui/bench.md`](../../gui/bench.md).
