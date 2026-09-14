# The bench

Working out whether a photograph really shows the same speck of surface as
another one is not a single decisive act. It means assembling a set of
sightings, measuring each against the rest, taking some and leaving others,
comparing what you have with something else that is also unsettled, and only
then writing the result down. A reconstruction has no room for work in that
state: everything in it is a point that exists. The **bench** is the place
beside it where things that are not settled yet are held. It is a list of
labelled **items**, in the order they were put there, with one **active** item
per kind of item, and nothing on it is part of the reconstruction: a save does
not write an item, the point count does not include one, and exactly one step
crosses from an item to a point.

The bench is a value. Putting something on, taking something off, renaming an
item and changing which one is active are each a function from one bench to the
next, and every item the operation did not touch is the same shared pointer in
both. That is what lets a caller keep a run of benches and go back to any of
them, and it is why the viewer can hold the bench as the second half of a
version and undo the reconstruction and the bench with one gesture.

The first and, today, only kind of item is the **editable track**
([editable-track.md](editable-track.md)). The enum is what makes it the first
rather than the only: a second kind joins by adding a variant and its own active
label, without touching the list, the labels or the steps that work on a track.

Related specs: [editable-track.md](editable-track.md) (the item),
[`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)
(the reconstruction value a commit writes into), and
[`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md)
(the proposal for the viewer's half: the history, the panels, the tree and the
wire).

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

pub struct BenchEntry {
    pub label: String,
    pub item: BenchItem,
}

pub enum BenchError {
    NoSuchItem(String),
    LabelTaken(String),
    EmptyLabel,
}

impl Bench {
    pub fn new() -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn entries(&self) -> &[BenchEntry];
    pub fn labels(&self) -> impl Iterator<Item = &str>;
    pub fn position(&self, label: &str) -> Option<usize>;
    pub fn get(&self, label: &str) -> Option<&BenchItem>;
    pub fn track(&self, label: &str) -> Option<&Arc<EditableTrack>>;
    pub fn active_label(&self, kind: ItemKind) -> Option<&str>;
    pub fn active_track(&self) -> Option<&Arc<EditableTrack>>;

    pub fn mint_label(&self, base: &str) -> String;
    pub fn put(&self, base: &str, item: BenchItem) -> (Bench, String);
    pub fn replace(&self, label: &str, item: BenchItem) -> Result<Bench, BenchError>;
    pub fn activate(&self, label: &str) -> Result<Bench, BenchError>;
    pub fn discard(&self, label: &str) -> Result<Bench, BenchError>;
    pub fn rename(&self, label: &str, to: &str) -> Result<Bench, BenchError>;
}
```

### Why it is shaped this way

**The label is the bench's, not the item's.** Uniqueness is a property of a
particular bench: two benches may hold items minted from the same origin and
neither knows about the other, so a label stored inside the value being worked
on would either be unenforceable or would make the item responsible for
something it cannot see. `BenchEntry` pairs the two, and the item is left as the
thing it is.

**Every operation returns a new bench.** There is no `&mut`, so a refusal cannot
leave a half-applied change behind, and a caller holding the bench from before
an operation still holds exactly that. The cost is one `Vec` clone of pointer
pairs plus the label strings, and nothing at all of the items themselves, which
are behind `Arc`.

**`put` mints and `replace` does not.** Putting an item on is where a label is
chosen and where activation moves; installing the result of a step on an item
that is already on the bench must change neither. Splitting them into two
functions is what keeps a hundred verdicts from renaming anything.

**The active item is per kind and lives in a map.** Each kind of item is edited
in its own panel and a gesture that names no target means "the active one of the
kind this panel edits". A single "active item" would make one panel's click
change what another panel is showing.

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
assert_eq!(bench.active_label(ItemKind::Track), Some("bull-nose"));
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

The label is the caller's to give, through `CreateTrackOptions::label`, and the
viewer gives the point's portable id: that id names the content a point is a
row of and the version graph that content sits in, and core has neither. What
core does when a caller names none is fall back to what it can see: the first
eight hex digits of the base's own content hash and the point's index there for
a point that is a row of that base, and `point_<index>` for a point an edit
added, which is a row of no content at all.

## Implementation notes

**A discard moves the activation to a neighbour of the same kind.** The item
before it in the list, or the one after it when the discarded item was the
first, and no active item at all when the kind has nothing left. It is a
neighbour *of the same kind* rather than the adjacent entry, because the list
interleaves kinds and each kind's activation is its own.

**A rename that changes nothing is not a collision.** Renaming an item to the
label it already holds is allowed, because the label it collides with is itself;
`rename` compares positions rather than strings for exactly that case.

**The active map is a `BTreeMap`, not a `HashMap`.** A bench is compared for
equality by the viewer's history and dumped by the wire, and a deterministic
iteration order is worth more here than the lookup speed of a map with one entry
in it.

## Python bindings

`sfmtool._sfmtool.bench.Bench`. Construct one with `Bench()`; read it with
`len(bench)`, `bench.labels`, `bench.track(label)`, `bench.active_label(kind)`
and `bench.active_track`; change it with `bench.activate`, `bench.discard`,
`bench.rename` and `bench.replace`, each of which returns the next bench and
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
label), the activation (a discard moving it to the neighbour, an empty bench
having none), and the sharing: a step on one item leaves every other item the
same `Arc`, which is the property the viewer's per-version budget rests on.
[tests/rust_bindings/test_bench_rust_bindings.py](../../../tests/rust_bindings/test_bench_rust_bindings.py)
covers the same through the bindings, and that a step leaves the Python object
it was called on unchanged.

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
  is the viewer's, and is proposed in the draft above.
