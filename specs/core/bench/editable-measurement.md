# The editable measurement

A reconstruction can store measurements: real-world quantities tied to places
in its photographs, such as a length taken with a tape measure between two
pixels, each in one image. Placing those pixels well is work that takes
several steps: a person moves an endpoint, compares the result with the other
measurements, and only then wants it written into the file. The **editable
measurement** is a stored measurement taken out of the reconstruction onto the
bench to be worked on that way, or a new one started from one pixel, and the
**commit** is the one step that writes it back. It holds only what the file
stores of a measurement, so whatever is derived from it, such as the 3D
position an endpoint lands on, is read again against whichever version of the
reconstruction the caller stands at.

The bench and its other kind of item, the editable track, are
[bench.md](bench.md) and [editable-track.md](editable-track.md). What the file
stores is [`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)
§ 10 "Measurements", and how an endpoint's pixel becomes a 3D position is
[`../reconstruction/measurement-endpoint.md`](../reconstruction/measurement-endpoint.md).
The viewer's half (the versions, Measurement View's edited mode and the wire)
is [`../../gui/measurement-view.md`](../../gui/measurement-view.md) and
[`../../gui/bench.md`](../../gui/bench.md). Three steps edit an item: setting
its label and setting its stated distance (§ "The label and the stated
distance"), and setting an endpoint's image and pixel (§ "Setting an
endpoint"); `start_measurement` starts a new item from one pixel (§ "Starting a
measurement").

## Rust API

The item and its steps live in
[bench/measurement.rs](../../../crates/sfmtool-core/src/bench/measurement.rs),
re-exported from `sfmtool_core::bench`, and bound as
`sfmtool.bench.EditableMeasurement`, `sfmtool.bench.put_measurement`,
`sfmtool.bench.start_measurement`,
`sfmtool.bench.set_measurement_label`, `sfmtool.bench.set_stated_distance`,
`sfmtool.bench.set_measurement_endpoint`,
`sfmtool.bench.parse_stated_distance` and `sfmtool.bench.commit_measurement`.

```rust
pub enum MeasurementKind {
    Distance,
}

impl MeasurementKind {
    pub const ALL: [MeasurementKind; 1];
    pub fn title(self) -> &'static str;               // "Distance"
    pub fn name(self) -> &'static str;                // "distance", the file's subdirectory
    pub fn from_name(name: &str) -> Option<MeasurementKind>;
    pub fn count(self, measurements: &MeasurementsData) -> usize;
}

/// `distance #2`.
pub fn measurement_title(kind: MeasurementKind, index: u32) -> String;

pub struct EditableMeasurement {
    pub origin: Option<MeasurementOrigin>,
    pub label: String,
    pub body: EditableMeasurementBody,
}

pub enum EditableMeasurementBody {
    Distance(EditableDistance),
}

pub struct EditableDistance {
    /// Endpoints A and B; `None` for one not placed.
    pub endpoints: [Option<MeasurementEndpoint>; 2],
    pub distance: Option<(f64, LengthUnit)>,
}

pub struct MeasurementEndpoint {
    pub image: u32,
    pub pixel: [f64; 2],
}

pub struct MeasurementOrigin {
    /// The version the row was read out of, as the caller numbers versions.
    pub version: u64,
    pub kind: MeasurementKind,
    pub index: u32,
}

impl EditableMeasurement {
    pub fn from_stored(
        edited: &EditedReconstruction,
        kind: MeasurementKind,
        index: u32,
        version: u64,
    ) -> Result<Self, NoSuchMeasurement>;
    pub fn kind(&self) -> MeasurementKind;
    pub fn as_distance(&self) -> Option<&EditableDistance>;
    pub fn with_origin(&self, version: u64, index: u32) -> Self;
    pub fn delete_image(&self, image: u32) -> Option<(EditableMeasurement, usize)>;
    pub fn placed_endpoints(&self) -> usize;
}

pub fn put_measurement(
    bench: &Bench,
    edited: &EditedReconstruction,
    kind: MeasurementKind,
    index: u32,
    version: u64,
) -> Result<(Bench, String), NoSuchMeasurement>;

/// A new distance with endpoint A at `endpoint` and nothing else, labelled
/// `distance from IMG_0042@142,198`.
pub fn start_measurement(
    bench: &Bench,
    edited: &EditedReconstruction,
    endpoint: MeasurementEndpoint,
) -> Result<(Bench, String), EndpointPlacementRefusal>;

pub fn set_measurement_label(item: &EditableMeasurement, label: &str) -> EditableMeasurement;

pub fn set_stated_distance(
    item: &EditableMeasurement,
    distance: f64,
    unit: LengthUnit,
) -> Result<EditableMeasurement, StatedDistanceRefusal>;

/// `12in`, `30.48 cm`, or a bare number in `bare_unit`.
pub fn parse_stated_distance(
    text: &str,
    bare_unit: LengthUnit,
) -> Result<(f64, LengthUnit), StatedDistanceRefusal>;

pub enum StatedDistanceRefusal {
    Unreadable { text: String },
    UnknownUnit { unit: String },
    NotPositive { distance: f64 },
}

/// `side` 0 for A, 1 for B.
pub fn set_measurement_endpoint(
    edited: &EditedReconstruction,
    item: &EditableMeasurement,
    side: usize,
    endpoint: MeasurementEndpoint,
) -> Result<EditableMeasurement, EndpointPlacementRefusal>;

pub enum EndpointPlacementRefusal {
    NoSuchEndpoint { side: usize },
    /// The commit's `NoSuchImage` or `PixelOutsideImage`.
    Outside(MeasurementCommitRefusal),
}

pub fn commit_measurement(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<MeasurementCommit, MeasurementCommitRefusal>;

pub fn commit_measurement_preconditions(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<(), MeasurementCommitRefusal>;

pub struct MeasurementCommit {
    pub value: EditedReconstruction,
    pub kind: MeasurementKind,
    pub index: u32,      // the row written, or the row that already held it
    pub replaced: bool,  // a row replaced rather than appended
    pub wrote: bool,     // false where the row already held exactly this
}

impl MeasurementCommit {
    pub fn label(&self, node: &str) -> String;
}

pub enum MeasurementCommitRefusal {
    EndpointNotPlaced { side: usize },
    NoStatedDistance,
    DistanceNotPositive { distance: f64 },
    NoSuchImage { side: usize, image: u32, image_count: usize },
    PixelOutsideImage { side: usize, image: u32, pixel: [f64; 2], width: u32, height: u32 },
    NoSuchRow(NoSuchMeasurement),
}

pub fn delete_measurement(
    base: &EditedReconstruction,
    kind: MeasurementKind,
    index: u32,
) -> Result<(EditedReconstruction, Vec<Option<u32>>), NoSuchMeasurement>;

pub struct NoSuchMeasurement {
    pub kind: MeasurementKind,
    pub index: u32,
    pub count: usize,
}
```

### Why it is shaped this way

**The item holds only what the file stores.** Each endpoint's image and pixel,
the stated distance and its unit, and the label. An endpoint's point, texel and
3D position are not held, because they follow from the image and pixel through
the resolver and would go stale with any edit of the reconstruction under the
item. A caller reads them when it needs them, with
`resolve_distance_endpoints` for both endpoints or
`resolve_measurement_endpoint` for one
([`../reconstruction/measurement-endpoint.md`](../reconstruction/measurement-endpoint.md)).

**An endpoint can be absent.** A stored measurement always has both, but a
new measurement starts with endpoint A alone (§ "Starting a measurement"), and
deleting the image an endpoint was in clears it (§ "When an image is
deleted"). An item with one endpoint is still worth keeping: the person who
placed it can place the other.

**The origin is a row at a version.** A measurement has no portable id, so its
row is the only name it has, and rows are renumbered by the edits that drop
them. The origin records the version it was read at in the caller's own
numbering, which core neither mints nor reads, as a track's `Origin` does; the
caller follows it to the version it stands at and hands the commit the row it
lands on. The commit does not follow anything itself, because core has no
history.

**The label is the measurement's, not the item's.** `label` is the text the
file stores for the measurement; it may be empty and need not be unique. The
item's name on the bench is the bench's, minted by `put_measurement` from the
kind and the row: `distance #2`, with the bench's collision suffix where
another item holds that ([bench.md](bench.md) § "Labels"). One started from a
pixel is minted by `start_measurement` from where it starts.

**The commit takes the row, not the origin.** `origin_row` is the row the
caller resolved the origin to, or `None` where it resolved to none, which
appends. A row past the rows is refused rather than appended, since it says the
caller's resolution went wrong.

**The new value shares the base.** The commit and the delete change the
measurements and nothing else, so each returns
`base.with_measurements(...)`: the same base `Arc`, the same point overlay, and
measurements of the value's own
([`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)
§ "Measurements of the value's own"). No point, track or patch bitmap is
copied, and every point index means in the result what it meant before.

**The preconditions are their own function.** A caller greys a commit button
with `commit_measurement_preconditions`, asked of the item the button would
commit, and the commit asks the same function first, so the button and the
step cannot disagree.

### Example

```rust
use sfmtool_core::bench::{commit_measurement, put_measurement, Bench, MeasurementKind};

let (bench, label) = put_measurement(&Bench::new(), &edited, MeasurementKind::Distance, 2, 7)?;
assert_eq!(label, "distance #2");
let item = bench.measurement(&label).expect("just put on");
// The caller follows the origin to the version it stands at; here it is the same.
let commit = commit_measurement(&edited, item, Some(2))?;
assert!(!commit.wrote); // nothing changed, so nothing is written
```

## Starting a measurement

`start_measurement` puts a new distance measurement on the bench with endpoint
A at an image and a pixel and nothing else: no origin, so its commit appends a
row; no endpoint B and no stated distance, which `set_measurement_endpoint`
and `set_stated_distance` give it; and an empty label of its own. Its bench
label is minted from where it starts, `distance from ` and the form a cluster
seeded at the pixel is labelled with (`ClusterSeed::label`): the image's stem
and the pixel rounded to whole pixels, `distance from IMG_0042@142,198`, with
the bench's collision suffix, ` (2)`, where another item holds that
([bench.md](bench.md) § "Labels"). The endpoint is asked what
`set_measurement_endpoint` asks of one, through the same check, so an image
past the image set and a pixel outside the image are refused in the commit's
sentence. SfM Explorer's *New Measurement from Here* and its wire tool
`create_bench_measurement` make it.

```rust
use sfmtool_core::bench::{start_measurement, Bench, MeasurementEndpoint};

let at = MeasurementEndpoint { image: 41, pixel: [141.6, 198.4] };
let (bench, label) = start_measurement(&Bench::new(), &edited, at)?;
assert_eq!(label, "distance from IMG_0042@142,198"); // image 41 is IMG_0042.jpg
```

## The label and the stated distance

`set_measurement_label` and `set_stated_distance` are the steps that edit an
item's label and stated distance, each from one `EditableMeasurement` to the
next, with nothing else changed. A caller that keeps versions replaces the item's `Arc` under the same
`ItemId` with the result. Any text is a label, the empty one included. A stated
distance is refused where it is not finite and strictly positive as the
`float32` the file will hold, in the commit's sentence: *"the stated distance
is -2, which must be finite and strictly positive"*.

**A typed distance is read by `parse_stated_distance`.** It is a number with an
optional unit suffix, `12in`, `30.48cm` or `30.48 cm`: an optional sign, digits
with an optional decimal point, and an optional exponent, then optionally a
space and one of the format's unit names (`mm`, `cm`, `m`, `in`, `ft`).
Surrounding space is ignored. A bare number is in `bare_unit`, which the caller
chooses: Measurement View passes the reconstruction's `world_space_unit`, or
metres where it has none, and `sfm xform --scale-by-measurements` passes the
unit its YAML file names
([`../../cli/reconstruction/xform/scale-by-measurements-command.md`](../../cli/reconstruction/xform/scale-by-measurements-command.md)),
so the panel and the command read a distance the same way. Text that is not
such a number (`inf` and `nan` among it), a suffix that names no unit, and a
value that is not finite and strictly positive are each refused with a
sentence saying which.

## Setting an endpoint

`set_measurement_endpoint` is the step that moves an endpoint to another image
and pixel, or places one that is not placed, from one `EditableMeasurement`
to the next with nothing else changed. It sets the image and the pixel and
nothing more: the point the endpoint lands on, its texel and its 3D position
follow from them through the resolver, so there is nothing else to set, and a
pixel no square holds is still an endpoint, as it is in the file. SfM
Explorer's *Switch image…* popup makes it with a pixel from `endpoint_views`
([`../reconstruction/measurement-endpoint.md`](../reconstruction/measurement-endpoint.md)
§ "The views of an endpoint"), and its endpoint fields, its two menus' *Set
Endpoint … Here* and its wire tool with any pixel.

It asks of the endpoint what the commit asks of each one, through the same
check, so a step and a commit cannot disagree: an image past the image set,
and a pixel that is not inside `[0, width) × [0, height)` of that image's
camera as the `float32` the file will hold, are each refused in the commit's
sentence (*"endpoint A's pixel (640, 3) is outside image 0, which is [0, 640)
x [0, 480)"*), wrapped as `EndpointPlacementRefusal::Outside`. A side other
than 0 and 1 is `NoSuchEndpoint`.

## The commit

The commit checks, in order, that both endpoints are placed, that each names an
image of the reconstruction and a pixel inside `[0, width) × [0, height)` of
that image's camera (as the `float32` the file will hold), that a stated
distance is given and is finite and strictly positive as a `float32`, and that
`origin_row`, where given, is a row of the item's kind. Each refusal's
`Display` is the sentence a person reads: *"endpoint B is not placed, and a
distance needs both endpoints"*.

An endpoint that the resolver refuses does not stop a commit: the pixel is what
the file stores, and it stores it whether or not a square of the reconstruction
holds it.

**Replacing a row that already holds the item writes nothing.** The row is
compared with the item as the file stores it, `float32` pixels and distance,
the unit code and the label; where they are equal, the commit reports
`wrote: false` with the value unchanged, and the caller pushes no version.
Otherwise the rows are rebuilt with the item in place of row `origin_row`, or
appended after the last, and the kind's columns are rebuilt from them.

`MeasurementCommit::label` is the sentence a log records: `Committed distance
#2 in dino` for a replaced row, `Added distance #4 in dino` for an appended
one, and `Committed distance #2 in dino: no effect, the row already holds it`
where nothing was written.

## The direct delete

`delete_measurement` removes one stored row. It is not a bench step: it edits
the reconstruction directly, as deleting a point does, and returns with the
value the row map of the kind, entry `i` the new row of row `i` and `None` for
the deleted one. The rows after it move down by one. Deleting the last row of a
kind leaves the kind absent, as a file with none of it stores it. A caller that
keeps versions follows the selected measurement and every item's origin
through that map; an item whose origin was the deleted row is left with none,
so its commit appends the measurement back.

## When an image is deleted

Deleting an image from the reconstruction moves every later image down by one,
and an endpoint names its image by index. `EditableMeasurement::delete_image`
is the item as it reads after the delete: an endpoint in the deleted image is
cleared, and one in a later image moves down by one. It returns `None` when no
endpoint is in the deleted image or after it, so the caller keeps the same
`Arc`, and otherwise the item and how many endpoints it cleared. The origin is
left alone: it names a row at a version, and the caller follows it through the
delete's own row map. `Bench::delete_image` discards an item left with no
endpoint ([bench.md](bench.md) § "Deleting an image").

## Python bindings

`sfmtool.bench.EditableMeasurement` has the read-only properties `kind`
(`"distance"`), `label`, `origin` (`{"version", "kind", "index"}` or `None`),
`endpoints` (two entries, each `(image, (x, y))` or `None`) and `distance`
(`(value, unit_name)` or `None`), and `with_origin(version, index)`.
`Bench.measurement(label)` gives the measurement called `label`, or `None`;
`Bench.kind(label)` gives `"track"` or `"measurement"`; `Bench.replace` takes
either kind of item.

```python
from sfmtool import bench as bench_module

bench, item = bench_module.put_measurement(
    bench_module.Bench(), edited, "distance", 2, version=7
)
assert bench.labels == ["distance #2"]
edited, report = bench_module.commit_measurement(edited, item, 2, node="dino")
assert report["wrote"] is False
```

`put_measurement(bench, edited, kind, index, *, version=0)` and
`start_measurement(bench, edited, image, xy)` return
`(Bench, EditableMeasurement)`. `set_measurement_label(measurement, label)`,
`set_stated_distance(measurement, distance, unit)`, the unit a name (`"cm"`),
and `set_measurement_endpoint(edited, measurement, side, image, xy)`, `side`
0 for A and 1 for B, each return the next `EditableMeasurement` and leave
their input as it was. `parse_stated_distance(text, bare_unit="m")` returns
`(value, unit_name)`. `commit_measurement(edited, measurement,
origin_row=None, *, node="the reconstruction")` returns
`(EditedReconstruction, report)`, the report carrying `index`, `replaced`,
`wrote` and `label`. A refusal is a `ValueError` carrying the core sentence.

## Testing

[bench/tests/measurement.rs](../../../crates/sfmtool-core/src/bench/tests/measurement.rs)
covers putting a stored measurement on the bench (its endpoints, distance,
unit and label copied as stored, its origin, its label and the collision
suffix, a row past the rows refused with the count), starting one from a pixel
(endpoint A alone, no origin, distance or label, its label and the collision
suffix, its commit refused until B and the distance are given and then
appending a row, an image past the image set and a pixel outside the image
refused in the commit's sentence), the kind's names, deleting
an image (an endpoint cleared, one moved down, an item left with none
discarded, an item the delete does not reach keeping its `Arc` and every item
its ID), the commit (a replaced row and the rows around it, an appended row, a
first row of a kind, the no-effect case, each refusal with the preconditions
giving the same sentence, a re-seated item committing to no effect), the
pointer equality of the shared base and point column, the direct delete and
its row map, the last row leaving the kind absent, a value with
measurements of its own materializing them, the three steps changing their
field and nothing else (an endpoint placed where none was, and a moved one
written by the commit), a stated distance that is not finite and positive
refused in the commit's sentence, an endpoint outside the images refused in
the commit's sentence and a side past B refused, and the parser: each unit
suffix with and without a space, a bare number in the unit given, the number's
other spellings, and each refusal.
[tests/rust_bindings/bench/test_bench_measurement_rust_bindings.py](../../../tests/rust_bindings/bench/test_bench_measurement_rust_bindings.py)
covers the same through the bindings.

## Non-goals

- **An endpoint's 3D position as part of the item.** It always follows from
  the image and the pixel.
- **A kind other than distance.** `MeasurementKind` has one variant, as the
  file has one kind.
