# Measurement Scale

A structure-from-motion solve has no real-world size: its coordinates are in
**scene units**, of whatever length the solve happened to give them. When the
reconstruction stores distance measurements, each a real length stated between
two places seen in its photographs, every measurement says how long a scene
unit is: its stated length over the distance its two ends lie apart in the
reconstruction. This operation turns those into one uniform scale factor to a
chosen physical unit (millimetres, centimetres, metres, inches or feet), the
median of the per-measurement factors, and reports how well the measurements
agree: their spread, each one's deviation from the median, and the
measurements that give no factor and why. Multiplying the reconstruction's
coordinates by the factor puts it in that unit.

The measurements are the `.sfmr` file's own
([formats/sfmr-file-format.md](../../formats/sfmr-file-format.md#distance-measurements-measurementsdistance)
§ 10 "Measurements", the distance kind). Each end of one is a pixel in an
image, and [measurement-endpoint.md](measurement-endpoint.md) is how that pixel
is read as a 3D position. The factor is what a person applies to give the
reconstruction a unit from what it stores; the statistics are what they check
first, since two measurements of the same rigid object that disagree point at
a misplaced endpoint or a distorted reconstruction.
`sfm xform --scale-to-median-measurement`
([cli/reconstruction/xform/scale-to-median-measurement-command.md](../../cli/reconstruction/xform/scale-to-median-measurement-command.md))
calls it through the Python binding, prints the factor, the statistics, each
measurement's factor and the exclusions, and applies the factor as a pipeline
step. SfM Explorer calls it for `Scale to Median
Measurement...` on a reconstruction's Scene tree row, whose dialog shows the
factor, the statistics, each measurement's factor and the exclusions, and whose
edit applies the factor
([gui/edits/scale-to-median-measurement.md](../../gui/edits/scale-to-median-measurement.md)).

## Rust API

In [`reconstruction::measurement_scale`](../../../crates/sfmtool-core/src/reconstruction/measurement_scale.rs),
with its tests in
[`measurement_scale/tests.rs`](../../../crates/sfmtool-core/src/reconstruction/measurement_scale/tests.rs).
Python reaches it as `SfmrReconstruction.distance_measurement_scale`
(§ "Python binding").

```rust
/// The bar `sfm xform --scale-to-median-measurement` warns at.
pub const SCALE_DISAGREEMENT_PERCENT: f64 = 5.0;

pub fn distance_measurement_scale(
    measurements: &MeasurementsData,
    readings: &[ResolvedDistanceMeasurement], // one per stored measurement
    unit: LengthUnit,                         // the unit to scale to
) -> Result<DistanceMeasurementScale, DistanceMeasurementScaleError>;

pub struct DistanceMeasurementScale {
    pub unit: LengthUnit,
    pub factors: Vec<MeasurementScaleFactor>, // the usable ones, stored order
    pub excluded: Vec<ExcludedMeasurement>,   // the rest, stored order
    pub statistics: ScaleFactorStatistics,
}
impl DistanceMeasurementScale { pub fn factor(&self) -> f64; } // the median

pub struct MeasurementScaleFactor {
    pub index: usize,           // in stored order
    pub stated: f64,            // in `unit`
    pub scene_distance: f64,    // in scene units
    pub factor: f64,            // stated / scene_distance
    pub deviation_percent: f64, // 100 · (factor / median − 1), signed
}

pub struct ScaleFactorStatistics {
    pub count: usize,                     // ≥ 1
    pub median: f64,
    pub mean: f64,
    pub sample_std: Option<f64>,          // None for one measurement
    pub sample_std_percent: Option<f64>,  // of the median
    pub median_absolute_deviation: f64,   // median of |factor − median|
    pub min: f64,
    pub max: f64,
    pub max_abs_deviation_percent: f64,
    pub disagrees: bool,                  // max_abs_deviation_percent > 5
}

pub struct ExcludedMeasurement { pub index: usize, pub reason: MeasurementScaleExclusion }

pub enum MeasurementScaleExclusion {
    EndpointRefused { endpoint: usize, refusal: EndpointRefusal }, // 0 is A, 1 is B
    EndpointAtInfinity { endpoint: usize },
    UnusableSceneDistance { scene_distance: f64 }, // zero or not finite
    NoSceneDistance,                               // only in a reading built by hand
    UnusableStatedDistance,                        // only in a value built by hand
    UnusableFactor { factor: f64 },                // the ratio overflows or underflows
}

pub enum DistanceMeasurementScaleError {
    NoMeasurements,
    ReadingCountMismatch { measurements: usize, readings: usize },
    NoUsableMeasurement { excluded: Vec<ExcludedMeasurement> },
}
```

**Why these arguments.** The operation takes the readings rather than the
reconstruction. Resolving the endpoints is the expensive part, and SfM Explorer
already resolves every stored measurement once per version and keeps the
result on the scene node (`SceneNode::measurement_readings`,
[gui/measurement-view.md](../../gui/measurement-view.md) § "The readings"),
where its Scale to Median Measurement dialog reads them afresh every frame;
taking `&[ResolvedDistanceMeasurement]` lets a caller holding them reuse them,
and one that does not makes them with one call to
`resolve_distance_measurements`. It takes `MeasurementsData`, the whole
section, because that is what `EditedReconstruction::measurements` gives: a
version's measurements are its own where an edit set them, so a caller reads
them from the edited value rather than from the base. The readings must be the
same version's, paired with the measurements by index; a count that differs
from the stored measurements' is refused as `ReadingCountMismatch` rather than
paired up short.

**Why it reads no `world_space_unit`.** A scene distance is in the file's own
coordinates whether or not the file declares a unit, as every reading in
Measurement View is. A factor `f` with `stated` in the target unit and
`scene_distance` in file coordinates makes `f ×` coordinates come out in the
target unit, whatever unit they were in before; the unit the file declared
before is replaced, not composed with. For a file that declares a unit, this
factor is Measurement View's `implied_scale` (`stated / reconstructed`, both in
the measurement's unit) times the number of target units in one of the file's
units, and for a file with no unit it is that view's `unit_per_scene_unit`
converted to the target unit.

**Why exclusions are values and not errors.** A person scaling a
reconstruction needs to see which measurements were left out and why, beside
the ones that counted, so each carries its stored index and its reason; a
refused endpoint keeps the resolver's `EndpointRefusal`, whose sentence names
the remedy. A refusal is reported before an endpoint at infinity, `A` before
`B`, as Measurement View reports them, since the refusal is the one a person can
act on. Only when nothing is usable is the call an error, and that error still
carries every exclusion.

**Why no histogram.** The factors are in the result, and the bins a histogram
draws are a choice of the drawing: a plot sized to a panel chooses its own
count, as SfM Explorer's dialog does over each measurement's deviation, and
the CLI prints each measurement's deviation on its own line and draws no
histogram. Binning is a few lines over `factors`, `min` and `max`, so it stays
with whoever draws it.

Example:

```rust
use std::sync::Arc;
use sfmtool_core::reconstruction::measurement_endpoint::{
    resolve_distance_measurements, MeasurementEndpointOptions,
};
use sfmtool_core::reconstruction::measurement_scale::distance_measurement_scale;
use sfmtool_core::reconstruction::EditedReconstruction;
use sfmtool_core::{LengthUnit, RotQuaternion, Se3Transform};

let edited = EditedReconstruction::new(Arc::new(recon));
let readings = resolve_distance_measurements(&edited, &MeasurementEndpointOptions::default());
let scale = distance_measurement_scale(edited.measurements(), &readings, LengthUnit::Millimetre)?;
if scale.statistics.disagrees {
    for f in &scale.factors {
        println!("measurement {}: {:+.1}%", f.index, f.deviation_percent);
    }
}
let (recon, _) = edited.materialize();
let mut scaled = recon.apply_se3_transform(&Se3Transform::new(
    RotQuaternion::identity(), nalgebra::Vector3::zeros(), scale.factor(),
));
scaled.metadata.world_space_unit = Some(LengthUnit::Millimetre.name().to_string());
```

### Python binding

```python
scale = recon.distance_measurement_scale("mm")
scale["factor"]                  # the median
scale["statistics"]["disagrees"]  # past disagreement_percent, 5
for f in scale["factors"]:       # index, label, stated, scene_distance, factor, deviation_percent
    ...
for e in scale["excluded"]:      # index, label, reason (a sentence)
    ...
```

In [`reconstruction/measurement_scale.rs`](../../../crates/sfmtool-py/src/reconstruction/measurement_scale.rs),
a method of `SfmrReconstruction` beside `resolve_distance_measurements`, since
it reads the reconstruction's own measurements. It resolves their endpoints
with the default options, as that method does, and calls this operation on
them, so a Python caller holds no readings to pair with the measurements and
cannot meet `ReadingCountMismatch`. It returns a dict of the result's fields:
`unit`, `factor`, `factors`, `excluded` with each reason as its `Display`
sentence, each factor and exclusion with the measurement's stored `label`
(`""` for none) beside its index, so a caller printing them needs nothing
else, and `statistics` with `disagreement_percent`, the
`SCALE_DISAGREEMENT_PERCENT` that `disagrees` is measured against. The unit is
given by name (`"mm"`, `"cm"`, `"m"`, `"in"` or `"ft"`). An unknown unit, no
stored distance measurement and no usable one raise `ValueError` with the
error's sentence. Its tests are in
[`test_measurement_scale_rust_bindings.py`](../../../tests/rust_bindings/reconstruction/test_measurement_scale_rust_bindings.py),
on the demo scene's measurement of its cube's 1 m edge: its factor in two
units, the same edge stated three times with one 20% long giving the shorter
two's factor, a 20% deviation and each one's label, and the three errors.

## Theory

Measurement `i` states a length `L_i` in unit `u_i` between two endpoints that
resolve to positions `d_i` scene units apart. Its factor to the target unit `U`
is

```
f_i = L_i · metres(u_i) / metres(U) / d_i
```

the stated length converted through metres (`LengthUnit::in_metres`) over the
scene distance. A uniform scale of the coordinates by `f` multiplies every
scene distance by `f`, so scaling by `f_i` makes measurement `i` read exactly
its stated length in `U`. With several measurements they disagree a little,
by endpoint placement and reconstruction error, and the factor applied is
their **median**, which one misplaced endpoint cannot move far; the mean and
standard deviation are reported beside it for the spread, not used to choose.
The median and the median absolute deviation come from `sfmtool_core::numeric`.

A measurement is left out where it has no finite scene distance: an endpoint
the resolver refuses, an endpoint on a point at infinity (a direction, with no
distance to it), or two endpoints whose distance is zero or not finite. A
stated length that is not finite and strictly positive, or a unit code that
names no unit, is left out too; a load refuses such a row, so only a value
built by hand holds one. So is a reading whose endpoints are both finite but
which holds no scene distance, which the resolver never gives. Last, a
measurement whose stated and scene distances are each finite and positive is
left out where their ratio is not: a scene distance near the smallest
positive `f64` overflows the factor to infinity, and a tiny stated length (a
`float32` subnormal is a valid stored distance) over a large scene distance
underflows it to zero. Either would make the median or every deviation infinite or NaN.
The mean divides each factor by the count before summing, so factors near the
largest `f64` do not overflow it.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SCALE_DISAGREEMENT_PERCENT` | `5.0` | The largest deviation from the median, as a percentage of it, at which the measurements still agree; the bar `sfm xform --scale-to-median-measurement` prints its warning above. |

## Testing

In [`measurement_scale/tests.rs`](../../../crates/sfmtool-core/src/reconstruction/measurement_scale/tests.rs),
mostly on readings built by hand, since the operation reads only each
reading's endpoint outcomes and scene distance:

- Stated lengths in centimetres, inches and metres read in millimetres convert
  through metres, and the median, mean, sample standard deviation, median
  absolute deviation, minimum, maximum and each deviation are those of the
  three factors; the same readings in centimetres give a tenth of the factor.
- A factor more than 5% from the median makes the measurements disagree, and
  factors within 5% do not.
- A single measurement is the factor, with no sample deviation.
- An endpoint refused, an endpoint at infinity, a zero and an infinite scene
  distance, and a stated length that is negative or in no known unit are each
  excluded with their index and reason, a refusal before infinity.
- A factor that overflows to infinity and one that underflows to zero are
  excluded while the rest give a finite factor, three factors of `6e307` have a
  finite mean, and finite endpoints with no scene distance are excluded with a
  reason that holds no NaN.
- No usable measurement, no stored measurement, and readings that are not one
  per measurement are errors.
- On the endpoint resolver's synthetic fixture, two endpoints 3 scene units
  apart stated as 30 cm give the factor 100 to millimetres whether the file
  declares no unit or metres, and the reconstruction scaled by it resolves them
  300 apart.
