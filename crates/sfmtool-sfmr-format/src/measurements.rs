// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements stored in the optional `measurements/` section of a
//! `.sfmr` file, one subdirectory per kind of measurement.
//!
//! The one kind is **distance**, under `measurements/distance/`: each row
//! records a physical distance between two points observed at pixel
//! coordinates in a pair of images of the reconstruction. A row may leave its
//! stated distance or either endpoint absent, which its columns store as a
//! sentinel; [`DistanceMeasurementsData::row`] reads those as `None` and
//! [`DistanceMeasurementsData::from_rows`] writes them, so no reader outside
//! this crate tests a sentinel. See
//! `specs/formats/sfmr-file-format.md` § "10. Measurements (Optional, version 17+)".

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::types::{SfmrCamera, SfmrMetadata, WORLD_SPACE_UNITS};

/// Physical length unit for a measurement distance — the meaning behind a
/// `measurements/distance/units` code, drawn from [`WORLD_SPACE_UNITS`].
///
/// On disk, `measurements/distance/metadata.json` carries a `unit_names` legend
/// and each `measurements/distance/units` row is a `uint8` index into *that*
/// legend. A reader
/// resolves every code through the file's legend and hands back the canonical
/// numbering this enum defines ([`LENGTH_UNIT_MM`] through [`LENGTH_UNIT_FT`]),
/// and a writer always states [`Self::NAMES`]. So in memory there is one
/// numbering, and a consumer of [`DistanceMeasurementsData::units`] reads it with this
/// enum or those constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LengthUnit {
    /// Millimetres (`"mm"`, `0.001` m).
    Millimetre = 0,
    /// Centimetres (`"cm"`, `0.01` m).
    Centimetre = 1,
    /// Metres (`"m"`, `1.0` m).
    Metre = 2,
    /// Inches (`"in"`, `0.0254` m).
    Inch = 3,
    /// Feet (`"ft"`, `0.3048` m).
    Foot = 4,
}

const _: () = assert!(LengthUnit::ALL.len() == WORLD_SPACE_UNITS.len());

impl LengthUnit {
    /// Every physical length unit this format defines, in canonical order: a
    /// unit's position here is its [`Self::code`], and this is the legend a
    /// writer states and a reader normalises onto.
    pub const ALL: [LengthUnit; 5] = [
        Self::Millimetre,
        Self::Centimetre,
        Self::Metre,
        Self::Inch,
        Self::Foot,
    ];

    /// The canonical legend, one name per entry of [`Self::ALL`], drawn from
    /// [`WORLD_SPACE_UNITS`].
    pub const NAMES: [&'static str; 5] = [
        WORLD_SPACE_UNITS[0].0,
        WORLD_SPACE_UNITS[1].0,
        WORLD_SPACE_UNITS[2].0,
        WORLD_SPACE_UNITS[3].0,
        WORLD_SPACE_UNITS[4].0,
    ];

    /// The canonical code for this unit: its index in [`Self::ALL`].
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The legend name for this unit (the same spelling as
    /// [`WORLD_SPACE_UNITS`]).
    pub const fn name(self) -> &'static str {
        WORLD_SPACE_UNITS[self.code() as usize].0
    }

    /// The length of one unit in metres, from [`WORLD_SPACE_UNITS`].
    pub const fn in_metres(self) -> f64 {
        WORLD_SPACE_UNITS[self.code() as usize].1
    }

    /// The unit a legend name denotes, or `None` for a name this format does
    /// not define.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|u| u.name() == name)
    }

    /// The unit a canonical code denotes, or `None` for a code outside
    /// `0..Self::ALL.len()`.
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Millimetre),
            1 => Some(Self::Centimetre),
            2 => Some(Self::Metre),
            3 => Some(Self::Inch),
            4 => Some(Self::Foot),
            _ => None,
        }
    }
}

/// In-memory [`DistanceMeasurementsData::units`] code for [`LengthUnit::Millimetre`].
pub const LENGTH_UNIT_MM: u8 = LengthUnit::Millimetre.code();
/// In-memory [`DistanceMeasurementsData::units`] code for [`LengthUnit::Centimetre`].
pub const LENGTH_UNIT_CM: u8 = LengthUnit::Centimetre.code();
/// In-memory [`DistanceMeasurementsData::units`] code for [`LengthUnit::Metre`].
pub const LENGTH_UNIT_M: u8 = LengthUnit::Metre.code();
/// In-memory [`DistanceMeasurementsData::units`] code for [`LengthUnit::Inch`].
pub const LENGTH_UNIT_IN: u8 = LengthUnit::Inch.code();
/// In-memory [`DistanceMeasurementsData::units`] code for [`LengthUnit::Foot`].
pub const LENGTH_UNIT_FT: u8 = LengthUnit::Foot.code();

/// The image index an absent endpoint stores in
/// `measurements/distance/image_indexes`: `u32::MAX`, the `uint32` form of
/// `-1`. Its two coordinates in `measurements/distance/endpoints_xy` are both
/// `NaN`, as an unconstrained point stores `NaN` in
/// `points3d/constraint_distances` beside `0xFFFFFFFF` in
/// `points3d/constraint_reference_images`.
pub const ABSENT_ENDPOINT_IMAGE: u32 = u32::MAX;

/// One endpoint of a stored distance measurement as the file stores it: an
/// image of the reconstruction and the `float32` pixel in it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceMeasurementEndpoint {
    /// The image, by its index into `images/`.
    pub image: u32,
    /// The pixel `(x, y)` in that image (top-left origin, half-pixel centres).
    pub xy: [f32; 2],
}

/// One row of [`DistanceMeasurementsData`] with its absent parts as `None`:
/// what [`DistanceMeasurementsData::row`] gives and
/// [`DistanceMeasurementsData::from_rows`] takes.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceMeasurementRow {
    /// Endpoints `0` and `1`, each `None` where it is absent.
    pub endpoints: [Option<DistanceMeasurementEndpoint>; 2],
    /// The stated distance in [`Self::unit`], or `None` where the row states
    /// none.
    pub distance: Option<f32>,
    /// The unit the stated distance is in, or is given in once there is one.
    pub unit: LengthUnit,
    /// The label ([`is_valid_measurement_label`]).
    pub label: String,
}

/// Metadata from `measurements/distance/metadata.json.zst`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistanceMeasurementsMetadata {
    /// Number of distance measurements (`>= 1`). Must match top-level
    /// `SfmrMetadata::distance_measurement_count`.
    pub distance_measurement_count: u32,
    /// The legend `measurements/distance/units` indexes, one unit name per code in code
    /// order. A writer always writes [`LengthUnit::NAMES`]; a reader accepts
    /// any non-empty list of unique defined unit names in any order and
    /// normalises the column onto the canonical order.
    pub unit_names: Vec<String>,
}

/// Physical distance measurements, the distance kind of the `.sfmr`
/// `measurements/` section, stored under `measurements/distance/`.
///
/// Each row records a physical distance between two points observed in
/// images of the reconstruction. Absent (`None` on
/// [`MeasurementsData::distance`]) when the reconstruction has no distance
/// measurements; when present, there is at least one (`len() >= 1`).
///
/// The columns are flat arrays in the shapes the file stores, as every other
/// column of the format is, and the Python binding hands them over as arrays.
/// A row's stated distance and each of its endpoints may be absent, stored as a
/// sentinel: `NaN` in [`Self::distances`], and [`ABSENT_ENDPOINT_IMAGE`] with
/// `(NaN, NaN)` for an endpoint. [`Self::row`], [`Self::stated_distance`] and
/// [`Self::endpoint`] read them as `None`, and [`Self::from_rows`] writes them,
/// so a reader of the rows works with `Option`s. Equality treats two `NaN`s
/// as equal, since both say the same thing is absent.
///
/// Both distances and endpoint pixel coordinates are stored as `float32`.
/// Physical distance measurements (tape measures, laser rangefinders,
/// calibrated scale bars) carry at most four to five significant digits, and
/// `float32`'s 24-bit significand (relative rounding error at most `2^-24`,
/// roughly seven decimal digits) preserves them across positive lengths from
/// micrometres to kilometres at half the storage of `float64`. Endpoint pixel
/// coordinates use `float32` for the same reason `tracks/keypoints_xy` does:
/// sub-pixel image coordinates up to tens of thousands of pixels need far fewer
/// than seven digits.
#[derive(Debug, Clone)]
pub struct DistanceMeasurementsData {
    /// `(N,)` stated physical distance per measurement, strictly positive and
    /// finite, in the unit named by the matching row of [`Self::units`], or
    /// `NaN` for a measurement that states none.
    ///
    /// On disk `measurements/distance/distances.{N}.float32.zst`.
    pub distances: Array1<f32>,
    /// `(N,)` physical length unit code per measurement in the canonical
    /// numbering of [`LengthUnit`] ([`LENGTH_UNIT_MM`] through
    /// [`LENGTH_UNIT_FT`]). A row that states no distance still carries a
    /// unit: the one its distance is given in once it is given.
    ///
    /// On disk `measurements/distance/units.{N}.uint8.zst`, indexing the
    /// `unit_names` legend in `measurements/distance/metadata.json.zst`. The reader resolves every
    /// code through the file's legend and normalises onto the canonical
    /// numbering at the I/O boundary; the writer always states
    /// [`LengthUnit::NAMES`].
    pub units: Array1<u8>,
    /// `(N, 2)` image index pair `[img_0, img_1]` into `images/`, each below
    /// `image_count`, or [`ABSENT_ENDPOINT_IMAGE`] for an absent endpoint.
    /// Both endpoints may lie in the same image (`img_0 == img_1`).
    ///
    /// On disk `measurements/distance/image_indexes.{N}.2.uint32.zst`.
    pub image_indexes: Array2<u32>,
    /// `(N, 2, 2)` sub-pixel endpoint coordinates where `[i, k, 0..2]` is
    /// `(x, y)` in pixel coordinates of `images[image_indexes[[i, k]]]` for
    /// endpoint `k` in `{0, 1}` (top-left origin, half-pixel centres, finite
    /// and within `[0, width) × [0, height)` of that image's camera), or
    /// `(NaN, NaN)` for an absent endpoint.
    ///
    /// On disk `measurements/distance/endpoints_xy.{N}.2.2.float32.zst`.
    pub endpoints_xy: Array3<f32>,
    /// `N` human-readable labels or descriptions, one per measurement, each
    /// valid by [`is_valid_measurement_label`]: not empty and not only white
    /// space.
    ///
    /// On disk `measurements/distance/labels.json.zst`.
    pub labels: Vec<String>,
}

impl PartialEq for DistanceMeasurementsData {
    fn eq(&self, other: &Self) -> bool {
        let same = |a: &f32, b: &f32| a == b || (a.is_nan() && b.is_nan());
        self.distances.shape() == other.distances.shape()
            && self
                .distances
                .iter()
                .zip(&other.distances)
                .all(|(a, b)| same(a, b))
            && self.units == other.units
            && self.image_indexes == other.image_indexes
            && self.endpoints_xy.shape() == other.endpoints_xy.shape()
            && self
                .endpoints_xy
                .iter()
                .zip(&other.endpoints_xy)
                .all(|(a, b)| same(a, b))
            && self.labels == other.labels
    }
}

impl DistanceMeasurementsData {
    /// Number of distance measurements.
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Whether this struct holds zero measurements (which is invalid on disk;
    /// leave [`MeasurementsData::distance`] `None` instead).
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Row `i`'s stated distance, or `None` where it states none (`NaN`) or
    /// there is no row `i`.
    pub fn stated_distance(&self, i: usize) -> Option<f32> {
        self.distances.get(i).copied().filter(|d| !d.is_nan())
    }

    /// Endpoint `k` (`0` or `1`) of row `i`, or `None` where it is absent
    /// ([`ABSENT_ENDPOINT_IMAGE`]) or there is no such entry.
    ///
    /// Read through `.get`, so a value built by hand whose columns disagree in
    /// length reads a missing image index as an absent endpoint and a missing
    /// coordinate as not a number, never as a panic. A load refuses such a
    /// value, so an endpoint read from a file is read exactly.
    pub fn endpoint(&self, i: usize, k: usize) -> Option<DistanceMeasurementEndpoint> {
        let image = self.image_indexes.get((i, k)).copied()?;
        if image == ABSENT_ENDPOINT_IMAGE {
            return None;
        }
        let xy = [0, 1].map(|c| {
            self.endpoints_xy
                .get((i, k, c))
                .copied()
                .unwrap_or(f32::NAN)
        });
        Some(DistanceMeasurementEndpoint { image, xy })
    }

    /// Row `i`, with its absent stated distance and endpoints read as `None`.
    ///
    /// Read through `.get`, as [`Self::endpoint`] is: in a value built by hand
    /// whose columns disagree in length, a missing distance reads as none, a
    /// missing unit code or one that names no unit as metres, and a missing
    /// label as empty.
    pub fn row(&self, i: usize) -> DistanceMeasurementRow {
        DistanceMeasurementRow {
            endpoints: [self.endpoint(i, 0), self.endpoint(i, 1)],
            distance: self.stated_distance(i),
            unit: self
                .units
                .get(i)
                .and_then(|&code| LengthUnit::from_code(code))
                .unwrap_or(LengthUnit::Metre),
            label: self.labels.get(i).cloned().unwrap_or_default(),
        }
    }

    /// Every row, in stored order ([`Self::row`]).
    pub fn rows(&self) -> Vec<DistanceMeasurementRow> {
        (0..self.len()).map(|i| self.row(i)).collect()
    }

    /// The columns that hold `rows`, each `None` written as its sentinel.
    ///
    /// A file holds at least one row, so for an empty `rows` this gives
    /// columns [`validate_distance_measurements`] refuses; leave
    /// [`MeasurementsData::distance`] `None` instead.
    pub fn from_rows(rows: &[DistanceMeasurementRow]) -> Self {
        let n = rows.len();
        let endpoint = |i: usize, k: usize| rows[i].endpoints[k];
        Self {
            distances: Array1::from_iter(rows.iter().map(|r| r.distance.unwrap_or(f32::NAN))),
            units: Array1::from_iter(rows.iter().map(|r| r.unit.code())),
            image_indexes: Array2::from_shape_fn((n, 2), |(i, k)| {
                endpoint(i, k).map_or(ABSENT_ENDPOINT_IMAGE, |e| e.image)
            }),
            endpoints_xy: Array3::from_shape_fn((n, 2, 2), |(i, k, c)| {
                endpoint(i, k).map_or(f32::NAN, |e| e.xy[c])
            }),
            labels: rows.iter().map(|r| r.label.clone()).collect(),
        }
    }
}

/// Whether `label` is one a stored measurement may carry: not empty, and not
/// only white space (`str::trim`). The label is stored as given, untrimmed.
pub fn is_valid_measurement_label(label: &str) -> bool {
    !label.trim().is_empty()
}

/// The measurements a `.sfmr` file stores in its optional `measurements/`
/// section, one field per kind of measurement, each stored in the
/// subdirectory of that section named for the kind.
///
/// A kind is `None` when the reconstruction has none of it. The section is
/// present in a file exactly when at least one kind is present; the default
/// value, with every kind `None`, is a reconstruction with no measurements.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeasurementsData {
    /// Physical distance measurements, under `measurements/distance/`.
    pub distance: Option<DistanceMeasurementsData>,
}

impl MeasurementsData {
    /// Whether no kind of measurement is present, so a file stores no
    /// `measurements/` section.
    pub fn is_empty(&self) -> bool {
        self.distance.is_none()
    }

    /// Number of distance measurements, `0` when there are none.
    pub fn distance_measurement_count(&self) -> usize {
        self.distance
            .as_ref()
            .map_or(0, DistanceMeasurementsData::len)
    }

    /// Set each kind's top-level count in `metadata` to what this holds:
    /// `distance_measurement_count` to the number of distance measurements, or
    /// `None` when there are none.
    pub fn set_metadata_counts(&self, metadata: &mut SfmrMetadata) {
        metadata.distance_measurement_count = self.distance.as_ref().map(|d| d.len() as u32);
    }
}

/// The unit each legend name denotes, in the order given — the legend a stored
/// `measurements/distance/units` code indexes.
///
/// A legend must name at least one unit, name only units in
/// [`LengthUnit::NAMES`], and name each at most once. Read and verify both
/// resolve stored codes through a legend this accepted, so a file states its
/// own numbering and a reader normalises it onto the canonical order at the
/// I/O boundary.
pub(crate) fn parse_unit_names<S: AsRef<str>>(names: &[S]) -> Result<Vec<LengthUnit>, String> {
    if names.is_empty() {
        return Err("unit_names is empty, so it names no unit at all".into());
    }
    let mut legend = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let name = name.as_ref();
        let unit = LengthUnit::from_name(name).ok_or_else(|| {
            format!(
                "unit_names[{i}] is {name:?}, not one of {:?}",
                LengthUnit::NAMES
            )
        })?;
        if legend.contains(&unit) {
            return Err(format!(
                "unit_names[{i}] repeats {name:?}, which already has a code"
            ));
        }
        legend.push(unit);
    }
    Ok(legend)
}

/// Read and validate the `unit_names` legend from
/// `measurements/distance/metadata.json`.
pub(crate) fn read_measurement_unit_legend(
    measurements_meta: &serde_json::Value,
) -> Result<Vec<LengthUnit>, String> {
    let value = measurements_meta.get("unit_names").ok_or_else(|| {
        "measurements/distance/metadata.json carries no unit_names to read the unit codes through"
            .to_string()
    })?;
    let names: Vec<String> = value
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .map(|v| Some(v.as_str()?.to_string()))
                .collect::<Option<Vec<String>>>()
        })
        .ok_or_else(|| {
            format!(
                "measurements/distance/metadata.json's unit_names is {value}, not a list of names"
            )
        })?;
    parse_unit_names(&names)
}

/// Resolve each `measurements/distance/units` code through `legend`.
///
/// A code past the legend's end is rejected rather than guessed at. Both the
/// reader and the verifier pass the file's legend.
pub(crate) fn resolve_measurement_units(
    codes: &[u8],
    legend: &[LengthUnit],
) -> Result<Vec<LengthUnit>, String> {
    codes
        .iter()
        .enumerate()
        .map(|(i, &code)| {
            legend.get(code as usize).copied().ok_or_else(|| {
                format!(
                    "measurements/distance/units row {i} is {code}, past the {} names its legend gives",
                    legend.len()
                )
            })
        })
        .collect()
}

/// Validate a [`MeasurementsData`] against the reconstruction's images and
/// cameras: each kind that is present, by its own rules
/// ([`validate_distance_measurements`] for the distance kind).
pub fn validate_measurements(
    measurements: &MeasurementsData,
    camera_indexes: &[u32],
    cameras: &[SfmrCamera],
) -> Result<(), String> {
    if let Some(distance) = &measurements.distance {
        validate_distance_measurements(distance, camera_indexes, cameras)?;
    }
    Ok(())
}

/// Validate a [`DistanceMeasurementsData`] against the reconstruction's images
/// and cameras.
///
/// Checks that `measurements` has at least one row (`>= 1` and fits in `u32`),
/// all five columns agree in shape, every distance is NaN (none stated) or
/// finite and strictly positive, every unit code is a canonical
/// [`LengthUnit`] code (`0..=4`), every label is valid
/// ([`is_valid_measurement_label`]), and every endpoint is either absent,
/// [`ABSENT_ENDPOINT_IMAGE`] with both coordinates `NaN`, or present, an image
/// index below `camera_indexes.len()` with a valid camera index and an
/// `(x, y)` that is finite and within `[0, width) × [0, height)` of its
/// image's camera. The index is [`ABSENT_ENDPOINT_IMAGE`] if and only if both
/// coordinates are `NaN`: a `NaN` in one coordinate only, `NaN` coordinates
/// with a present index, and that index with a coordinate that is not `NaN`
/// are each refused.
///
/// Read, write and verify all route through this so the three enforce the same
/// rules.
pub fn validate_distance_measurements(
    measurements: &DistanceMeasurementsData,
    camera_indexes: &[u32],
    cameras: &[SfmrCamera],
) -> Result<(), String> {
    let n = measurements.distances.len();
    if n == 0 {
        return Err(
            "the distance kind must have at least one measurement (leave it None when there are none)"
                .into(),
        );
    }
    if u32::try_from(n).is_err() {
        return Err(format!(
            "distance measurement count {n} exceeds u32::MAX ({})",
            u32::MAX
        ));
    }
    if measurements.units.len() != n {
        return Err(format!(
            "measurements/distance/units len {} != distance_measurement_count {n}",
            measurements.units.len()
        ));
    }
    if measurements.image_indexes.shape() != [n, 2] {
        return Err(format!(
            "measurements/distance/image_indexes shape {:?} != [{n}, 2]",
            measurements.image_indexes.shape()
        ));
    }
    if measurements.endpoints_xy.shape() != [n, 2, 2] {
        return Err(format!(
            "measurements/distance/endpoints_xy shape {:?} != [{n}, 2, 2]",
            measurements.endpoints_xy.shape()
        ));
    }
    if measurements.labels.len() != n {
        return Err(format!(
            "measurements/distance/labels len {} != distance_measurement_count {n}",
            measurements.labels.len()
        ));
    }

    let image_count = camera_indexes.len();
    for i in 0..n {
        let d = measurements.distances[i];
        if !d.is_nan() && !(d.is_finite() && d > 0.0) {
            return Err(format!(
                "measurements/distance/distances row {i} is {d}, which must be finite and \
                 strictly positive, or NaN for no stated distance"
            ));
        }
        let code = measurements.units[i];
        if LengthUnit::from_code(code).is_none() {
            return Err(format!(
                "measurements/distance/units row {i} is {code}, past the {} names its legend gives",
                LengthUnit::ALL.len()
            ));
        }
        let label = &measurements.labels[i];
        if !is_valid_measurement_label(label) {
            return Err(format!(
                "measurements/distance/labels row {i} is {label:?}, and a label must not be \
                 empty or only white space"
            ));
        }
        for k in 0..2 {
            let x = measurements.endpoints_xy[[i, k, 0]];
            let y = measurements.endpoints_xy[[i, k, 1]];
            let absent_xy = x.is_nan() && y.is_nan();
            let image = measurements.image_indexes[[i, k]];
            if image == ABSENT_ENDPOINT_IMAGE {
                if !absent_xy {
                    return Err(format!(
                        "measurements/distance/endpoints_xy[{i}, {k}] is ({x}, {y}), but \
                         image_indexes[{i}, {k}] is {ABSENT_ENDPOINT_IMAGE}, an absent endpoint, \
                         whose coordinates must both be NaN"
                    ));
                }
                continue;
            }
            if absent_xy {
                return Err(format!(
                    "measurements/distance/endpoints_xy[{i}, {k}] is ({x}, {y}), an absent \
                     endpoint, but image_indexes[{i}, {k}] is {image}, not \
                     {ABSENT_ENDPOINT_IMAGE}"
                ));
            }
            let img = image as usize;
            let cam = *camera_indexes.get(img).ok_or_else(|| {
                format!(
                    "measurements/distance/image_indexes[{i}, {k}] = {img} is out of range for \
                     {image_count} images"
                )
            })? as usize;
            let c = cameras.get(cam).ok_or_else(|| {
                format!(
                    "measurements/distance/image_indexes[{i}, {k}] = {img}: camera index {cam} out of range"
                )
            })?;
            if !x.is_finite() || !y.is_finite() {
                return Err(format!(
                    "measurements/distance/endpoints_xy[{i}, {k}] is not finite: ({x}, {y})"
                ));
            }
            if !(x >= 0.0 && x < c.width as f32 && y >= 0.0 && y < c.height as f32) {
                return Err(format!(
                    "measurements/distance/endpoints_xy[{i}, {k}] = ({x}, {y}) is outside image bounds \
                     [0, {}) x [0, {})",
                    c.width, c.height
                ));
            }
        }
    }
    Ok(())
}
