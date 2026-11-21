// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Physical distance measurements stored in the optional `measurements/`
//! section of a `.sfmr` file.
//!
//! Each row records a known physical distance between two 3D points observed at
//! pixel coordinates in a pair of images of the reconstruction. See
//! `specs/formats/sfmr-file-format.md` § "Measurements (Optional)".

use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

use crate::types::{SfmrCamera, WORLD_SPACE_UNITS};

/// Physical length unit for a measurement distance — the meaning behind a
/// `measurements/units` code, drawn from [`WORLD_SPACE_UNITS`].
///
/// On disk, `measurements/metadata.json` carries a `unit_names` legend and each
/// `measurements/units` row is a `uint8` index into *that* legend. A reader
/// resolves every code through the file's legend and hands back the canonical
/// numbering this enum defines ([`LENGTH_UNIT_MM`] through [`LENGTH_UNIT_FT`]),
/// and a writer always states [`Self::NAMES`]. So in memory there is one
/// numbering, and a consumer of [`MeasurementsData::units`] reads it with this
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

/// In-memory [`MeasurementsData::units`] code for [`LengthUnit::Millimetre`].
pub const LENGTH_UNIT_MM: u8 = LengthUnit::Millimetre.code();
/// In-memory [`MeasurementsData::units`] code for [`LengthUnit::Centimetre`].
pub const LENGTH_UNIT_CM: u8 = LengthUnit::Centimetre.code();
/// In-memory [`MeasurementsData::units`] code for [`LengthUnit::Metre`].
pub const LENGTH_UNIT_M: u8 = LengthUnit::Metre.code();
/// In-memory [`MeasurementsData::units`] code for [`LengthUnit::Inch`].
pub const LENGTH_UNIT_IN: u8 = LengthUnit::Inch.code();
/// In-memory [`MeasurementsData::units`] code for [`LengthUnit::Foot`].
pub const LENGTH_UNIT_FT: u8 = LengthUnit::Foot.code();

/// Metadata from `measurements/metadata.json.zst`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementsMetadata {
    /// Number of measurements (`>= 1`). Must match top-level
    /// `SfmrMetadata::measurement_count`.
    pub measurement_count: u32,
    /// The legend `measurements/units` indexes, one unit name per code in code
    /// order. A writer always writes [`LengthUnit::NAMES`]; a reader accepts
    /// any non-empty list of unique defined unit names in any order and
    /// normalises the column onto the canonical order.
    pub unit_names: Vec<String>,
}

/// Optional physical distance measurements stored in the `.sfmr`
/// `measurements/` section.
///
/// Each row records a known physical distance between two points observed in
/// images of the reconstruction. Absent (`None` on
/// [`crate::SfmrData::measurements`]) when the reconstruction has no
/// measurements; when present, there is at least one measurement (`len() >= 1`).
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
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementsData {
    /// `(N,)` measured physical distance per measurement, strictly positive and
    /// finite, in the unit named by the matching row of [`Self::units`].
    ///
    /// On disk `measurements/distances.{N}.float32.zst`.
    pub distances: Array1<f32>,
    /// `(N,)` physical length unit code per measurement in the canonical
    /// numbering of [`LengthUnit`] ([`LENGTH_UNIT_MM`] through
    /// [`LENGTH_UNIT_FT`]).
    ///
    /// On disk `measurements/units.{N}.uint8.zst`, indexing the `unit_names`
    /// legend in `measurements/metadata.json.zst`. The reader resolves every
    /// code through the file's legend and normalises onto the canonical
    /// numbering at the I/O boundary; the writer always states
    /// [`LengthUnit::NAMES`].
    pub units: Array1<u8>,
    /// `(N, 2)` image index pair `[img_0, img_1]` into `images/`, each below
    /// `image_count`. Both endpoints may lie in the same image
    /// (`img_0 == img_1`).
    ///
    /// On disk `measurements/image_indexes.{N}.2.uint32.zst`.
    pub image_indexes: Array2<u32>,
    /// `(N, 2, 2)` sub-pixel endpoint coordinates where `[i, k, 0..2]` is
    /// `(x, y)` in pixel coordinates of `images[image_indexes[[i, k]]]` for
    /// endpoint `k` in `{0, 1}` (top-left origin, half-pixel centres, finite
    /// and within `[0, width) × [0, height)` of that image's camera).
    ///
    /// On disk `measurements/endpoints_xy.{N}.2.2.float32.zst`.
    pub endpoints_xy: Array3<f32>,
    /// `N` human-readable labels or descriptions, one per measurement. An empty
    /// string `""` is allowed for an unlabelled measurement.
    ///
    /// On disk `measurements/labels.json.zst`.
    pub labels: Vec<String>,
}

impl MeasurementsData {
    /// Number of measurements in this section.
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Whether this struct holds zero measurements (which is invalid on disk;
    /// omit the section as `None` instead).
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }
}

/// The unit each legend name denotes, in the order given — the legend a stored
/// `measurements/units` code indexes.
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

/// Read and validate the `unit_names` legend from `measurements/metadata.json`.
pub(crate) fn read_measurement_unit_legend(
    measurements_meta: &serde_json::Value,
) -> Result<Vec<LengthUnit>, String> {
    let value = measurements_meta.get("unit_names").ok_or_else(|| {
        "measurements/metadata.json carries no unit_names to read the unit codes through"
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
            format!("measurements/metadata.json's unit_names is {value}, not a list of names")
        })?;
    parse_unit_names(&names)
}

/// Resolve each `measurements/units` code through `legend`.
///
/// A code past the legend's end is rejected rather than guessed at. The reader
/// and verifier pass the file's legend; the writer passes [`LengthUnit::ALL`]
/// to refuse codes outside the canonical numbering it states.
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
                    "measurements/units row {i} is {code}, past the {} names its legend gives",
                    legend.len()
                )
            })
        })
        .collect()
}

/// Validate a [`MeasurementsData`] against the reconstruction's images and
/// cameras.
///
/// Checks that `measurements` has at least one row (`>= 1` and fits in `u32`),
/// all five columns agree in shape, every distance is finite and strictly
/// positive, every unit code is a canonical [`LengthUnit`] code (`0..=4`),
/// every image index is below `camera_indexes.len()` with a valid camera index,
/// and every endpoint `(x, y)` is finite and within `[0, width) × [0, height)`
/// of its image's camera.
///
/// Read, write and verify all route through this so the three enforce the same
/// rules.
pub fn validate_measurements(
    measurements: &MeasurementsData,
    camera_indexes: &[u32],
    cameras: &[SfmrCamera],
) -> Result<(), String> {
    let n = measurements.distances.len();
    if n == 0 {
        return Err(
            "measurements section must have at least one measurement (use None when there are none)"
                .into(),
        );
    }
    if u32::try_from(n).is_err() {
        return Err(format!(
            "measurements count {n} exceeds u32::MAX ({})",
            u32::MAX
        ));
    }
    if measurements.units.len() != n {
        return Err(format!(
            "measurements/units len {} != measurement_count {n}",
            measurements.units.len()
        ));
    }
    if measurements.image_indexes.shape() != [n, 2] {
        return Err(format!(
            "measurements/image_indexes shape {:?} != [{n}, 2]",
            measurements.image_indexes.shape()
        ));
    }
    if measurements.endpoints_xy.shape() != [n, 2, 2] {
        return Err(format!(
            "measurements/endpoints_xy shape {:?} != [{n}, 2, 2]",
            measurements.endpoints_xy.shape()
        ));
    }
    if measurements.labels.len() != n {
        return Err(format!(
            "measurements/labels len {} != measurement_count {n}",
            measurements.labels.len()
        ));
    }

    let image_count = camera_indexes.len();
    for i in 0..n {
        let d = measurements.distances[i];
        if !d.is_finite() || d <= 0.0 {
            return Err(format!(
                "measurements/distances row {i} is {d}, which must be finite and strictly positive"
            ));
        }
        let code = measurements.units[i];
        if LengthUnit::from_code(code).is_none() {
            return Err(format!(
                "measurements/units row {i} is {code}, past the {} names its legend gives",
                LengthUnit::ALL.len()
            ));
        }
        for k in 0..2 {
            let img = measurements.image_indexes[[i, k]] as usize;
            let cam = *camera_indexes.get(img).ok_or_else(|| {
                format!(
                    "measurements/image_indexes[{i}, {k}] = {img} is out of range for \
                     {image_count} images"
                )
            })? as usize;
            let c = cameras.get(cam).ok_or_else(|| {
                format!(
                    "measurements/image_indexes[{i}, {k}] = {img}: camera index {cam} out of range"
                )
            })?;
            let x = measurements.endpoints_xy[[i, k, 0]];
            let y = measurements.endpoints_xy[[i, k, 1]];
            if !x.is_finite() || !y.is_finite() {
                return Err(format!(
                    "measurements/endpoints_xy[{i}, {k}] is not finite: ({x}, {y})"
                ));
            }
            if !(x >= 0.0 && x < c.width as f32 && y >= 0.0 && y < c.height as f32) {
                return Err(format!(
                    "measurements/endpoints_xy[{i}, {k}] = ({x}, {y}) is outside image bounds \
                     [0, {}) x [0, {})",
                    c.width, c.height
                ));
            }
        }
    }
    Ok(())
}
