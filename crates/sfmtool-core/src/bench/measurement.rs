// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The editable measurement: a stored measurement put on the bench to be worked
//! on, or a new one started from a pixel, the steps that set its label, its
//! stated distance and an endpoint, and the commit that writes it back.
//!
//! `specs/core/bench/editable-measurement.md` is the design. An
//! [`EditableMeasurement`] holds only what the file stores of a measurement:
//! each endpoint's image and pixel, the stated distance and its unit, and the
//! label. Everything else about an endpoint (its point, its texel, its 3D
//! position) is read again by the resolver every time
//! ([`resolve_distance_endpoints`](crate::reconstruction::measurement_endpoint::resolve_distance_endpoints)).
//!
//! Every function here is pure, as the track's steps are: [`put_measurement`]
//! and [`start_measurement`] give back the next bench, [`set_measurement_label`],
//! [`set_stated_distance`] and [`set_measurement_endpoint`] the next item, [`commit_measurement`] the next
//! reconstruction
//! value, and [`delete_measurement`] the value with one row fewer, which is a
//! direct edit of the reconstruction rather than a bench step.

use std::sync::Arc;

use crate::reconstruction::edited::EditedReconstruction;
use crate::{
    is_valid_measurement_label, DistanceMeasurementEndpoint, DistanceMeasurementRow,
    DistanceMeasurementsData, LengthUnit, MeasurementsData,
};

use super::steps::ClusterSeed;
use super::{image_stem, Bench, BenchItem};

/// The kind of a stored measurement, which names the subdirectory of the
/// file's `measurements/` section its rows are in. Each kind's rows are
/// numbered within that kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeasurementKind {
    /// A physical length between two endpoints, under `measurements/distance/`.
    Distance,
}

impl MeasurementKind {
    /// Every kind, in the order a list of them is given.
    pub const ALL: [MeasurementKind; 1] = [MeasurementKind::Distance];

    /// The kind's name at the start of a title: `Distance`.
    pub fn title(self) -> &'static str {
        match self {
            MeasurementKind::Distance => "Distance",
        }
    }

    /// The kind's name in the middle of a sentence, and on the wire: the
    /// file's subdirectory name, `distance`.
    pub fn name(self) -> &'static str {
        match self {
            MeasurementKind::Distance => "distance",
        }
    }

    /// The kind [`Self::name`] spells, or `None`.
    pub fn from_name(name: &str) -> Option<MeasurementKind> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    /// How many measurements of this kind `measurements` holds.
    pub fn count(self, measurements: &MeasurementsData) -> usize {
        match self {
            MeasurementKind::Distance => measurements.distance_measurement_count(),
        }
    }
}

impl std::fmt::Display for MeasurementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `distance #2`: a stored measurement named by its kind and row, which is the
/// label a measurement put on the bench from that row is minted from.
pub fn measurement_title(kind: MeasurementKind, index: u32) -> String {
    format!("{} #{index}", kind.name())
}

/// One end of a distance measurement, as the file stores it: an image of the
/// reconstruction and a pixel in it (top-left origin, half-pixel centres).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementEndpoint {
    /// The image, by its index in the reconstruction's image set.
    pub image: u32,
    /// The pixel in that image.
    pub pixel: [f64; 2],
}

/// A distance measurement being worked on.
#[derive(Debug, Clone, PartialEq)]
pub struct EditableDistance {
    /// Endpoints A and B (the format's endpoint 0 and 1). `None` for an
    /// endpoint that is not placed: deleting the image an endpoint was in
    /// clears it.
    pub endpoints: [Option<MeasurementEndpoint>; 2],
    /// The stated length in [`Self::unit`], once given.
    pub distance: Option<f64>,
    /// The unit the stated length is in, or is given in once there is one:
    /// the file stores a unit on every row.
    pub unit: LengthUnit,
}

/// What an editable measurement measures, one variant per kind.
#[derive(Debug, Clone, PartialEq)]
pub enum EditableMeasurementBody {
    /// A physical length between two endpoints.
    Distance(EditableDistance),
}

/// Where an editable measurement came from: a stored row, named by the
/// version it was read out of.
///
/// A measurement has no portable id, so the row is the only name it has, and
/// a caller that keeps versions follows the row from `version` to the one it
/// stands at, as it follows a track's [`Origin`](super::Origin).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasurementOrigin {
    /// The version the measurement was read out of, as the caller numbers
    /// versions. Not interpreted here.
    pub version: u64,
    /// The kind, whose rows `index` counts.
    pub kind: MeasurementKind,
    /// The row within that kind's rows.
    pub index: u32,
}

/// A measurement being worked on: where it came from, and what it will write.
///
/// It holds only what the file stores. An endpoint's point, texel and 3D
/// position are not held: they follow from its image and pixel through the
/// resolver, read against whichever version the caller stands at.
#[derive(Debug, Clone, PartialEq)]
pub struct EditableMeasurement {
    /// The stored measurement it was taken from, or `None` for one with no row
    /// to replace. A commit replaces the row the origin resolves to, and
    /// appends a row where it resolves to none.
    pub origin: Option<MeasurementOrigin>,
    /// The measurement's own label, written into the file on commit. A commit
    /// needs it not empty and not only white space, but it need not be unique,
    /// so it is not the item's bench label.
    pub label: String,
    /// What it measures.
    pub body: EditableMeasurementBody,
}

/// A stored measurement that is not there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoSuchMeasurement {
    /// The kind asked for.
    pub kind: MeasurementKind,
    /// The row asked for.
    pub index: u32,
    /// How many of that kind the reconstruction holds.
    pub count: usize,
}

impl std::fmt::Display for NoSuchMeasurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = self.kind.name();
        match self.count {
            0 => write!(
                f,
                "there is no {kind} #{}: the reconstruction has no {kind} measurements",
                self.index
            ),
            1 => write!(
                f,
                "there is no {kind} #{}: the reconstruction has 1 {kind} measurement, {kind} #0",
                self.index
            ),
            n => write!(
                f,
                "there is no {kind} #{}: the reconstruction has {n} {kind} measurements, \
                 {kind} #0 to #{}",
                self.index,
                n - 1
            ),
        }
    }
}

impl std::error::Error for NoSuchMeasurement {}

impl EditableMeasurement {
    /// The stored measurement at row `index` of `kind` in `edited`, as an item
    /// whose origin is that row at `version`.
    ///
    /// The endpoints, the stated distance, its unit and the label are copied
    /// as the file holds them, so a commit of the item unchanged writes the
    /// same row.
    pub fn from_stored(
        edited: &EditedReconstruction,
        kind: MeasurementKind,
        index: u32,
        version: u64,
    ) -> Result<Self, NoSuchMeasurement> {
        let measurements = edited.measurements();
        let missing = || NoSuchMeasurement {
            kind,
            index,
            count: kind.count(measurements),
        };
        let row = index as usize;
        let body = match kind {
            MeasurementKind::Distance => {
                let m = measurements.distance.as_ref().ok_or_else(missing)?;
                if row >= m.len() {
                    return Err(missing());
                }
                let stored = m.row(row);
                EditableMeasurementBody::Distance(EditableDistance {
                    endpoints: stored.endpoints.map(|e| {
                        e.map(|e| MeasurementEndpoint {
                            image: e.image,
                            pixel: e.xy.map(f64::from),
                        })
                    }),
                    distance: stored.distance.map(f64::from),
                    unit: stored.unit,
                })
            }
        };
        Ok(Self {
            origin: Some(MeasurementOrigin {
                version,
                kind,
                index,
            }),
            label: measurements
                .distance
                .as_ref()
                .and_then(|m| m.labels.get(row).cloned())
                .unwrap_or_default(),
            body,
        })
    }

    /// The kind of measurement this is.
    pub fn kind(&self) -> MeasurementKind {
        match self.body {
            EditableMeasurementBody::Distance(_) => MeasurementKind::Distance,
        }
    }

    /// The distance this is, when it is one.
    pub fn as_distance(&self) -> Option<&EditableDistance> {
        match &self.body {
            EditableMeasurementBody::Distance(distance) => Some(distance),
        }
    }

    /// This item with its origin set to row `index` at `version`, and nothing
    /// else changed: what a commit seats the item on.
    pub fn with_origin(&self, version: u64, index: u32) -> Self {
        Self {
            origin: Some(MeasurementOrigin {
                version,
                kind: self.kind(),
                index,
            }),
            ..self.clone()
        }
    }

    /// This item as it reads after image `image` is deleted from its
    /// reconstruction, which moves every later image down by one: an endpoint
    /// in `image` is cleared, and one in a later image moves down by one.
    ///
    /// `None` when no endpoint is in `image` or after it, so the caller keeps
    /// the item it has. Otherwise the item and how many endpoints were
    /// cleared. The origin is left alone: it names a row at a version, and the
    /// caller follows it through the delete's own row map.
    pub fn delete_image(&self, image: u32) -> Option<(EditableMeasurement, usize)> {
        let EditableMeasurementBody::Distance(distance) = &self.body;
        let reached = distance
            .endpoints
            .iter()
            .flatten()
            .any(|endpoint| endpoint.image >= image);
        if !reached {
            return None;
        }
        let mut cleared = 0;
        let endpoints = distance.endpoints.map(|endpoint| {
            let endpoint = endpoint?;
            match endpoint.image.cmp(&image) {
                std::cmp::Ordering::Less => Some(endpoint),
                std::cmp::Ordering::Equal => {
                    cleared += 1;
                    None
                }
                std::cmp::Ordering::Greater => Some(MeasurementEndpoint {
                    image: endpoint.image - 1,
                    ..endpoint
                }),
            }
        });
        let next = Self {
            body: EditableMeasurementBody::Distance(EditableDistance {
                endpoints,
                ..*distance
            }),
            ..self.clone()
        };
        Some((next, cleared))
    }

    /// How many endpoints are placed.
    pub fn placed_endpoints(&self) -> usize {
        match &self.body {
            EditableMeasurementBody::Distance(distance) => {
                distance.endpoints.iter().flatten().count()
            }
        }
    }
}

/// Put the stored measurement at row `index` of `kind` on the bench, under a
/// label minted from its kind and row (`distance #2`), and give back the bench
/// and the label it took.
///
/// The item's origin is that row at `version`, the caller's own numbering of
/// the version `edited` is. Whether the measurement is on the bench already is
/// the caller's to ask, since only the caller can follow an origin from the
/// version it was read at to the one it stands at.
pub fn put_measurement(
    bench: &Bench,
    edited: &EditedReconstruction,
    kind: MeasurementKind,
    index: u32,
    version: u64,
) -> Result<(Bench, String), NoSuchMeasurement> {
    let item = EditableMeasurement::from_stored(edited, kind, index, version)?;
    Ok(bench.put(
        &measurement_title(kind, index),
        BenchItem::Measurement(Arc::new(item)),
    ))
}

/// Put a new distance measurement on the bench with endpoint A at `endpoint`
/// and nothing else, under a label minted from where it starts, and give back
/// the bench and the label it took.
///
/// The item has no origin, so its commit appends a row; it has no endpoint B
/// and no stated distance, which the other steps give it, and its unit is the
/// one a bare stated distance is read in, the reconstruction's
/// `world_space_unit`, or metres where it names none. Its own label is
/// [`default_measurement_label`]. Its bench label is `distance from ` and the
/// form a cluster seeded at the pixel is labelled with, the image stem and the
/// rounded pixel (`distance from IMG_0042@142,198`), with the bench's
/// collision suffix where another item holds that.
///
/// # Errors
///
/// The commit's refusals of an endpoint, in its sentences, as
/// [`set_measurement_endpoint`] gives them: an image past the image set, and
/// a pixel that is not inside `[0, width) × [0, height)` of that image's
/// camera as the `float32` the file will hold.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::bench::{start_measurement, Bench, MeasurementEndpoint};
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::SfmrReconstruction;
///
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let at = MeasurementEndpoint { image: 1, pixel: [141.6, 198.2] };
/// let (bench, label) = start_measurement(&Bench::new(), &edited, at).unwrap();
/// assert!(label.starts_with("distance from ") && label.ends_with("@142,198"));
/// let item = bench.measurement(&label).unwrap();
/// assert_eq!(item.as_distance().unwrap().endpoints, [Some(at), None]);
/// assert_eq!(item.origin, None);
/// assert_eq!(item.label, "measurement 1");
/// ```
pub fn start_measurement(
    bench: &Bench,
    edited: &EditedReconstruction,
    endpoint: MeasurementEndpoint,
) -> Result<(Bench, String), EndpointPlacementRefusal> {
    check_endpoint(edited, 0, endpoint)?;
    let kind = MeasurementKind::Distance;
    let seed = ClusterSeed::from_pixel(
        endpoint.image,
        image_stem(edited, endpoint.image),
        endpoint.pixel,
        1.0,
    );
    let unit = edited
        .base
        .metadata
        .world_space_unit
        .as_deref()
        .and_then(LengthUnit::from_name)
        .unwrap_or(LengthUnit::Metre);
    let item = EditableMeasurement {
        origin: None,
        label: default_measurement_label(bench, edited),
        body: EditableMeasurementBody::Distance(EditableDistance {
            endpoints: [Some(endpoint), None],
            distance: None,
            unit,
        }),
    };
    Ok(bench.put(
        &format!("{} from {}", kind.name(), seed.label()),
        BenchItem::Measurement(Arc::new(item)),
    ))
}

/// `measurement N`, the label a new measurement starts with: `N` the smallest
/// integer from 1 that no stored distance measurement of `edited` and no
/// measurement item on `bench` has as its own label.
pub fn default_measurement_label(bench: &Bench, edited: &EditedReconstruction) -> String {
    let stored = edited
        .measurements()
        .distance
        .as_ref()
        .map_or(&[][..], |m| m.labels.as_slice());
    let taken = |label: &str| {
        stored.iter().any(|l| l == label)
            || bench
                .labels()
                .any(|item| bench.measurement(item).is_some_and(|m| m.label == label))
    };
    (1..)
        .map(|n: u64| format!("measurement {n}"))
        .find(|label| !taken(label))
        .expect("an unbounded range always has a free label")
}

/// The item with its own label set to `label`, and nothing else changed: the
/// step that sets the label the file stores for the measurement.
///
/// Labels need not be unique, and the text is kept as given, untrimmed.
///
/// # Errors
///
/// [`MeasurementLabelRefusal::Empty`] for a label that is empty or only white
/// space, which the file refuses, in the commit's sentence.
pub fn set_measurement_label(
    item: &EditableMeasurement,
    label: &str,
) -> Result<EditableMeasurement, MeasurementLabelRefusal> {
    if !is_valid_measurement_label(label) {
        return Err(MeasurementLabelRefusal::Empty);
    }
    Ok(EditableMeasurement {
        label: label.to_string(),
        ..item.clone()
    })
}

/// Why a label cannot be given to a measurement ([`set_measurement_label`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementLabelRefusal {
    /// The label is empty or only white space.
    Empty,
}

impl std::fmt::Display for MeasurementLabelRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementLabelRefusal::Empty => empty_label_sentence(f),
        }
    }
}

impl std::error::Error for MeasurementLabelRefusal {}

/// The sentence for a label that is empty or only white space, shared by
/// [`MeasurementLabelRefusal::Empty`] and
/// [`MeasurementCommitRefusal::EmptyLabel`].
fn empty_label_sentence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
        f,
        "a measurement needs a label, and this one is empty or only white space"
    )
}

/// The item with its stated distance set to `distance` in `unit`, and nothing
/// else changed: the step that sets the length a measurement states.
///
/// # Errors
///
/// [`StatedDistanceRefusal::NotPositive`] where `distance` is not finite and
/// strictly positive as the `float32` the file will hold, the rule the commit
/// asks too.
pub fn set_stated_distance(
    item: &EditableMeasurement,
    distance: f64,
    unit: LengthUnit,
) -> Result<EditableMeasurement, StatedDistanceRefusal> {
    if !stated_distance_is_positive(distance) {
        return Err(StatedDistanceRefusal::NotPositive { distance });
    }
    let EditableMeasurementBody::Distance(held) = &item.body;
    Ok(EditableMeasurement {
        body: EditableMeasurementBody::Distance(EditableDistance {
            endpoints: held.endpoints,
            distance: Some(distance),
            unit,
        }),
        ..item.clone()
    })
}

/// The item with endpoint `side` (0 for A, 1 for B) set to `endpoint`, and
/// nothing else changed: the step that moves an endpoint to another image and
/// pixel, or places one that is not placed.
///
/// Only the image and the pixel are set. The point the endpoint lands on, its
/// texel and its 3D position follow from them through the resolver, read
/// against whichever version the caller stands at; a pixel no square holds is
/// still an endpoint, as it is in the file.
///
/// # Errors
///
/// [`EndpointPlacementRefusal::NoSuchEndpoint`] for a side other than 0 and 1,
/// and the commit's own refusals of an endpoint
/// ([`MeasurementCommitRefusal::NoSuchImage`],
/// [`MeasurementCommitRefusal::PixelOutsideImage`]) in its sentences: an
/// image past the image set, and a pixel that is not inside
/// `[0, width) × [0, height)` of that image's camera as the `float32` the file
/// will hold.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::bench::{
///     set_measurement_endpoint, EditableDistance, EditableMeasurement,
///     EditableMeasurementBody, MeasurementEndpoint,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::{LengthUnit, SfmrReconstruction};
///
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let item = EditableMeasurement {
///     origin: None,
///     label: "ear to ear".to_string(),
///     body: EditableMeasurementBody::Distance(EditableDistance {
///         endpoints: [Some(MeasurementEndpoint { image: 0, pixel: [10.0, 12.5] }), None],
///         distance: Some(12.0),
///         unit: LengthUnit::Centimetre,
///     }),
/// };
/// let moved = MeasurementEndpoint { image: 1, pixel: [40.0, 30.5] };
/// let next = set_measurement_endpoint(&edited, &item, 0, moved).unwrap();
/// assert_eq!(next.as_distance().unwrap().endpoints[0], Some(moved));
/// let outside = MeasurementEndpoint { image: 1, pixel: [-1.0, 30.5] };
/// assert!(set_measurement_endpoint(&edited, &item, 0, outside).is_err());
/// ```
pub fn set_measurement_endpoint(
    edited: &EditedReconstruction,
    item: &EditableMeasurement,
    side: usize,
    endpoint: MeasurementEndpoint,
) -> Result<EditableMeasurement, EndpointPlacementRefusal> {
    if side > 1 {
        return Err(EndpointPlacementRefusal::NoSuchEndpoint { side });
    }
    check_endpoint(edited, side, endpoint)?;
    let EditableMeasurementBody::Distance(held) = &item.body;
    let mut endpoints = held.endpoints;
    endpoints[side] = Some(endpoint);
    Ok(EditableMeasurement {
        body: EditableMeasurementBody::Distance(EditableDistance { endpoints, ..*held }),
        ..item.clone()
    })
}

/// Why an endpoint cannot be set to an image and a pixel
/// ([`set_measurement_endpoint`]).
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointPlacementRefusal {
    /// The side is neither 0 (A) nor 1 (B).
    NoSuchEndpoint {
        /// The side asked for.
        side: usize,
    },
    /// The image is past the image set, or the pixel is outside the image:
    /// the commit's refusal of such an endpoint, in its sentence.
    Outside(MeasurementCommitRefusal),
}

impl std::fmt::Display for EndpointPlacementRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointPlacementRefusal::NoSuchEndpoint { side } => write!(
                f,
                "a distance has endpoints 0 (A) and 1 (B), so there is no endpoint {side}"
            ),
            EndpointPlacementRefusal::Outside(refusal) => write!(f, "{refusal}"),
        }
    }
}

impl std::error::Error for EndpointPlacementRefusal {}

impl From<MeasurementCommitRefusal> for EndpointPlacementRefusal {
    fn from(refusal: MeasurementCommitRefusal) -> Self {
        EndpointPlacementRefusal::Outside(refusal)
    }
}

/// Whether `endpoint`, on `side`, names an image of `base` and a pixel inside
/// `[0, width) × [0, height)` of its camera as the `float32` the file will
/// hold: what the commit asks of each endpoint, and what
/// [`set_measurement_endpoint`] asks of the one it sets.
fn check_endpoint(
    base: &EditedReconstruction,
    side: usize,
    endpoint: MeasurementEndpoint,
) -> Result<(), MeasurementCommitRefusal> {
    let images = &base.base.image_set;
    let image = endpoint.image;
    let no_such_image = MeasurementCommitRefusal::NoSuchImage {
        side,
        image,
        image_count: images.image_count(),
    };
    let camera = images
        .images
        .get(image as usize)
        .and_then(|i| images.cameras.get(i.camera_index as usize))
        .ok_or(no_such_image)?;
    let [x, y] = endpoint.pixel;
    // The pixel the file will hold is the `float32` one, so that is the one
    // checked against the bounds.
    let (xf, yf) = (x as f32, y as f32);
    let inside = xf.is_finite()
        && yf.is_finite()
        && xf >= 0.0
        && xf < camera.width as f32
        && yf >= 0.0
        && yf < camera.height as f32;
    if !inside {
        return Err(MeasurementCommitRefusal::PixelOutsideImage {
            side,
            image,
            pixel: endpoint.pixel,
            width: camera.width,
            height: camera.height,
        });
    }
    Ok(())
}

/// Whether `distance` is finite and strictly positive as the `float32` the
/// file stores it as.
fn stated_distance_is_positive(distance: f64) -> bool {
    let stored = distance as f32;
    stored.is_finite() && stored > 0.0
}

/// The sentence for a stated distance that is not finite and strictly
/// positive, shared by [`StatedDistanceRefusal::NotPositive`] and
/// [`MeasurementCommitRefusal::DistanceNotPositive`].
fn not_positive_sentence(f: &mut std::fmt::Formatter<'_>, distance: f64) -> std::fmt::Result {
    write!(
        f,
        "the stated distance is {distance}, which must be finite and strictly positive"
    )
}

/// Read a distance written as a number with an optional unit suffix, `12in`,
/// `30.48 cm` or `304.8`, into its value and unit.
///
/// A bare number is in `bare_unit`. The suffixes are the format's unit names
/// (`mm`, `cm`, `m`, `in`, `ft`), with or without a space before them. The
/// number is an optional sign, digits with an optional decimal point (`12`,
/// `12.`, `0.3048`, `.5`), and an optional exponent (`1e-3`). Surrounding
/// space is ignored. This is how Measurement View's stated distance field, and
/// the wire's `set_bench_measurement_distance`, read what is typed in it.
///
/// # Errors
///
/// [`StatedDistanceRefusal::Unreadable`] for text that is not such a number,
/// [`StatedDistanceRefusal::UnknownUnit`] for a suffix that names no unit, and
/// [`StatedDistanceRefusal::NotPositive`] for a value that is not finite and
/// strictly positive as a `float32`.
///
/// ```
/// use sfmtool_core::bench::parse_stated_distance;
/// use sfmtool_core::LengthUnit;
///
/// assert_eq!(
///     parse_stated_distance("12in", LengthUnit::Metre),
///     Ok((12.0, LengthUnit::Inch))
/// );
/// assert_eq!(
///     parse_stated_distance("4.2", LengthUnit::Centimetre),
///     Ok((4.2, LengthUnit::Centimetre))
/// );
/// ```
pub fn parse_stated_distance(
    text: &str,
    bare_unit: LengthUnit,
) -> Result<(f64, LengthUnit), StatedDistanceRefusal> {
    let trimmed = text.trim();
    let unreadable = || StatedDistanceRefusal::Unreadable {
        text: trimmed.to_string(),
    };
    // Split on characters, not bytes, so text holding a multi-byte character
    // is refused rather than cut inside one.
    let number = trimmed.trim_end_matches(|c: char| c.is_ascii_lowercase());
    let suffix = &trimmed[number.len()..];
    let number = number.trim_end();
    if !is_decimal_number(number) {
        return Err(unreadable());
    }
    let value: f64 = number.parse().map_err(|_| unreadable())?;
    let unit = if suffix.is_empty() {
        bare_unit
    } else {
        LengthUnit::from_name(suffix).ok_or_else(|| StatedDistanceRefusal::UnknownUnit {
            unit: suffix.to_string(),
        })?
    };
    if !stated_distance_is_positive(value) {
        return Err(StatedDistanceRefusal::NotPositive { distance: value });
    }
    Ok((value, unit))
}

/// Whether `text` is an optional sign, then digits with an optional decimal
/// point or a decimal point and digits, then an optional exponent: the
/// numbers [`parse_stated_distance`] reads, which leaves out `inf`, `nan` and
/// the other spellings Rust's own parser takes.
fn is_decimal_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    if matches!(bytes.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let digits = |from: usize| {
        bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let whole = digits(i);
    i += whole;
    let mut fraction = 0;
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        fraction = digits(i);
        i += fraction;
    }
    if whole == 0 && fraction == 0 {
        return false;
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let exponent = digits(i);
        if exponent == 0 {
            return false;
        }
        i += exponent;
    }
    i == bytes.len()
}

/// Why a stated distance cannot be read or given to a measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum StatedDistanceRefusal {
    /// The text is not a number with an optional unit suffix.
    Unreadable {
        /// The text, trimmed.
        text: String,
    },
    /// The suffix names no unit the format defines.
    UnknownUnit {
        /// The suffix.
        unit: String,
    },
    /// The distance is not finite and strictly positive as a `float32`.
    NotPositive {
        /// The distance given.
        distance: f64,
    },
}

impl std::fmt::Display for StatedDistanceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatedDistanceRefusal::Unreadable { text } => write!(
                f,
                "{text:?} is not a distance: give a number with an optional unit suffix, \
                 such as 304.8, 12in or 0.3048 m"
            ),
            StatedDistanceRefusal::UnknownUnit { unit } => write!(
                f,
                "{unit:?} is not a unit: the units are {}",
                LengthUnit::NAMES.join(", ")
            ),
            StatedDistanceRefusal::NotPositive { distance } => not_positive_sentence(f, *distance),
        }
    }
}

impl std::error::Error for StatedDistanceRefusal {}

/// Why a measurement cannot be written into the reconstruction.
///
/// Each says what is missing or wrong in a sentence a person can act on,
/// naming the endpoint as the panel does, `A` or `B`.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementCommitRefusal {
    /// The label is empty or only white space.
    EmptyLabel,
    /// The stated distance is not finite and positive.
    DistanceNotPositive {
        /// The distance given.
        distance: f64,
    },
    /// An endpoint names an image past the image set.
    NoSuchImage {
        /// 0 for endpoint A, 1 for B.
        side: usize,
        /// The image named.
        image: u32,
        /// How many images the reconstruction has.
        image_count: usize,
    },
    /// An endpoint's pixel is not finite or lies outside
    /// `[0, width) × [0, height)` of its image's camera.
    PixelOutsideImage {
        /// 0 for endpoint A, 1 for B.
        side: usize,
        /// The image.
        image: u32,
        /// The pixel.
        pixel: [f64; 2],
        /// The camera's width in pixels.
        width: u32,
        /// The camera's height in pixels.
        height: u32,
    },
    /// The row the commit was asked to replace is not there.
    NoSuchRow(NoSuchMeasurement),
}

/// `A` or `B` for an endpoint's side.
fn letter(side: usize) -> &'static str {
    if side == 0 {
        "A"
    } else {
        "B"
    }
}

impl std::fmt::Display for MeasurementCommitRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeasurementCommitRefusal::EmptyLabel => empty_label_sentence(f),
            MeasurementCommitRefusal::DistanceNotPositive { distance } => {
                not_positive_sentence(f, *distance)
            }
            MeasurementCommitRefusal::NoSuchImage {
                side,
                image,
                image_count,
            } => write!(
                f,
                "endpoint {} is in image {image}, past the reconstruction's {image_count} images",
                letter(*side)
            ),
            MeasurementCommitRefusal::PixelOutsideImage {
                side,
                image,
                pixel,
                width,
                height,
            } => write!(
                f,
                "endpoint {}'s pixel ({}, {}) is outside image {image}, which is \
                 [0, {width}) x [0, {height})",
                letter(*side),
                pixel[0],
                pixel[1]
            ),
            MeasurementCommitRefusal::NoSuchRow(missing) => write!(f, "{missing}"),
        }
    }
}

impl std::error::Error for MeasurementCommitRefusal {}

/// What [`commit_measurement`] wrote.
#[derive(Debug, Clone)]
pub struct MeasurementCommit {
    /// The reconstruction with the measurement written. Its base is the same
    /// `Arc` as the one committed into, and its point overlay the same edits
    /// ([`EditedReconstruction::with_measurements`]). Where nothing was
    /// written it is a clone of the value committed into.
    pub value: EditedReconstruction,
    /// The kind of the row written.
    pub kind: MeasurementKind,
    /// The row written, or, where nothing was written, the row that already
    /// holds the item.
    pub index: u32,
    /// Whether a stored row was replaced, rather than one appended.
    pub replaced: bool,
    /// `false` where the row already held exactly what the item holds, as the
    /// file stores it, so the caller pushes no version.
    pub wrote: bool,
}

impl MeasurementCommit {
    /// The sentence the log records, given the name the caller knows the
    /// reconstruction by: `Committed distance #2 in dino` for a replaced row,
    /// `Added distance #4 in dino` for an appended one, and a no-effect
    /// sentence where nothing was written.
    pub fn label(&self, node: &str) -> String {
        let title = measurement_title(self.kind, self.index);
        if !self.wrote {
            format!("Committed {title} in {node}: no effect, the row already holds it")
        } else if self.replaced {
            format!("Committed {title} in {node}")
        } else {
            format!("Added {title} in {node}")
        }
    }
}

/// Why `item` cannot be written into `base`, or `Ok` when it can: each
/// refusal [`commit_measurement`] makes, without building the value.
///
/// What a caller greys a commit button with, asked of the item the button
/// would commit, so the button and the step give one answer. `origin_row` is
/// the row the item's origin resolves to in `base`, as [`commit_measurement`]
/// takes it.
pub fn commit_measurement_preconditions(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<(), MeasurementCommitRefusal> {
    checked_distance(base, item, origin_row).map(|_| ())
}

/// The row `item` writes, as the file stores it, once every refusal of
/// [`commit_measurement_preconditions`] has been asked.
fn checked_distance(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<DistanceMeasurementRow, MeasurementCommitRefusal> {
    let EditableMeasurementBody::Distance(distance) = &item.body;
    if !is_valid_measurement_label(&item.label) {
        return Err(MeasurementCommitRefusal::EmptyLabel);
    }
    for (side, endpoint) in distance.endpoints.iter().enumerate() {
        if let Some(endpoint) = endpoint {
            check_endpoint(base, side, *endpoint)?;
        }
    }
    if let Some(stated) = distance.distance {
        if !stated_distance_is_positive(stated) {
            return Err(MeasurementCommitRefusal::DistanceNotPositive { distance: stated });
        }
    }
    if let Some(row) = origin_row {
        let count = item.kind().count(base.measurements());
        if row as usize >= count {
            return Err(MeasurementCommitRefusal::NoSuchRow(NoSuchMeasurement {
                kind: item.kind(),
                index: row,
                count,
            }));
        }
    }
    Ok(DistanceMeasurementRow {
        endpoints: distance.endpoints.map(|e| {
            e.map(|e| DistanceMeasurementEndpoint {
                image: e.image,
                xy: e.pixel.map(|v| v as f32),
            })
        }),
        distance: distance.distance.map(|d| d as f32),
        unit: distance.unit,
        label: item.label.clone(),
    })
}

/// Write `item` into `base`: replace row `origin_row` of its kind, or append a
/// row where `origin_row` is `None`.
///
/// The caller resolves the item's origin to a row of `base` first, as it
/// resolves a track's origin to a point. The new value replaces the
/// measurements and nothing else: it shares `base`'s base and carries its
/// point overlay across unchanged, since no point index moves
/// ([`EditedReconstruction::with_measurements`]).
///
/// A row that already holds exactly what the item holds, as the file stores
/// it (`float32` pixels and distance), is not written: the commit reports
/// `wrote: false` and the caller pushes no version.
///
/// # Errors
///
/// Every refusal [`commit_measurement_preconditions`] makes. An endpoint the
/// resolver would refuse does not stop a commit, since the pixel is what the
/// file stores either way.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::bench::{
///     commit_measurement, EditableDistance, EditableMeasurement, EditableMeasurementBody,
///     MeasurementCommitRefusal, MeasurementEndpoint,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::{LengthUnit, SfmrReconstruction};
///
/// let base = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let item = EditableMeasurement {
///     origin: None,
///     label: " ".to_string(),
///     body: EditableMeasurementBody::Distance(EditableDistance {
///         endpoints: [Some(MeasurementEndpoint { image: 0, pixel: [10.0, 12.5] }), None],
///         distance: None,
///         unit: LengthUnit::Centimetre,
///     }),
/// };
/// let refused = commit_measurement(&base, &item, None).unwrap_err();
/// assert_eq!(refused, MeasurementCommitRefusal::EmptyLabel);
/// ```
pub fn commit_measurement(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<MeasurementCommit, MeasurementCommitRefusal> {
    let row = checked_distance(base, item, origin_row)?;
    let kind = item.kind();
    let held = base.measurements().distance.as_ref();
    let mut rows: Vec<DistanceMeasurementRow> =
        held.map(DistanceMeasurementsData::rows).unwrap_or_default();
    let index = match origin_row {
        Some(at) => {
            if rows[at as usize] == row {
                return Ok(MeasurementCommit {
                    value: base.clone(),
                    kind,
                    index: at,
                    replaced: true,
                    wrote: false,
                });
            }
            rows[at as usize] = row;
            at
        }
        None => {
            rows.push(row);
            (rows.len() - 1) as u32
        }
    };
    let measurements = MeasurementsData {
        distance: Some(DistanceMeasurementsData::from_rows(&rows)),
    };
    Ok(MeasurementCommit {
        value: base.with_measurements(measurements),
        kind,
        index,
        replaced: origin_row.is_some(),
        wrote: true,
    })
}

/// Delete the stored measurement at row `index` of `kind` from `base`: a
/// direct edit of the reconstruction, as deleting a point is, and no bench
/// step.
///
/// Gives back the value, which shares `base`'s base and point overlay
/// ([`EditedReconstruction::with_measurements`]), and the row map of the
/// kind: entry `i` is where row `i` went, `None` for the deleted row. The rows
/// after it move down by one. Deleting the last row of a kind leaves the kind
/// absent, as a file with none of it stores it.
pub fn delete_measurement(
    base: &EditedReconstruction,
    kind: MeasurementKind,
    index: u32,
) -> Result<(EditedReconstruction, Vec<Option<u32>>), NoSuchMeasurement> {
    let count = kind.count(base.measurements());
    if index as usize >= count {
        return Err(NoSuchMeasurement { kind, index, count });
    }
    match kind {
        MeasurementKind::Distance => {
            let held = base
                .measurements()
                .distance
                .as_ref()
                .expect("a count above zero means the kind is present");
            let mut rows = held.rows();
            rows.remove(index as usize);
            let map = (0..count as u32)
                .map(|i| match i.cmp(&index) {
                    std::cmp::Ordering::Less => Some(i),
                    std::cmp::Ordering::Equal => None,
                    std::cmp::Ordering::Greater => Some(i - 1),
                })
                .collect();
            let measurements = MeasurementsData {
                distance: (!rows.is_empty()).then(|| DistanceMeasurementsData::from_rows(&rows)),
            };
            Ok((base.with_measurements(measurements), map))
        }
    }
}
