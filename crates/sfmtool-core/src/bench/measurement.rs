// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The editable measurement: a stored measurement put on the bench to be worked
//! on, and the commit that writes it back.
//!
//! `specs/core/bench/editable-measurement.md` is the design. An
//! [`EditableMeasurement`] holds only what the file stores of a measurement:
//! each endpoint's image and pixel, the stated distance and its unit, and the
//! label. Everything else about an endpoint (its point, its texel, its 3D
//! position) is read again by the resolver every time
//! ([`resolve_distance_endpoints`](crate::reconstruction::measurement_endpoint::resolve_distance_endpoints)).
//!
//! Every function here is pure, as the track's steps are: [`put_measurement`]
//! gives back the next bench, [`commit_measurement`] the next reconstruction
//! value, and [`delete_measurement`] the value with one row fewer, which is a
//! direct edit of the reconstruction rather than a bench step.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use crate::reconstruction::edited::EditedReconstruction;
use crate::{DistanceMeasurementsData, LengthUnit, MeasurementsData};

use super::{Bench, BenchItem};

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
    /// The stated length and its unit, once given.
    pub distance: Option<(f64, LengthUnit)>,
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
    /// The measurement's own label, written into the file on commit. It may be
    /// empty and need not be unique, so it is not the item's bench label.
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
                let endpoint = |k: usize| MeasurementEndpoint {
                    image: m.image_indexes[[row, k]],
                    pixel: [
                        f64::from(m.endpoints_xy[[row, k, 0]]),
                        f64::from(m.endpoints_xy[[row, k, 1]]),
                    ],
                };
                // A load refuses a unit code past the legend, so the fallback
                // is for a value built by hand.
                let unit = LengthUnit::from_code(m.units[row]).unwrap_or(LengthUnit::Metre);
                EditableMeasurementBody::Distance(EditableDistance {
                    endpoints: [Some(endpoint(0)), Some(endpoint(1))],
                    distance: Some((f64::from(m.distances[row]), unit)),
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
                distance: distance.distance,
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

/// Why a measurement cannot be written into the reconstruction.
///
/// Each says what is missing or wrong in a sentence a person can act on,
/// naming the endpoint as the panel does, `A` or `B`.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementCommitRefusal {
    /// The endpoint on this side (0 for A, 1 for B) is not placed.
    EndpointNotPlaced {
        /// 0 for endpoint A, 1 for B.
        side: usize,
    },
    /// No stated distance is given.
    NoStatedDistance,
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
            MeasurementCommitRefusal::EndpointNotPlaced { side } => write!(
                f,
                "endpoint {} is not placed, and a distance needs both endpoints",
                letter(*side)
            ),
            MeasurementCommitRefusal::NoStatedDistance => {
                write!(f, "no stated distance is given")
            }
            MeasurementCommitRefusal::DistanceNotPositive { distance } => write!(
                f,
                "the stated distance is {distance}, which must be finite and strictly positive"
            ),
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

/// The endpoints and the stated distance of `item`, once every refusal of
/// [`commit_measurement_preconditions`] has been asked.
fn checked_distance(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<([MeasurementEndpoint; 2], f64, LengthUnit), MeasurementCommitRefusal> {
    let EditableMeasurementBody::Distance(distance) = &item.body;
    let images = &base.base.image_set;
    let mut placed = [MeasurementEndpoint {
        image: 0,
        pixel: [0.0; 2],
    }; 2];
    for (side, endpoint) in distance.endpoints.iter().enumerate() {
        let endpoint = endpoint.ok_or(MeasurementCommitRefusal::EndpointNotPlaced { side })?;
        placed[side] = endpoint;
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
        // The pixel the file will hold is the `float32` one, so that is the
        // one checked against the bounds.
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
    }
    let (stated, unit) = distance
        .distance
        .ok_or(MeasurementCommitRefusal::NoStatedDistance)?;
    let stored = stated as f32;
    if !(stored.is_finite() && stored > 0.0) {
        return Err(MeasurementCommitRefusal::DistanceNotPositive { distance: stated });
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
    Ok((placed, stated, unit))
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
///     label: "ear to ear".to_string(),
///     body: EditableMeasurementBody::Distance(EditableDistance {
///         endpoints: [Some(MeasurementEndpoint { image: 0, pixel: [10.0, 12.5] }), None],
///         distance: Some((12.0, LengthUnit::Centimetre)),
///     }),
/// };
/// let refused = commit_measurement(&base, &item, None).unwrap_err();
/// assert_eq!(refused, MeasurementCommitRefusal::EndpointNotPlaced { side: 1 });
/// ```
pub fn commit_measurement(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<MeasurementCommit, MeasurementCommitRefusal> {
    let ([a, b], stated, unit) = checked_distance(base, item, origin_row)?;
    let kind = item.kind();
    let row = DistanceRow {
        distance: stated as f32,
        unit: unit.code(),
        images: [a.image, b.image],
        pixels: [a.pixel, b.pixel].map(|p| [p[0] as f32, p[1] as f32]),
        label: item.label.clone(),
    };
    let held = base.measurements().distance.as_ref();
    let mut rows: Vec<DistanceRow> = held.map(distance_rows_of).unwrap_or_default();
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
        distance: Some(distance_data(&rows)),
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
            let mut rows = distance_rows_of(held);
            rows.remove(index as usize);
            let map = (0..count as u32)
                .map(|i| match i.cmp(&index) {
                    std::cmp::Ordering::Less => Some(i),
                    std::cmp::Ordering::Equal => None,
                    std::cmp::Ordering::Greater => Some(i - 1),
                })
                .collect();
            let measurements = MeasurementsData {
                distance: (!rows.is_empty()).then(|| distance_data(&rows)),
            };
            Ok((base.with_measurements(measurements), map))
        }
    }
}

/// One distance row as the file stores it.
#[derive(Debug, Clone, PartialEq)]
struct DistanceRow {
    distance: f32,
    unit: u8,
    images: [u32; 2],
    pixels: [[f32; 2]; 2],
    label: String,
}

/// The rows of `m`, in stored order.
fn distance_rows_of(m: &DistanceMeasurementsData) -> Vec<DistanceRow> {
    (0..m.len())
        .map(|i| DistanceRow {
            distance: m.distances[i],
            unit: m.units[i],
            images: [m.image_indexes[[i, 0]], m.image_indexes[[i, 1]]],
            pixels: [0, 1].map(|k| [m.endpoints_xy[[i, k, 0]], m.endpoints_xy[[i, k, 1]]]),
            label: m.labels.get(i).cloned().unwrap_or_default(),
        })
        .collect()
}

/// The columns that hold `rows`, which must not be empty.
fn distance_data(rows: &[DistanceRow]) -> DistanceMeasurementsData {
    let n = rows.len();
    DistanceMeasurementsData {
        distances: Array1::from_iter(rows.iter().map(|r| r.distance)),
        units: Array1::from_iter(rows.iter().map(|r| r.unit)),
        image_indexes: Array2::from_shape_fn((n, 2), |(i, k)| rows[i].images[k]),
        endpoints_xy: Array3::from_shape_fn((n, 2, 2), |(i, k, c)| rows[i].pixels[k][c]),
        labels: rows.iter().map(|r| r.label.clone()).collect(),
    }
}
