// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The measurements a reconstruction stores, as the viewer reads them: the
//! selection that names one, the readings every panel shares, and the text
//! the Scene tree's rows, Measurement View and the wire print.
//!
//! A measurement is named by a [`MeasurementRef`]: its node, its kind and its
//! row within that kind's section of the file. The one kind is distance
//! ([`MeasurementKind::Distance`]). Each endpoint of a distance measurement is
//! resolved against the version on screen by core's
//! `resolve_distance_measurements`, once per version: [`readings`] keeps the
//! result on the node, keyed by the version's document serial, since every
//! row, the panel and the wire read the same results and a version never
//! changes.
//!
//! A reading is always taken in the file's own coordinates. The node's display
//! transform is view state and never scales one.
//!
//! See `specs/gui/measurement-view.md`.

use std::sync::Arc;

use sfmtool_core::reconstruction::measurement_endpoint::{
    resolve_distance_measurements, texel_pixel, EndpointRefusal, EndpointResolution,
    MeasurementEndpointOptions, ResolvedDistanceMeasurement,
};
use sfmtool_core::{EditedReconstruction, LengthUnit, MeasurementsData, SfmrReconstruction};

use crate::scene::{ReconId, SceneNode};

#[cfg(test)]
pub(crate) mod fixture;
#[cfg(test)]
mod tests;

/// The kind of a stored measurement: core's, which names the subsection of the
/// file's `measurements/` section its rows are in (`title()` for a row, a
/// header and the Action Log; `name()` in a sentence and on the wire).
pub(crate) use sfmtool_core::bench::MeasurementKind;

/// One measurement of one loaded reconstruction: its node, its kind, and its
/// row within that kind's rows.
///
/// The kind is part of the reference because each kind's rows are numbered
/// within their own subsection of the file, and a later kind would otherwise
/// reuse the same indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MeasurementRef {
    pub(crate) recon: ReconId,
    pub(crate) kind: MeasurementKind,
    pub(crate) index: u32,
}

impl MeasurementRef {
    /// The distance measurement at row `index` of `recon`.
    pub(crate) fn distance(recon: ReconId, index: usize) -> Self {
        Self {
            recon,
            kind: MeasurementKind::Distance,
            index: index as u32,
        }
    }

    /// The row within its kind's rows.
    pub(crate) fn index(self) -> usize {
        self.index as usize
    }

    /// `Distance #2`: the kind and the row, as the Scene tree's row and the
    /// panel's header name it.
    pub(crate) fn title(self) -> String {
        format!("{} #{}", self.kind.title(), self.index)
    }

    /// `distance #2`, the same in the middle of a sentence, and the label a
    /// measurement put on the bench from this row is minted from.
    pub(crate) fn lower_title(self) -> String {
        sfmtool_core::bench::measurement_title(self.kind, self.index)
    }
}

/// One end of a distance measurement: endpoint 0 of the format is `A`, and
/// endpoint 1 is `B`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EndpointSide {
    A,
    B,
}

impl EndpointSide {
    /// Both sides, `A` first.
    pub(crate) const BOTH: [EndpointSide; 2] = [EndpointSide::A, EndpointSide::B];

    /// The endpoint's index in the format's `(N, 2)` columns.
    pub(crate) fn index(self) -> usize {
        match self {
            EndpointSide::A => 0,
            EndpointSide::B => 1,
        }
    }

    /// `A` or `B`.
    pub(crate) fn letter(self) -> &'static str {
        match self {
            EndpointSide::A => "A",
            EndpointSide::B => "B",
        }
    }

    /// `a` or `b`, the endpoint's name on the wire.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            EndpointSide::A => "a",
            EndpointSide::B => "b",
        }
    }

    /// The side `name` spells on the wire, or `None`.
    #[cfg_attr(not(feature = "mcp"), allow(dead_code))]
    pub(crate) fn from_wire_name(name: &str) -> Option<EndpointSide> {
        Self::BOTH.into_iter().find(|side| side.wire_name() == name)
    }
}

/// One end of a stored distance measurement: its image and its pixel there,
/// core's form of an endpoint.
pub(crate) use sfmtool_core::bench::MeasurementEndpoint as RowEndpoint;

/// One stored distance measurement, read out of the file's columns.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DistanceRow {
    /// What was measured. The file requires one, but a value built by hand
    /// can hold an empty one.
    pub(crate) label: String,
    /// The stated distance, in [`Self::unit`], or `None` for a measurement
    /// that states none.
    pub(crate) stated: Option<f64>,
    /// The unit the stated distance is in, or is given in once there is one.
    pub(crate) unit: LengthUnit,
    /// Each endpoint, `A` first, or `None` for one not placed.
    pub(crate) endpoints: [Option<RowEndpoint>; 2],
}

impl DistanceRow {
    /// The endpoint on `side`, or `None` where it is not placed.
    pub(crate) fn endpoint(&self, side: EndpointSide) -> Option<RowEndpoint> {
        self.endpoints[side.index()]
    }

    /// The image of the endpoint on `side`, which a test knows is placed.
    #[cfg(test)]
    pub(crate) fn image(&self, side: EndpointSide) -> u32 {
        self.endpoint(side).expect("a placed endpoint").image
    }

    /// The pixel of the endpoint on `side`, which a test knows is placed.
    #[cfg(test)]
    pub(crate) fn pixel(&self, side: EndpointSide) -> [f64; 2] {
        self.endpoint(side).expect("a placed endpoint").pixel
    }

    /// `12 cm`: the stated distance with its unit, or `None` where it states
    /// none.
    pub(crate) fn stated_text(&self) -> Option<String> {
        self.stated
            .map(|stated| format!("{} {}", length_number(stated), self.unit.name()))
    }
}

/// What a read-only header and a hover say for a measurement that states no
/// distance.
pub(crate) const NO_STATED_DISTANCE: &str = "No stated distance";

/// The distance measurements `measurements` holds, in stored order; empty
/// where it holds none.
///
/// A version's measurements are its own where an edit set them, so a caller
/// reads them through `EditedReconstruction::measurements` rather than off the
/// base.
pub(crate) fn distance_rows(measurements: &MeasurementsData) -> Vec<DistanceRow> {
    let Some(m) = measurements.distance.as_ref() else {
        return Vec::new();
    };
    // The format crate's rows read the absent parts as `None`, and read a
    // value built by hand whose columns disagree in length without a panic,
    // as core's readers take it.
    m.rows()
        .into_iter()
        .map(|row| DistanceRow {
            label: row.label,
            stated: row.distance.map(f64::from),
            unit: row.unit,
            endpoints: row.endpoints.map(|e| {
                e.map(|e| RowEndpoint {
                    image: e.image,
                    pixel: e.xy.map(f64::from),
                })
            }),
        })
        .collect()
}

/// How many measurements of every kind `measurements` holds.
pub(crate) fn measurement_count(measurements: &MeasurementsData) -> usize {
    measurements.distance_measurement_count()
}

/// What one version reads its measurements as: each distance measurement's
/// endpoints resolved, in stored order.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct MeasurementReadings {
    /// One per stored distance measurement.
    pub(crate) distance: Vec<ResolvedDistanceMeasurement>,
}

impl MeasurementReadings {
    /// Resolve every measurement `edited` stores, with the resolver's default
    /// options.
    pub(crate) fn of(edited: &EditedReconstruction) -> Self {
        Self {
            distance: resolve_distance_measurements(edited, &MeasurementEndpointOptions::default()),
        }
    }
}

/// The readings of the version `node` shows, resolved the first time a version
/// is asked for and kept on the node after that ([`SceneNode`]'s cache, keyed
/// by the version's document serial).
///
/// A pass over the endpoints' images' observations, well under a second, so it
/// runs on the GUI thread when a version first needs it.
pub(crate) fn readings(node: &SceneNode) -> Arc<MeasurementReadings> {
    node.measurement_readings()
}

/// What the reconstruction makes of one distance measurement: the result
/// block of Measurement View, and the reading its Scene tree row prints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DistanceResult {
    /// Both endpoints finite and no stated distance: the reconstructed distance
    /// alone, a ruler's reading.
    Unstated {
        /// The reconstructed distance, in [`Self::Unstated::unit`].
        reconstructed: f64,
        /// The file's `world_space_unit`, or `None` for scene units.
        unit: Option<LengthUnit>,
    },
    /// Both endpoints finite, and the file declares a `world_space_unit`.
    WithUnit {
        /// The reconstructed distance in the measurement's own unit.
        reconstructed: f64,
        /// `reconstructed − stated`, in the measurement's own unit.
        error: f64,
        /// [`Self::WithUnit::error`] as a percentage of the stated distance.
        error_percent: f64,
        /// `stated / reconstructed`: the factor the measurement says the
        /// reconstruction's lengths are off by.
        implied_scale: f64,
    },
    /// Both endpoints finite, and the file declares no unit.
    SceneUnits {
        /// The reconstructed distance in scene units.
        scene_distance: f64,
        /// How long one scene unit is, in the measurement's own unit:
        /// `stated / scene_distance`.
        unit_per_scene_unit: f64,
    },
    /// The endpoint on this side lies on a point at infinity (the first such
    /// side, where neither is refused).
    AtInfinity(EndpointSide),
    /// The endpoint on this side has no 3D position (the first such side,
    /// where both are placed).
    Refused(EndpointSide),
    /// The endpoint on this side is not placed (the first such side).
    NotPlaced(EndpointSide),
}

/// The result of `row` as `resolved` reads it, in a reconstruction whose file
/// declares `world_space_unit` (`None` for scene units).
///
/// An endpoint not placed is reported first, then a refusal, then a point at
/// infinity, `A` before `B` each time, since the refusal is the one a person
/// can act on. A `world_space_unit` the format does not know is read as scene
/// units. With both endpoints finite and no stated distance, the result is the
/// reconstructed distance alone, in the file's unit.
pub(crate) fn distance_result(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> DistanceResult {
    for side in EndpointSide::BOTH {
        if matches!(
            resolved.endpoints[side.index()],
            Err(EndpointRefusal::NotPlaced)
        ) {
            return DistanceResult::NotPlaced(side);
        }
    }
    for side in EndpointSide::BOTH {
        if resolved.endpoints[side.index()].is_err() {
            return DistanceResult::Refused(side);
        }
    }
    let Some(scene_distance) = resolved.scene_distance else {
        let side = EndpointSide::BOTH
            .into_iter()
            .find(|side| {
                resolved.endpoints[side.index()]
                    .as_ref()
                    .is_ok_and(|r| r.position().finite().is_none())
            })
            .unwrap_or(EndpointSide::A);
        return DistanceResult::AtInfinity(side);
    };
    let Some(stated) = row.stated else {
        return DistanceResult::Unstated {
            reconstructed: scene_distance,
            unit: world_space_unit.and_then(LengthUnit::from_name),
        };
    };
    match world_space_unit.and_then(sfmtool_core::world_space_unit_in_metres) {
        Some(metres_per_unit) => {
            let reconstructed = scene_distance * metres_per_unit / row.unit.in_metres();
            let error = reconstructed - stated;
            DistanceResult::WithUnit {
                reconstructed,
                error,
                error_percent: 100.0 * error / stated,
                implied_scale: stated / reconstructed,
            }
        }
        None => DistanceResult::SceneUnits {
            scene_distance,
            unit_per_scene_unit: stated / scene_distance,
        },
    }
}

/// `11.94 cm`, or `0.3175 su` without a unit: a ruler's reading, the
/// reconstructed distance of a measurement that states none.
pub(crate) fn unstated_text(reconstructed: f64, unit: Option<LengthUnit>, digits: i32) -> String {
    let name = unit.map_or("su", LengthUnit::name);
    format!("{} {name}", significant(reconstructed, digits))
}

/// The few words that stand for `refusal` where a row has no room for its
/// sentence: `no patch holds the pixel`.
pub(crate) fn refusal_short(refusal: &EndpointRefusal) -> &'static str {
    match refusal {
        EndpointRefusal::NotPlaced => "not placed",
        EndpointRefusal::NoSuchImage { .. } => "no such image",
        EndpointRefusal::NoPose { .. } => "the image has no pose",
        EndpointRefusal::PixelOutsideImage { .. } => "the pixel is outside the image",
        EndpointRefusal::NoPatchFrames => "no patch frames",
        EndpointRefusal::NoObservations { .. } => "no point is seen in the image",
        EndpointRefusal::NoCoveringSquare { .. } => "no patch holds the pixel",
        EndpointRefusal::AllExcluded { .. } => "every patch that holds it faces away",
    }
}

/// U+2212 MINUS SIGN, the sign a negative reading is printed with.
pub(crate) const MINUS: &str = "\u{2212}";

/// U+2192 RIGHTWARDS ARROW, between a stated distance and the
/// reconstruction's reading of it.
pub(crate) const ARROW: &str = "\u{2192}";

/// U+26A0 WARNING SIGN, in place of a reading an endpoint refused.
pub(crate) const WARNING: &str = "\u{26a0}";

/// `value` with `digits` significant digits, in plain notation.
pub(crate) fn significant(value: f64, digits: i32) -> String {
    if !value.is_finite() {
        return format!("{value}");
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (digits - 1 - magnitude).max(0) as usize;
    format!("{value:.decimals$}")
}

/// A stated length as the file holds it: `12`, `304.8`, `4.2`. The file stores
/// it as a `float32`, which prints the digits it was entered with.
pub(crate) fn length_number(value: f64) -> String {
    format!("{}", value as f32)
}

/// `value` with `decimals` decimals and its sign, the minus as [`MINUS`]:
/// `+0.7`, `−0.5`.
pub(crate) fn signed(value: f64, decimals: usize) -> String {
    let text = format!("{:.decimals$}", value.abs());
    // A value that rounds to zero is printed as `+0.0`, not `−0.0`.
    let rounds_to_zero = text.parse::<f64>().is_ok_and(|v| v == 0.0);
    if value < 0.0 && !rounds_to_zero {
        format!("{MINUS}{text}")
    } else {
        format!("+{text}")
    }
}

/// The reading a Scene tree row prints after the stated distance:
/// `→ 11.94 cm  −0.5%` with a unit, `→ 0.317 su` without one, the
/// reconstructed distance alone (`→ 11.94 cm`) for a measurement that states
/// none, `→ ∞` for an endpoint at infinity, `⚠ endpoint B: no patch holds the
/// pixel` for a refused endpoint and `endpoint B: not placed` for one not
/// placed.
pub(crate) fn reading_text(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    result: DistanceResult,
) -> String {
    match result {
        DistanceResult::WithUnit {
            reconstructed,
            error_percent,
            ..
        } => format!(
            "{ARROW} {} {}  {}%",
            significant(reconstructed, 4),
            row.unit.name(),
            signed(error_percent, 1)
        ),
        DistanceResult::SceneUnits { scene_distance, .. } => {
            format!("{ARROW} {} su", significant(scene_distance, 3))
        }
        DistanceResult::Unstated {
            reconstructed,
            unit,
        } => format!(
            "{ARROW} {}",
            unstated_text(reconstructed, unit, if unit.is_some() { 4 } else { 3 })
        ),
        DistanceResult::NotPlaced(side) => format!("endpoint {}: not placed", side.letter()),
        DistanceResult::AtInfinity(_) => format!("{ARROW} \u{221e}"),
        DistanceResult::Refused(side) => {
            let why = resolved.endpoints[side.index()]
                .as_ref()
                .err()
                .map_or("no position", refusal_short);
            format!("{WARNING} endpoint {}: {why}", side.letter())
        }
    }
}

/// A Scene tree row's whole text: `Distance #0  ear to ear  12 cm  → 11.94 cm
/// −0.5%`, the label left out where it is empty and the stated distance where
/// there is none.
pub(crate) fn row_text(
    measurement: MeasurementRef,
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> String {
    let result = distance_result(row, resolved, world_space_unit);
    let mut text = measurement.title();
    if !row.label.is_empty() {
        text.push_str("  ");
        text.push_str(&row.label);
    }
    if let Some(stated) = row.stated_text() {
        text.push_str("  ");
        text.push_str(&stated);
    }
    format!("{text}  {}", reading_text(row, resolved, result))
}

/// What the header's comparison line says: the median scale the measurements
/// of one kind with a reading imply, and how far this one's is from it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScaleComparison {
    /// How many measurements have a reading.
    pub(crate) count: usize,
    /// The median implied length of one scene unit, in this measurement's
    /// unit.
    pub(crate) median_unit_per_scene_unit: f64,
    /// This measurement's implied scale against the median, as a percentage:
    /// `+0.4` for a measurement whose stated distance is 0.4% longer than the
    /// median scale gives its reading. `None` where it has no reading or
    /// states no distance.
    pub(crate) deviation_percent: Option<f64>,
    /// Whether this measurement states a distance.
    pub(crate) states_distance: bool,
}

/// The comparison line for measurement `index` of `rows`, or `None` where
/// fewer than two have a reading.
///
/// Each measurement with a stated distance and a reading implies a length for
/// one scene unit, `stated / scene_distance` in metres; the line gives the
/// median of those (`sfmtool_core::numeric`) in `rows[index]`'s unit, and a
/// measurement that states no distance counts toward neither. It does not
/// read the file's unit, so it is the same check with one and without one.
pub(crate) fn scale_comparison(
    rows: &[DistanceRow],
    resolved: &[ResolvedDistanceMeasurement],
    index: usize,
) -> Option<ScaleComparison> {
    let implied = |i: usize| -> Option<f64> {
        let d = resolved.get(i)?.scene_distance?;
        let row = rows.get(i)?;
        let stated = row.stated?;
        (d > 0.0).then(|| stated * row.unit.in_metres() / d)
    };
    let metres_per_scene_unit: Vec<f64> = (0..rows.len()).filter_map(implied).collect();
    if metres_per_scene_unit.len() < 2 {
        return None;
    }
    let median = sfmtool_core::numeric::median(&metres_per_scene_unit);
    let row = rows.get(index)?;
    Some(ScaleComparison {
        count: metres_per_scene_unit.len(),
        median_unit_per_scene_unit: median / row.unit.in_metres(),
        deviation_percent: implied(index).map(|mine| 100.0 * (mine / median - 1.0)),
        states_distance: row.stated.is_some(),
    })
}

/// `Median over 4: 1 su = 37.8 cm; this one +0.4%`.
pub(crate) fn scale_comparison_text(line: &ScaleComparison, unit: LengthUnit) -> String {
    let mine = match line.deviation_percent {
        Some(percent) => format!("this one {}%", signed(percent, 1)),
        None if !line.states_distance => "this one states no distance".to_string(),
        None => "this one has no reading".to_string(),
    };
    format!(
        "Median over {}: 1 su = {} {}; {mine}",
        line.count,
        significant(line.median_unit_per_scene_unit, 3),
        unit.name()
    )
}

/// What the version on screen makes of a measurement item on the bench: its
/// **live readings**, read from the item rather than from any stored row.
///
/// Each placed endpoint is resolved against the version with the resolver's
/// default options, as a stored row's are, so the item and the row it came
/// from read alike until a step changes one of them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ItemReading {
    /// Each endpoint's resolution, `A` first, or `None` for one not placed.
    pub(crate) endpoints: [Option<Result<EndpointResolution, EndpointRefusal>>; 2],
    /// Both endpoints resolved together, as a stored row with the item's
    /// endpoints reads: an endpoint not placed is `EndpointRefusal::NotPlaced`
    /// there.
    pub(crate) resolved: ResolvedDistanceMeasurement,
}

/// The live readings of `item` against `edited`.
pub(crate) fn item_reading(
    edited: &EditedReconstruction,
    item: &sfmtool_core::bench::EditableMeasurement,
) -> ItemReading {
    let options = MeasurementEndpointOptions::default();
    let endpoints = item
        .as_distance()
        .map_or([None, None], |distance| distance.endpoints);
    let resolved = sfmtool_core::reconstruction::measurement_endpoint::resolve_distance_endpoints(
        edited,
        endpoints.map(|e| e.map(|e| (e.image, e.pixel))),
        &options,
    );
    ItemReading {
        endpoints: [0, 1].map(|k| endpoints[k].map(|_| resolved.endpoints[k].clone())),
        resolved,
    }
}

/// `item` as a stored row would hold it.
pub(crate) fn item_row(item: &sfmtool_core::bench::EditableMeasurement) -> DistanceRow {
    let (endpoints, stated, unit) = item
        .as_distance()
        .map_or(([None, None], None, LengthUnit::Metre), |d| {
            (d.endpoints, d.distance, d.unit)
        });
    DistanceRow {
        label: item.label.clone(),
        stated,
        unit,
        endpoints,
    }
}

/// The median line for an item: the item compared with every stored
/// measurement of its kind but the row `origin` it came from, so a person
/// sees whether a change brings it into line before committing it.
pub(crate) fn item_scale_comparison(
    rows: &[DistanceRow],
    resolved: &[ResolvedDistanceMeasurement],
    origin: Option<u32>,
    row: &DistanceRow,
    reading: &ResolvedDistanceMeasurement,
) -> Option<ScaleComparison> {
    let keep = |i: &usize| Some(*i as u32) != origin;
    let mut others: Vec<DistanceRow> = (0..rows.len())
        .filter(keep)
        .map(|i| rows[i].clone())
        .collect();
    let mut readings: Vec<ResolvedDistanceMeasurement> = (0..resolved.len())
        .filter(keep)
        .map(|i| resolved[i].clone())
        .collect();
    others.push(row.clone());
    readings.push(reading.clone());
    scale_comparison(&others, &readings, others.len() - 1)
}

/// The resolution of the endpoint on `side`, or why it has none.
pub(crate) fn endpoint(
    resolved: &ResolvedDistanceMeasurement,
    side: EndpointSide,
) -> &Result<EndpointResolution, EndpointRefusal> {
    &resolved.endpoints[side.index()]
}

/// The name of image `image` of `recon`, or `image #N` for an index past its
/// images.
pub(crate) fn image_name(recon: &SfmrReconstruction, image: u32) -> String {
    recon
        .image_set
        .images
        .get(image as usize)
        .map_or_else(|| format!("image #{image}"), |i| i.name.clone())
}

/// A point's **reference pixel**: the image and keypoint of its reference
/// observation (`tracks/reference_observations`), the pixel that stands for
/// the point where a measurement's endpoint is taken from one
/// ([`reference_pixel`]), or another pixel of that image, one that shows a
/// spot on the point's patch ([`spot_pixel`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReferencePixel {
    /// The reference observation's index in the point's track.
    pub(crate) observation: usize,
    /// Its image.
    pub(crate) image: u32,
    /// Its keypoint, the centre of its square, or for [`spot_pixel`] the
    /// pixel of that image that shows the spot.
    pub(crate) pixel: [f64; 2],
}

/// Why a point with no patch frame has no reference pixel.
pub(crate) const NO_PATCH_FOR_PIXEL: &str = "This point has no patch, so no pixel stands for it.";

/// Why a point with no reference observation has no reference pixel.
pub(crate) const NO_REFERENCE_FOR_PIXEL: &str =
    "This point has no reference observation to take a pixel from.";

/// Why a point at infinity has no reference pixel.
pub(crate) const AT_INFINITY_FOR_PIXEL: &str =
    "This point is at infinity, so no distance can be measured to it.";

/// Point `point`'s reference pixel in `edited`, or the sentence saying why it
/// has none: no patch frame (a `sift_files` reconstruction, or a point the
/// patch pass left without one), no reference observation, or a point at
/// infinity, asked in that order.
///
/// A measurement's endpoint is a pixel and a point is not, so a gesture that
/// takes an endpoint from a point takes this pixel: the keypoint is the centre
/// of the reference observation's square, so the endpoint resolves to texel
/// `(0, 0)` of the point's own patch, the point itself, where no other square
/// in that image holds the pixel and wins the ranking. The reference is the
/// one the point holds in this version, a pick only the display render made
/// included, since that is the observation its bitmap on screen was rendered
/// from. Nothing here checks that the pixel resolves back to the point.
pub(crate) fn reference_pixel(
    edited: &EditedReconstruction,
    point: u32,
) -> Result<ReferencePixel, String> {
    let view = edited
        .point(point)
        .ok_or_else(|| format!("There is no point {point} in this version."))?;
    if view.patch_frame().is_none() {
        return Err(NO_PATCH_FOR_PIXEL.to_string());
    }
    let observation = view
        .reference_observation()
        .and_then(|k| usize::try_from(k).ok())
        .filter(|&k| k < view.observations().len())
        .ok_or_else(|| NO_REFERENCE_FOR_PIXEL.to_string())?;
    if view.point().is_at_infinity() {
        return Err(AT_INFINITY_FOR_PIXEL.to_string());
    }
    // An `embedded_patches` value keeps a keypoint per observation, and only
    // such a value carries patch frames, so this is there whenever the frame
    // is; a value with frames and no inline keypoints has nothing to stand
    // for the point either.
    let [x, y] = view
        .keypoint_xy(observation)
        .ok_or_else(|| NO_PATCH_FOR_PIXEL.to_string())?;
    Ok(ReferencePixel {
        observation,
        image: view.observations()[observation].image_index,
        pixel: [f64::from(x), f64::from(y)],
    })
}

/// Where on a point the 3D view's point menu was opened: the point, and the
/// texel `(s, t)` of its patch under the pointer when the right-click landed
/// on the point's drawn square ([`texel_under_ray`]), else `None`.
///
/// Recorded on the frame the menu opens, since the entries are chosen on a
/// later frame, after the pointer has moved. The texel rather than the pixel
/// is kept, so the pixel is read against the version on screen when an entry
/// is chosen, as the reference pixel is ([`spot_pixel`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PointSpot {
    /// The point.
    pub(crate) point: crate::scene::PointRef,
    /// The texel of its patch under the pointer, or `None` for its centre.
    pub(crate) texel: Option<[f64; 2]>,
}

#[cfg(test)]
impl PointSpot {
    /// The point's centre, texel `(0, 0)`: what its reference pixel shows.
    pub(crate) fn centre(point: crate::scene::PointRef) -> Self {
        Self { point, texel: None }
    }
}

/// The pixel a measurement's endpoint is taken at for texel `texel` of point
/// `point`'s patch in `edited`: the pixel of its reference observation's image
/// where that observation's square holds the texel (core's `texel_pixel`, the
/// mapping `endpoint_views` offers its views through), or the point's
/// reference pixel, texel `(0, 0)`, where `texel` is `None` or that mapping
/// answers `None`: the texel's place is outside the reference image, or the
/// square faces away from that camera.
///
/// Refused as [`reference_pixel`] refuses: no patch frame, no reference
/// observation, or a point at infinity. Resolving the pixel lands back on the
/// point at about `texel`, unless another square in that image holds the
/// pixel and wins the ranking there; nothing here checks that.
pub(crate) fn spot_pixel(
    edited: &EditedReconstruction,
    point: u32,
    texel: Option<[f64; 2]>,
) -> Result<ReferencePixel, String> {
    let reference = reference_pixel(edited, point)?;
    let pixel = texel
        .and_then(|texel| texel_pixel(edited, point, reference.observation, texel))
        .unwrap_or(reference.pixel);
    Ok(ReferencePixel { pixel, ..reference })
}

/// The texel `(s, t)` of `patch_frame`'s square, drawn at `surfel_scale` times
/// its stored size, that the ray from `origin` along `direction` meets, all in
/// the reconstruction's own coordinates: the place on the drawn surfel the
/// pointer is over, as the texel of the bitmap drawn there.
///
/// The surfel is `c + surfel_scale · (s·u + t·v)` with its bitmap stretched
/// over it, so the bitmap's texel `(s, t)` is drawn where the ray meets that
/// square at `(s·surfel_scale, t·surfel_scale)`, and that texel is the one
/// answered. `None` where the ray meets the square nowhere in front of the eye
/// or outside it (`|s|` or `|t|` over 1), sees its back, which the viewport
/// culls, or meets its plane at under
/// [`MIN_PLANE_ANGLE_DEG`](crate::bench::geometry::MIN_PLANE_ANGLE_DEG): there
/// a pixel of pointer travel crosses more than eleven pixels' worth of the
/// square, so the texel under the pointer is not the one aimed at, which is
/// the bar the bench's handles refuse a press under. `None` too for a point at
/// infinity, which has no place to measure to, and for a scale that is not a
/// positive number.
pub(crate) fn texel_under_ray(
    patch_frame: &sfmtool_core::patch::cloud::OrientedPatch,
    origin: nalgebra::Point3<f64>,
    direction: nalgebra::Vector3<f64>,
    surfel_scale: f64,
) -> Option<[f64; 2]> {
    if patch_frame.w == 0.0 || !(surfel_scale.is_finite() && surfel_scale > 0.0) {
        return None;
    }
    let ray = direction.try_normalize(1e-12)?;
    let normal = patch_frame.normal();
    // The ray runs against the outward normal on the side the surfel is drawn
    // on; `sin` of the angle between the ray and the plane is `-ray · n`.
    // A normal that is not a number (axes of no length) fails the bar too.
    let facing = -ray.dot(&normal);
    let bar = crate::bench::geometry::MIN_PLANE_ANGLE_DEG
        .to_radians()
        .sin();
    if facing.is_nan() || facing < bar {
        return None;
    }
    let u = patch_frame.u_axis * (patch_frame.half_extent[0] * surfel_scale);
    let v = patch_frame.v_axis * (patch_frame.half_extent[1] * surfel_scale);
    let system = nalgebra::Matrix3::from_columns(&[ray, -u, -v]);
    let x = system.lu().solve(&(patch_frame.center - origin))?;
    let (lambda, s, t) = (x[0], x[1], x[2]);
    (lambda > 0.0 && s.abs() <= 1.0 && t.abs() <= 1.0).then_some([s, t])
}

/// Where the finite endpoints of `measurement` are drawn, in the shared world
/// space: each finite position put through its node's display transform, as
/// its points are. Empty for a measurement the version on screen does not
/// hold, and for one with no finite endpoint.
///
/// What a Scene tree double-click on the measurement's row frames.
pub(crate) fn world_endpoints(
    scene: &[SceneNode],
    measurement: MeasurementRef,
) -> Vec<nalgebra::Point3<f64>> {
    let Some(node) = crate::scene::node_by_id(scene, measurement.recon) else {
        return Vec::new();
    };
    let readings = readings(node);
    let Some(resolved) = readings.distance.get(measurement.index()) else {
        return Vec::new();
    };
    resolved
        .endpoints
        .iter()
        .filter_map(|e| e.as_ref().ok()?.position().finite())
        .map(|p| node.transform().apply_to_point(&nalgebra::Point3::from(p)))
        .collect()
}

/// Where each distance measurement of `recon` goes when image `removed` is
/// deleted: `None` for a row with an endpoint in it or past the image set,
/// which the subset drops, and otherwise its index among the rows that
/// survive, in stored order, as
/// `SfmrReconstruction::subset_by_image_indices` keeps them. An endpoint not
/// placed is in no image, so it drops no row.
pub(crate) fn distance_rows_after_deleting_image(
    recon: &SfmrReconstruction,
    removed: u32,
) -> Vec<Option<u32>> {
    let mut next = 0;
    // A row survives where the image of each placed endpoint does, as the
    // subset keeps it: an image past the set survives no delete.
    let count = recon.image_count() as u32;
    distance_rows(&recon.image_set.measurements)
        .iter()
        .map(|row| {
            if row
                .endpoints
                .iter()
                .flatten()
                .any(|e| e.image == removed || e.image >= count)
            {
                None
            } else {
                next += 1;
                Some(next - 1)
            }
        })
        .collect()
}
