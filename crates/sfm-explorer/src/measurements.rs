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
    resolve_distance_measurements, EndpointRefusal, EndpointResolution, MeasurementEndpointOptions,
    ResolvedDistanceMeasurement,
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

/// One stored distance measurement, read out of the file's columns.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DistanceRow {
    /// What was measured; empty for an unlabelled measurement.
    pub(crate) label: String,
    /// The stated distance, in [`Self::unit`].
    pub(crate) stated: f64,
    /// The unit the stated distance is in.
    pub(crate) unit: LengthUnit,
    /// The image of each endpoint, `A` first.
    pub(crate) images: [u32; 2],
    /// The pixel of each endpoint in its image, `A` first.
    pub(crate) pixels: [[f64; 2]; 2],
}

impl DistanceRow {
    /// The image of the endpoint on `side`.
    pub(crate) fn image(&self, side: EndpointSide) -> u32 {
        self.images[side.index()]
    }

    /// The pixel of the endpoint on `side`.
    pub(crate) fn pixel(&self, side: EndpointSide) -> [f64; 2] {
        self.pixels[side.index()]
    }

    /// `12 cm`: the stated distance with its unit.
    pub(crate) fn stated_text(&self) -> String {
        format!("{} {}", length_number(self.stated), self.unit.name())
    }
}

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
    (0..m.len())
        .map(|i| DistanceRow {
            label: m.labels.get(i).cloned().unwrap_or_default(),
            stated: f64::from(m.distances[i]),
            unit: LengthUnit::from_code(m.units[i]).unwrap_or(LengthUnit::Metre),
            images: [m.image_indexes[[i, 0]], m.image_indexes[[i, 1]]],
            pixels: [0, 1].map(|k| {
                [
                    f64::from(m.endpoints_xy[[i, k, 0]]),
                    f64::from(m.endpoints_xy[[i, k, 1]]),
                ]
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
    /// The endpoint on this side has no 3D position (the first such side).
    Refused(EndpointSide),
}

/// The result of `row` as `resolved` reads it, in a reconstruction whose file
/// declares `world_space_unit` (`None` for scene units).
///
/// A refusal is reported before a point at infinity, `A` before `B`, since the
/// refusal is the one a person can act on. A `world_space_unit` the format
/// does not know is read as scene units.
pub(crate) fn distance_result(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> DistanceResult {
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
    match world_space_unit.and_then(sfmtool_core::world_space_unit_in_metres) {
        Some(metres_per_unit) => {
            let reconstructed = scene_distance * metres_per_unit / row.unit.in_metres();
            let error = reconstructed - row.stated;
            DistanceResult::WithUnit {
                reconstructed,
                error,
                error_percent: 100.0 * error / row.stated,
                implied_scale: row.stated / reconstructed,
            }
        }
        None => DistanceResult::SceneUnits {
            scene_distance,
            unit_per_scene_unit: row.stated / scene_distance,
        },
    }
}

/// The few words that stand for `refusal` where a row has no room for its
/// sentence: `no patch holds the pixel`.
pub(crate) fn refusal_short(refusal: &EndpointRefusal) -> &'static str {
    match refusal {
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
/// `→ 11.94 cm  −0.5%` with a unit, `→ 0.317 su` without one, `→ ∞` for an
/// endpoint at infinity, and `⚠ endpoint B: no patch holds the pixel` for a
/// refused endpoint.
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
/// −0.5%`, the label left out where it is empty.
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
    format!(
        "{text}  {}  {}",
        row.stated_text(),
        reading_text(row, resolved, result)
    )
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
    /// median scale gives its reading. `None` where it has no reading.
    pub(crate) deviation_percent: Option<f64>,
}

/// The comparison line for measurement `index` of `rows`, or `None` where
/// fewer than two have a reading.
///
/// Each reading implies a length for one scene unit, `stated / scene_distance`
/// in metres; the line gives the median of those (`sfmtool_core::numeric`)
/// in `rows[index]`'s unit. It does not read the file's unit, so it is the
/// same check with one and without one.
pub(crate) fn scale_comparison(
    rows: &[DistanceRow],
    resolved: &[ResolvedDistanceMeasurement],
    index: usize,
) -> Option<ScaleComparison> {
    let implied = |i: usize| -> Option<f64> {
        let d = resolved.get(i)?.scene_distance?;
        let row = rows.get(i)?;
        (d > 0.0).then(|| row.stated * row.unit.in_metres() / d)
    };
    let metres_per_scene_unit: Vec<f64> = (0..rows.len()).filter_map(implied).collect();
    if metres_per_scene_unit.len() < 2 {
        return None;
    }
    let median = sfmtool_core::numeric::median(&metres_per_scene_unit);
    let unit = rows.get(index)?.unit;
    Some(ScaleComparison {
        count: metres_per_scene_unit.len(),
        median_unit_per_scene_unit: median / unit.in_metres(),
        deviation_percent: implied(index).map(|mine| 100.0 * (mine / median - 1.0)),
    })
}

/// `Median over 4: 1 su = 37.8 cm; this one +0.4%`.
pub(crate) fn scale_comparison_text(line: &ScaleComparison, unit: LengthUnit) -> String {
    let mine = match line.deviation_percent {
        Some(percent) => format!("this one {}%", signed(percent, 1)),
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
    /// Both endpoints resolved together, where both are placed: what a
    /// stored row with the item's endpoints would read as.
    pub(crate) resolved: Option<ResolvedDistanceMeasurement>,
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
    match endpoints {
        [Some(a), Some(b)] => {
            let resolved = sfmtool_core::reconstruction::measurement_endpoint::resolve_distance_endpoints(
                edited,
                [(a.image, a.pixel), (b.image, b.pixel)],
                &options,
            );
            ItemReading {
                endpoints: [
                    Some(resolved.endpoints[0].clone()),
                    Some(resolved.endpoints[1].clone()),
                ],
                resolved: Some(resolved),
            }
        }
        partial => ItemReading {
            endpoints: partial.map(|endpoint| {
                endpoint.map(|e| {
                    sfmtool_core::reconstruction::measurement_endpoint::resolve_measurement_endpoint(
                        edited, e.image, e.pixel, &options,
                    )
                })
            }),
            resolved: None,
        },
    }
}

/// `item` as a stored row would hold it, or `None` where an endpoint is not
/// placed or no stated distance is given.
pub(crate) fn item_row(item: &sfmtool_core::bench::EditableMeasurement) -> Option<DistanceRow> {
    let distance = item.as_distance()?;
    let [Some(a), Some(b)] = distance.endpoints else {
        return None;
    };
    let (stated, unit) = distance.distance?;
    Some(DistanceRow {
        label: item.label.clone(),
        stated,
        unit,
        images: [a.image, b.image],
        pixels: [a.pixel, b.pixel],
    })
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
/// deleted: `None` for a row with an endpoint in it, which the subset drops,
/// and otherwise its index among the rows that survive, in stored order, as
/// `SfmrReconstruction::subset_by_image_indices` keeps them.
pub(crate) fn distance_rows_after_deleting_image(
    recon: &SfmrReconstruction,
    removed: u32,
) -> Vec<Option<u32>> {
    let mut next = 0;
    distance_rows(&recon.image_set.measurements)
        .iter()
        .map(|row| {
            if row.images.contains(&removed) {
                None
            } else {
                next += 1;
                Some(next - 1)
            }
        })
        .collect()
}
