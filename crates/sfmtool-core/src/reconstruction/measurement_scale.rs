// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The one uniform scale factor that puts a reconstruction in a physical
//! length unit, read from the distance measurements it stores: one factor per
//! usable measurement, the stated distance in the unit over the distance its
//! resolved endpoints lie apart, and their median, with the statistics that
//! show how well the measurements agree.
//!
//! `specs/core/reconstruction/measurement-scale.md` is the design.

use crate::numeric;
use crate::reconstruction::measurement_endpoint::{EndpointRefusal, ResolvedDistanceMeasurement};
use crate::{LengthUnit, MeasurementsData};

#[cfg(test)]
mod tests;

/// The deviation from the median, as a percentage of it, beyond which the
/// measurements are said to disagree ([`ScaleFactorStatistics::disagrees`]):
/// the bar `sfm xform --scale-to-median-measurement` warns at.
pub const SCALE_DISAGREEMENT_PERCENT: f64 = 5.0;

/// The scale factor one usable measurement implies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementScaleFactor {
    /// The measurement's index in stored order.
    pub index: usize,
    /// The stated distance, converted to the target unit.
    pub stated: f64,
    /// The distance between the two resolved endpoints, in scene units.
    pub scene_distance: f64,
    /// `stated / scene_distance`: how many of the target unit one scene unit
    /// is, by this measurement.
    pub factor: f64,
    /// `100 · (factor / median − 1)`: how far this factor lies from the median
    /// of all of them, signed, as a percentage of the median.
    pub deviation_percent: f64,
}

/// Why a stored distance measurement gives no scale factor.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementScaleExclusion {
    /// An endpoint has no position. `endpoint` is `0` for `A` and `1` for `B`,
    /// the first refused one.
    EndpointRefused {
        /// The endpoint, `0` or `1`.
        endpoint: usize,
        /// Why the resolver refused it.
        refusal: EndpointRefusal,
    },
    /// An endpoint lies on a point at infinity, so no distance can be measured
    /// to it: the first such endpoint, where neither is refused.
    EndpointAtInfinity {
        /// The endpoint, `0` or `1`.
        endpoint: usize,
    },
    /// The two endpoints are a scene distance apart that is zero or not
    /// finite, so the ratio is no scale.
    UnusableSceneDistance {
        /// The distance, in scene units.
        scene_distance: f64,
    },
    /// Both endpoints have finite positions but the reading holds no scene
    /// distance. The resolver always gives one there, so only a reading built
    /// by hand holds this.
    NoSceneDistance,
    /// The stated distance is not finite and strictly positive, or its unit
    /// code names no unit. A load refuses such a row, so only a value built by
    /// hand holds one.
    UnusableStatedDistance,
    /// The stated distance and the scene distance are each usable, but their
    /// ratio overflows to infinity or underflows to zero, which is no scale.
    UnusableFactor {
        /// The ratio, `0` or infinite.
        factor: f64,
    },
}

impl std::fmt::Display for MeasurementScaleExclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let letter = |endpoint: usize| if endpoint == 0 { "A" } else { "B" };
        match self {
            Self::EndpointRefused { endpoint, refusal } => {
                write!(
                    f,
                    "endpoint {} has no position: {refusal}",
                    letter(*endpoint)
                )
            }
            Self::EndpointAtInfinity { endpoint } => write!(
                f,
                "endpoint {} lies on a point at infinity, so no distance can be measured to it",
                letter(*endpoint)
            ),
            Self::UnusableSceneDistance { scene_distance } => write!(
                f,
                "its endpoints are {scene_distance} scene units apart, which gives no scale"
            ),
            Self::NoSceneDistance => write!(
                f,
                "its reading holds no distance between its two finite endpoints"
            ),
            Self::UnusableStatedDistance => write!(
                f,
                "its stated distance is not a positive length in a known unit"
            ),
            Self::UnusableFactor { factor } => write!(
                f,
                "its stated distance over its scene distance is {factor}, which gives no scale"
            ),
        }
    }
}

/// One stored distance measurement that gives no scale factor, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct ExcludedMeasurement {
    /// The measurement's index in stored order.
    pub index: usize,
    /// Why it was left out.
    pub reason: MeasurementScaleExclusion,
}

/// How the usable measurements' scale factors spread around their median.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleFactorStatistics {
    /// How many measurements give a factor, at least `1`.
    pub count: usize,
    /// The median factor ([`numeric::median`]), the one a caller scales by.
    pub median: f64,
    /// The mean factor.
    pub mean: f64,
    /// The sample standard deviation (`n − 1` in the denominator); `None` for
    /// a single measurement, which has no spread to estimate.
    pub sample_std: Option<f64>,
    /// [`Self::sample_std`] as a percentage of [`Self::median`].
    pub sample_std_percent: Option<f64>,
    /// The median of `|factor − median|` over the factors: `0` for a single
    /// measurement.
    pub median_absolute_deviation: f64,
    /// The smallest factor.
    pub min: f64,
    /// The largest factor.
    pub max: f64,
    /// The largest `|deviation_percent|` of any measurement.
    pub max_abs_deviation_percent: f64,
    /// Whether [`Self::max_abs_deviation_percent`] exceeds
    /// [`SCALE_DISAGREEMENT_PERCENT`].
    pub disagrees: bool,
}

/// The scale the stored distance measurements give a reconstruction in one
/// length unit ([`distance_measurement_scale`]).
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceMeasurementScale {
    /// The unit the factors convert scene units to.
    pub unit: LengthUnit,
    /// One per usable measurement, in stored order.
    pub factors: Vec<MeasurementScaleFactor>,
    /// One per measurement that gives no factor, in stored order.
    pub excluded: Vec<ExcludedMeasurement>,
    /// The spread of [`Self::factors`].
    pub statistics: ScaleFactorStatistics,
}

impl DistanceMeasurementScale {
    /// The factor to scale the reconstruction's coordinates by, the median of
    /// the measurements' factors: after it, scene units are [`Self::unit`].
    pub fn factor(&self) -> f64 {
        self.statistics.median
    }
}

/// Why the stored distance measurements give no scale.
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMeasurementScaleError {
    /// The reconstruction stores no distance measurement.
    NoMeasurements,
    /// The readings are not one per stored measurement: they were resolved
    /// against another version's measurements.
    ReadingCountMismatch {
        /// How many distance measurements are stored.
        measurements: usize,
        /// How many readings were given.
        readings: usize,
    },
    /// Every stored measurement was excluded.
    NoUsableMeasurement {
        /// Each measurement, with why it gives no factor.
        excluded: Vec<ExcludedMeasurement>,
    },
}

impl std::fmt::Display for DistanceMeasurementScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMeasurements => write!(
                f,
                "the reconstruction stores no distance measurement, so no stated length gives \
                 it a scale"
            ),
            Self::ReadingCountMismatch {
                measurements,
                readings,
            } => write!(
                f,
                "{readings} readings were given for {measurements} distance measurements; \
                 resolve the measurements of the same version"
            ),
            Self::NoUsableMeasurement { excluded } => {
                write!(
                    f,
                    "none of the {} distance measurements gives a scale",
                    excluded.len()
                )?;
                for e in excluded {
                    write!(f, "; measurement {}: {}", e.index, e.reason)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for DistanceMeasurementScaleError {}

/// The uniform scale factor that puts a reconstruction in `unit`, from the
/// distance measurements `measurements` stores and their `readings`, one per
/// stored measurement in stored order, as
/// [`resolve_distance_measurements`](crate::reconstruction::measurement_endpoint::resolve_distance_measurements)
/// gives them.
///
/// Each usable measurement gives `f = stated / scene_distance`, its stated
/// distance converted to `unit` over the distance its endpoints lie apart in
/// the file's own coordinates. The factor is the median of those, so
/// multiplying the reconstruction's coordinates by it puts them in `unit`,
/// whatever `world_space_unit` the file declares before. A measurement with
/// an endpoint refused, an endpoint at infinity, a scene distance that is
/// zero or not finite, or a factor that overflows to infinity or underflows
/// to zero is excluded, with its reason.
///
/// # Errors
///
/// [`DistanceMeasurementScaleError::NoMeasurements`] where none is stored,
/// [`DistanceMeasurementScaleError::ReadingCountMismatch`] where `readings`
/// is not one per stored measurement, and
/// [`DistanceMeasurementScaleError::NoUsableMeasurement`] where every one is
/// excluded.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::reconstruction::measurement_endpoint::{
///     resolve_distance_measurements, MeasurementEndpointOptions,
/// };
/// use sfmtool_core::reconstruction::measurement_scale::{
///     distance_measurement_scale, DistanceMeasurementScaleError,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::{LengthUnit, SfmrReconstruction};
///
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let readings = resolve_distance_measurements(&edited, &MeasurementEndpointOptions::default());
/// match distance_measurement_scale(edited.measurements(), &readings, LengthUnit::Millimetre) {
///     Ok(scale) => println!("scale by {} to millimetres", scale.factor()),
///     Err(e) => assert_eq!(e, DistanceMeasurementScaleError::NoMeasurements),
/// }
/// ```
pub fn distance_measurement_scale(
    measurements: &MeasurementsData,
    readings: &[ResolvedDistanceMeasurement],
    unit: LengthUnit,
) -> Result<DistanceMeasurementScale, DistanceMeasurementScaleError> {
    let Some(m) = measurements.distance.as_ref() else {
        return Err(DistanceMeasurementScaleError::NoMeasurements);
    };
    if m.is_empty() {
        return Err(DistanceMeasurementScaleError::NoMeasurements);
    }
    if readings.len() != m.len() {
        return Err(DistanceMeasurementScaleError::ReadingCountMismatch {
            measurements: m.len(),
            readings: readings.len(),
        });
    }

    let mut factors = Vec::new();
    let mut excluded = Vec::new();
    for (index, reading) in readings.iter().enumerate() {
        match usable(m, index, reading, unit) {
            Ok((stated, scene_distance, factor)) => factors.push(MeasurementScaleFactor {
                index,
                stated,
                scene_distance,
                factor,
                deviation_percent: 0.0,
            }),
            Err(reason) => excluded.push(ExcludedMeasurement { index, reason }),
        }
    }
    if factors.is_empty() {
        return Err(DistanceMeasurementScaleError::NoUsableMeasurement { excluded });
    }

    let values: Vec<f64> = factors.iter().map(|f| f.factor).collect();
    let median = numeric::median(&values);
    for f in &mut factors {
        f.deviation_percent = 100.0 * (f.factor / median - 1.0);
    }
    let n = values.len();
    // Each term divided first, so factors near `f64::MAX` do not overflow the
    // sum.
    let mean = values.iter().map(|v| v / n as f64).sum::<f64>();
    let sample_std = (n > 1)
        .then(|| (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt());
    let absolute_deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
    let max_abs_deviation_percent = factors
        .iter()
        .map(|f| f.deviation_percent.abs())
        .fold(0.0, f64::max);
    let statistics = ScaleFactorStatistics {
        count: n,
        median,
        mean,
        sample_std,
        sample_std_percent: sample_std.map(|s| 100.0 * s / median),
        median_absolute_deviation: numeric::median(&absolute_deviations),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        max_abs_deviation_percent,
        disagrees: max_abs_deviation_percent > SCALE_DISAGREEMENT_PERCENT,
    };
    Ok(DistanceMeasurementScale {
        unit,
        factors,
        excluded,
        statistics,
    })
}

/// Measurement `index`'s stated distance in `unit` and its scene distance, or
/// why it gives no factor. A refusal is reported before a point at infinity,
/// `A` before `B`, as Measurement View reports them.
fn usable(
    m: &crate::DistanceMeasurementsData,
    index: usize,
    reading: &ResolvedDistanceMeasurement,
    unit: LengthUnit,
) -> Result<(f64, f64, f64), MeasurementScaleExclusion> {
    for (endpoint, resolved) in reading.endpoints.iter().enumerate() {
        if let Err(refusal) = resolved {
            return Err(MeasurementScaleExclusion::EndpointRefused {
                endpoint,
                refusal: refusal.clone(),
            });
        }
    }
    let at_infinity = reading
        .endpoints
        .iter()
        .position(|e| e.as_ref().is_ok_and(|r| r.position().finite().is_none()));
    if let Some(endpoint) = at_infinity {
        return Err(MeasurementScaleExclusion::EndpointAtInfinity { endpoint });
    }
    let Some(scene_distance) = reading.scene_distance else {
        return Err(MeasurementScaleExclusion::NoSceneDistance);
    };
    if !(scene_distance.is_finite() && scene_distance > 0.0) {
        return Err(MeasurementScaleExclusion::UnusableSceneDistance { scene_distance });
    }
    let stated = m.distances.get(index).map(|&d| f64::from(d));
    let stated_unit = m.units.get(index).and_then(|&c| LengthUnit::from_code(c));
    let stated = match (stated, stated_unit) {
        (Some(d), Some(u)) if d.is_finite() && d > 0.0 => d * u.in_metres() / unit.in_metres(),
        _ => return Err(MeasurementScaleExclusion::UnusableStatedDistance),
    };
    // Both are finite and positive, so the ratio is never NaN, but it can
    // overflow to infinity or underflow to zero.
    let factor = stated / scene_distance;
    if !(factor.is_finite() && factor > 0.0) {
        return Err(MeasurementScaleExclusion::UnusableFactor { factor });
    }
    Ok((stated, scene_distance, factor))
}
