// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scale from stored distance measurements: mostly on readings built by
//! hand, since the operation reads only their scene distances and endpoint
//! outcomes, and once end to end on the endpoint resolver's fixture.

use std::sync::Arc;

use nalgebra::Vector3;
use ndarray::{Array1, Array2, Array3};

use crate::geometry::{RotQuaternion, Se3Transform};
use crate::reconstruction::measurement_endpoint::tests::{pixel_of_texel, pt, scene};
use crate::reconstruction::measurement_endpoint::{
    resolve_distance_measurements, CandidateStanding, EndpointCandidate, EndpointPosition,
    EndpointRefusal, EndpointResolution, MeasurementEndpointOptions,
};
use crate::reconstruction::EditedReconstruction;
use crate::{
    DistanceMeasurementsData, LengthUnit, MeasurementsData, LENGTH_UNIT_CM, LENGTH_UNIT_IN,
    LENGTH_UNIT_M, LENGTH_UNIT_MM,
};

use super::*;

/// An endpoint resolved to `position`.
fn at(position: EndpointPosition) -> Result<EndpointResolution, EndpointRefusal> {
    Ok(EndpointResolution {
        chosen: 0,
        candidates: vec![EndpointCandidate {
            point: 0,
            observation: 0,
            patch_coords: [0.0, 0.0],
            position,
            depth: 1.0,
            standing: CandidateStanding::Alone,
        }],
    })
}

fn finite(x: f64) -> Result<EndpointResolution, EndpointRefusal> {
    at(EndpointPosition::Finite {
        position: [x, 0.0, 0.0],
    })
}

fn sky() -> Result<EndpointResolution, EndpointRefusal> {
    at(EndpointPosition::AtInfinity {
        direction: [0.0, 0.0, 1.0],
    })
}

/// Two finite endpoints `d` scene units apart.
fn apart(d: f64) -> ResolvedDistanceMeasurement {
    ResolvedDistanceMeasurement {
        endpoints: [finite(0.0), finite(d)],
        scene_distance: Some(d),
    }
}

/// Stored distance measurements stating `rows`, each a distance and a unit
/// code; the endpoints are placeholders, since only the readings are read.
fn stored(rows: &[(f32, u8)]) -> MeasurementsData {
    let n = rows.len();
    MeasurementsData {
        distance: Some(DistanceMeasurementsData {
            distances: rows.iter().map(|r| r.0).collect::<Array1<f32>>(),
            units: rows.iter().map(|r| r.1).collect::<Array1<u8>>(),
            image_indexes: Array2::zeros((n, 2)),
            endpoints_xy: Array3::zeros((n, 2, 2)),
            labels: vec![String::new(); n],
        }),
    }
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} against {b}");
}

/// Stated distances in centimetres, inches and metres, read in millimetres:
/// each converted through metres, the median taken over the three factors,
/// and every statistic of 297, 300 and 303.
#[test]
fn mixed_units_convert_to_the_target_unit() {
    let measurements = stored(&[
        (12.0, LENGTH_UNIT_CM),
        (4.8, LENGTH_UNIT_IN),
        (0.15, LENGTH_UNIT_M),
    ]);
    let inch_mm = f64::from(4.8_f32) * 25.4;
    let cm_mm = f64::from(12.0_f32) * 10.0;
    let m_mm = f64::from(0.15_f32) * 1000.0;
    let readings = [
        apart(cm_mm / 300.0),
        apart(inch_mm / 303.0),
        apart(m_mm / 297.0),
    ];
    let scale = distance_measurement_scale(&measurements, &readings, LengthUnit::Millimetre)
        .expect("three usable measurements");
    assert_eq!(scale.unit, LengthUnit::Millimetre);
    assert!(scale.excluded.is_empty());
    assert_eq!(
        scale.factors.iter().map(|f| f.index).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for (f, stated) in scale.factors.iter().zip([cm_mm, inch_mm, m_mm]) {
        assert_close(f.stated, stated, 1e-9, "stated in mm");
    }
    for (f, expected) in scale.factors.iter().zip([300.0, 303.0, 297.0]) {
        assert_close(f.factor, expected, 1e-9, "factor");
    }
    let s = &scale.statistics;
    assert_eq!(s.count, 3);
    assert_close(scale.factor(), 300.0, 1e-9, "factor");
    assert_close(s.median, 300.0, 1e-9, "median");
    assert_close(s.mean, 300.0, 1e-9, "mean");
    assert_close(s.sample_std.unwrap(), 3.0, 1e-9, "sample std");
    assert_close(s.sample_std_percent.unwrap(), 1.0, 1e-9, "std percent");
    assert_close(s.median_absolute_deviation, 3.0, 1e-9, "MAD");
    assert_close(s.min, 297.0, 1e-9, "min");
    assert_close(s.max, 303.0, 1e-9, "max");
    let deviations: Vec<f64> = scale.factors.iter().map(|f| f.deviation_percent).collect();
    for (d, expected) in deviations.iter().zip([0.0, 1.0, -1.0]) {
        assert_close(*d, expected, 1e-9, "deviation percent");
    }
    assert_close(s.max_abs_deviation_percent, 1.0, 1e-9, "max deviation");
    assert!(!s.disagrees);

    // The same readings in centimetres give a tenth of the factor.
    let in_cm =
        distance_measurement_scale(&measurements, &readings, LengthUnit::Centimetre).unwrap();
    assert_close(in_cm.factor(), 30.0, 1e-9, "factor in cm");
}

/// A factor more than 5% from the median makes the measurements disagree;
/// factors within it do not.
#[test]
fn a_factor_beyond_five_percent_disagrees() {
    let measurements = stored(&[(1.0, LENGTH_UNIT_M); 4]);
    let readings = [
        apart(0.01),
        apart(1.0 / 101.0),
        apart(1.0 / 99.0),
        apart(1.0 / 110.0),
    ];
    let scale = distance_measurement_scale(&measurements, &readings, LengthUnit::Metre).unwrap();
    assert_close(scale.factor(), 100.5, 1e-9, "median of an even count");
    assert_close(
        scale.factors[3].deviation_percent,
        100.0 * (110.0 / 100.5 - 1.0),
        1e-9,
        "the outlier's deviation",
    );
    assert!(scale.statistics.disagrees);

    // Either side of the bar: 5.1% disagrees, 4.9% does not.
    let measurements = stored(&[(1.0, LENGTH_UNIT_M); 3]);
    for (top, disagrees) in [(105.1, true), (104.9, false)] {
        let readings = [apart(0.01), apart(0.01), apart(1.0 / top)];
        let scale =
            distance_measurement_scale(&measurements, &readings, LengthUnit::Metre).unwrap();
        assert_close(
            scale.statistics.max_abs_deviation_percent,
            top - 100.0,
            1e-9,
            "largest deviation",
        );
        assert_eq!(scale.statistics.disagrees, disagrees, "at {top}");
    }
}

/// One measurement is its own factor, with no sample deviation.
#[test]
fn a_single_measurement_is_the_factor() {
    let scale = distance_measurement_scale(
        &stored(&[(250.0, LENGTH_UNIT_MM)]),
        &[apart(0.5)],
        LengthUnit::Millimetre,
    )
    .unwrap();
    let s = &scale.statistics;
    assert_eq!(s.count, 1);
    assert_close(scale.factor(), 500.0, 1e-12, "factor");
    assert_eq!((s.mean, s.min, s.max), (500.0, 500.0, 500.0));
    assert_eq!((s.sample_std, s.sample_std_percent), (None, None));
    assert_eq!(s.median_absolute_deviation, 0.0);
    assert_eq!(scale.factors[0].deviation_percent, 0.0);
    assert!(!s.disagrees);
}

/// Each exclusion is reported with its measurement's index and reason, and
/// the rest give the factor.
#[test]
fn each_exclusion_names_its_reason() {
    let refused = ResolvedDistanceMeasurement {
        endpoints: [finite(0.0), Err(EndpointRefusal::NoPatchFrames)],
        scene_distance: None,
    };
    let to_the_sky = ResolvedDistanceMeasurement {
        endpoints: [sky(), finite(1.0)],
        scene_distance: None,
    };
    let refused_and_sky = ResolvedDistanceMeasurement {
        endpoints: [sky(), Err(EndpointRefusal::NoPose { image: 3 })],
        scene_distance: None,
    };
    let coincident = ResolvedDistanceMeasurement {
        endpoints: [finite(2.0), finite(2.0)],
        scene_distance: Some(0.0),
    };
    let overflowed = ResolvedDistanceMeasurement {
        endpoints: [finite(0.0), finite(1.0)],
        scene_distance: Some(f64::INFINITY),
    };
    let measurements = stored(&[(1.0, LENGTH_UNIT_M); 7]);
    let mut bad_stated = stored(&[(1.0, LENGTH_UNIT_M); 7]);
    let d = bad_stated.distance.as_mut().unwrap();
    d.distances[5] = -1.0;
    d.units[6] = 200;
    let readings = [
        refused,
        apart(0.5),
        to_the_sky,
        refused_and_sky,
        coincident,
        apart(0.5),
        overflowed,
    ];
    let scale = distance_measurement_scale(&measurements, &readings, LengthUnit::Metre).unwrap();
    assert_eq!(
        scale.factors.iter().map(|f| f.index).collect::<Vec<_>>(),
        [1, 5]
    );
    assert_close(scale.factor(), 2.0, 1e-12, "factor");
    let reasons: Vec<_> = scale
        .excluded
        .iter()
        .map(|e| (e.index, e.reason.clone()))
        .collect();
    assert_eq!(
        reasons[..3],
        [
            (
                0,
                MeasurementScaleExclusion::EndpointRefused {
                    endpoint: 1,
                    refusal: EndpointRefusal::NoPatchFrames
                }
            ),
            (
                2,
                MeasurementScaleExclusion::EndpointAtInfinity { endpoint: 0 }
            ),
            (
                3,
                MeasurementScaleExclusion::EndpointRefused {
                    endpoint: 1,
                    refusal: EndpointRefusal::NoPose { image: 3 }
                }
            ),
        ]
    );
    assert_eq!(
        reasons[3],
        (
            4,
            MeasurementScaleExclusion::UnusableSceneDistance {
                scene_distance: 0.0
            }
        )
    );
    assert_eq!(
        reasons[4],
        (
            6,
            MeasurementScaleExclusion::UnusableSceneDistance {
                scene_distance: f64::INFINITY
            }
        )
    );
    assert_eq!(reasons.len(), 5);
    assert!(scale.excluded[0]
        .reason
        .to_string()
        .starts_with("endpoint B has no position: the reconstruction stores no patch frames"));
    assert!(scale.excluded[1]
        .reason
        .to_string()
        .contains("endpoint A lies on a point at infinity"));

    // A stated distance that is not positive, or in no known unit, is excluded
    // too; only a value built by hand holds one.
    let readings = [
        apart(1.0),
        apart(1.0),
        apart(1.0),
        apart(1.0),
        apart(1.0),
        apart(1.0),
        apart(1.0),
    ];
    let scale = distance_measurement_scale(&bad_stated, &readings, LengthUnit::Metre).unwrap();
    assert_eq!(scale.statistics.count, 5);
    assert_eq!(
        scale.excluded,
        [5, 6].map(|index| ExcludedMeasurement {
            index,
            reason: MeasurementScaleExclusion::UnusableStatedDistance
        })
    );
}

/// No usable measurement is an error carrying every exclusion; so is none
/// stored, and readings that are not one per measurement.
#[test]
fn no_usable_measurement_is_an_error() {
    let readings = [
        ResolvedDistanceMeasurement {
            endpoints: [sky(), finite(1.0)],
            scene_distance: None,
        },
        ResolvedDistanceMeasurement {
            endpoints: [finite(1.0), finite(1.0)],
            scene_distance: Some(0.0),
        },
    ];
    let err = distance_measurement_scale(
        &stored(&[(1.0, LENGTH_UNIT_M); 2]),
        &readings,
        LengthUnit::Metre,
    )
    .unwrap_err();
    let DistanceMeasurementScaleError::NoUsableMeasurement { excluded } = &err else {
        panic!("not NoUsableMeasurement: {err:?}");
    };
    assert_eq!(excluded.iter().map(|e| e.index).collect::<Vec<_>>(), [0, 1]);
    let text = err.to_string();
    assert!(text.starts_with("none of the 2 distance measurements gives a scale"));
    assert!(text.contains("measurement 1: its endpoints are 0 scene units apart"));

    assert_eq!(
        distance_measurement_scale(&MeasurementsData::default(), &[], LengthUnit::Metre),
        Err(DistanceMeasurementScaleError::NoMeasurements)
    );
    assert_eq!(
        distance_measurement_scale(
            &stored(&[(1.0, LENGTH_UNIT_M); 2]),
            &[apart(1.0)],
            LengthUnit::Metre
        ),
        Err(DistanceMeasurementScaleError::ReadingCountMismatch {
            measurements: 2,
            readings: 1
        })
    );
}

/// End to end on the endpoint fixture: two endpoints 3 scene units apart
/// stated as 30 cm give 100 to millimetres whether the file declares no unit
/// or metres, and scaling the reconstruction by it puts the endpoints 300
/// apart, the stated distance in millimetres.
#[test]
fn the_factor_puts_the_reconstruction_in_the_unit_with_or_without_a_unit() {
    let mut recon = scene(&[pt(1.5, 0.0, 0.0, 0.2), pt(-1.5, 0.0, 0.0, 0.2)]);
    let a = pixel_of_texel(&recon, 0, 0, 0.0, 0.0);
    let b = pixel_of_texel(&recon, 1, 1, 0.0, 0.0);
    recon.image_set.measurements.distance = Some(DistanceMeasurementsData {
        distances: Array1::from(vec![30.0_f32]),
        units: Array1::from(vec![LENGTH_UNIT_CM]),
        image_indexes: Array2::from_shape_vec((1, 2), vec![0_u32, 1]).unwrap(),
        endpoints_xy: Array3::from_shape_vec(
            (1, 2, 2),
            vec![a[0] as f32, a[1] as f32, b[0] as f32, b[1] as f32],
        )
        .unwrap(),
        labels: vec!["across".into()],
    });
    let opts = MeasurementEndpointOptions::default();
    for unit in [None, Some("m".to_string())] {
        let mut recon = recon.clone();
        recon.metadata.world_space_unit = unit.clone();
        let edited = EditedReconstruction::new(Arc::new(recon.clone()));
        let readings = resolve_distance_measurements(&edited, &opts);
        let scale =
            distance_measurement_scale(edited.measurements(), &readings, LengthUnit::Millimetre)
                .unwrap();
        // The endpoints are 3 apart to within the f32 pixels.
        assert_close(
            scale.factor(),
            100.0,
            0.05,
            &format!("factor with {unit:?}"),
        );

        let mut scaled = recon.apply_se3_transform(&Se3Transform::new(
            RotQuaternion::identity(),
            Vector3::zeros(),
            scale.factor(),
        ));
        scaled.metadata.world_space_unit = Some(LengthUnit::Millimetre.name().to_string());
        let after =
            resolve_distance_measurements(&EditedReconstruction::new(Arc::new(scaled)), &opts);
        assert_close(
            after[0].scene_distance.unwrap(),
            300.0,
            1e-6,
            &format!("scaled distance with {unit:?}"),
        );
    }
}
