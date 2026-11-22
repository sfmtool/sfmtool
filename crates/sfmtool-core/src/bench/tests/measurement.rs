// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The editable measurement: putting a stored one on the bench, its label,
//! what deleting an image does to it, its commit and the direct delete.

use std::sync::Arc;

use ndarray::{Array1, Array2, Array3};

use super::super::*;
use crate::reconstruction::edited::EditedReconstruction;
use crate::{DistanceMeasurementsData, LengthUnit, MeasurementsData, SfmrReconstruction};

/// The demo reconstruction (eight images, one 1920 x 1080 camera) with three
/// distance measurements: rows 0 and 1 in images 0 and 3, row 2 in images 5
/// and 5.
fn measured() -> EditedReconstruction {
    let mut recon = SfmrReconstruction::demo(10);
    recon.image_set.measurements = MeasurementsData {
        distance: Some(DistanceMeasurementsData {
            distances: Array1::from(vec![12.0, 31.5, 4.25]),
            units: Array1::from(vec![
                LengthUnit::Centimetre.code(),
                LengthUnit::Centimetre.code(),
                LengthUnit::Inch.code(),
            ]),
            image_indexes: Array2::from_shape_vec((3, 2), vec![0, 3, 0, 3, 5, 5]).unwrap(),
            endpoints_xy: Array3::from_shape_vec(
                (3, 2, 2),
                vec![
                    100.5, 200.25, 300.0, 400.0, //
                    10.0, 20.0, 30.0, 40.0, //
                    500.0, 600.0, 700.0, 800.0,
                ],
            )
            .unwrap(),
            labels: vec!["ear to ear".into(), String::new(), "paw".into()],
        }),
    };
    EditedReconstruction::new(Arc::new(recon))
}

fn distance(item: &EditableMeasurement) -> &EditableDistance {
    item.as_distance().expect("a distance")
}

fn row(value: &EditedReconstruction, i: usize) -> (f32, u8, [u32; 2], [f32; 4], String) {
    let m = value.measurements().distance.as_ref().unwrap();
    (
        m.distances[i],
        m.units[i],
        [m.image_indexes[[i, 0]], m.image_indexes[[i, 1]]],
        [
            m.endpoints_xy[[i, 0, 0]],
            m.endpoints_xy[[i, 0, 1]],
            m.endpoints_xy[[i, 1, 0]],
            m.endpoints_xy[[i, 1, 1]],
        ],
        m.labels[i].clone(),
    )
}

#[test]
fn a_stored_measurement_goes_on_as_its_row_says() {
    let value = measured();
    let (bench, label) =
        put_measurement(&Bench::new(), &value, MeasurementKind::Distance, 0, 7).unwrap();
    assert_eq!(label, "distance #0");
    let item = bench.measurement(&label).expect("put on");
    assert_eq!(bench.get(&label).unwrap().kind(), ItemKind::Measurement);
    assert!(bench.track(&label).is_none());
    assert_eq!(
        item.origin,
        Some(MeasurementOrigin {
            version: 7,
            kind: MeasurementKind::Distance,
            index: 0,
        })
    );
    assert_eq!(item.label, "ear to ear");
    let d = distance(item);
    assert_eq!(
        d.endpoints,
        [
            Some(MeasurementEndpoint {
                image: 0,
                pixel: [100.5, 200.25],
            }),
            Some(MeasurementEndpoint {
                image: 3,
                pixel: [300.0, 400.0],
            }),
        ]
    );
    assert_eq!(d.distance, Some((12.0, LengthUnit::Centimetre)));
}

#[test]
fn a_second_put_of_the_same_row_takes_the_collision_suffix() {
    let value = measured();
    let (bench, _) =
        put_measurement(&Bench::new(), &value, MeasurementKind::Distance, 2, 0).unwrap();
    let (bench, label) = put_measurement(&bench, &value, MeasurementKind::Distance, 2, 0).unwrap();
    assert_eq!(label, "distance #2 (2)");
    assert_eq!(bench.len(), 2);
    assert_ne!(bench.id("distance #2"), bench.id("distance #2 (2)"));
}

#[test]
fn a_row_past_the_rows_is_refused_with_the_count() {
    let value = measured();
    let refused =
        put_measurement(&Bench::new(), &value, MeasurementKind::Distance, 3, 0).unwrap_err();
    assert_eq!(
        refused,
        NoSuchMeasurement {
            kind: MeasurementKind::Distance,
            index: 3,
            count: 3,
        }
    );
    assert_eq!(
        refused.to_string(),
        "there is no distance #3: the reconstruction has 3 distance measurements, distance #0 to #2"
    );
    let none = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
    assert_eq!(
        put_measurement(&Bench::new(), &none, MeasurementKind::Distance, 0, 0)
            .unwrap_err()
            .to_string(),
        "there is no distance #0: the reconstruction has no distance measurements"
    );
}

#[test]
fn the_kind_names_itself_one_way_everywhere() {
    assert_eq!(MeasurementKind::Distance.title(), "Distance");
    assert_eq!(MeasurementKind::Distance.name(), "distance");
    assert_eq!(
        MeasurementKind::from_name("distance"),
        Some(MeasurementKind::Distance)
    );
    assert_eq!(MeasurementKind::from_name("Distance"), None);
    assert_eq!(ItemKind::Measurement.to_string(), "measurement");
}

// ---- Deleting an image ----------------------------------------------------

fn put_rows(value: &EditedReconstruction, rows: &[u32]) -> Bench {
    rows.iter().fold(Bench::new(), |bench, &row| {
        put_measurement(&bench, value, MeasurementKind::Distance, row, 0)
            .unwrap()
            .0
    })
}

fn images_of(bench: &Bench, label: &str) -> [Option<u32>; 2] {
    distance(bench.measurement(label).unwrap())
        .endpoints
        .map(|e| e.map(|e| e.image))
}

#[test]
fn deleting_an_image_clears_its_endpoints_and_moves_later_ones_down() {
    let value = measured();
    let bench = put_rows(&value, &[0, 2]);
    let id = bench.id("distance #0").unwrap();
    let (next, report) = bench.delete_image(3);
    // Row 0's B was in image 3: cleared. Row 2's two were in image 5: moved
    // down to 4.
    assert_eq!(images_of(&next, "distance #0"), [Some(0), None]);
    assert_eq!(images_of(&next, "distance #2"), [Some(4), Some(4)]);
    assert_eq!(report.endpoints_cleared, 1);
    assert_eq!(report.moved_measurements.len(), 2);
    assert!(report.discarded.is_empty());
    assert!(report.changed());
    // The item keeps its ID, and the origin is left for the caller to follow.
    assert_eq!(next.id("distance #0"), Some(id));
    assert_eq!(
        next.measurement("distance #0").unwrap().origin,
        bench.measurement("distance #0").unwrap().origin
    );
}

#[test]
fn a_measurement_left_with_no_endpoint_is_discarded() {
    let value = measured();
    let bench = put_rows(&value, &[2, 0]);
    let (next, report) = bench.delete_image(5);
    assert_eq!(report.discarded, vec!["distance #2".to_string()]);
    assert_eq!(report.endpoints_cleared, 2);
    assert!(next.get("distance #2").is_none());
    assert_eq!(next.len(), 1);
}

#[test]
fn a_measurement_the_delete_does_not_reach_keeps_its_arc() {
    let value = measured();
    let bench = put_rows(&value, &[0]);
    let (next, report) = bench.delete_image(6);
    assert!(!report.changed());
    assert!(next
        .get("distance #0")
        .unwrap()
        .same_value(bench.get("distance #0").unwrap()));
}

// ---- The commit -----------------------------------------------------------

fn item_of(value: &EditedReconstruction, row: u32) -> EditableMeasurement {
    EditableMeasurement::from_stored(value, MeasurementKind::Distance, row, 0).unwrap()
}

fn with_distance(
    item: &EditableMeasurement,
    edit: impl FnOnce(&mut EditableDistance),
) -> EditableMeasurement {
    let mut next = item.clone();
    let EditableMeasurementBody::Distance(d) = &mut next.body;
    edit(d);
    next
}

#[test]
fn a_commit_replaces_the_row_and_shares_the_base() {
    let value = measured();
    let item = with_distance(&item_of(&value, 1), |d| {
        d.endpoints[0] = Some(MeasurementEndpoint {
            image: 2,
            pixel: [11.0, 22.5],
        });
        d.distance = Some((30.0, LengthUnit::Centimetre));
    });
    let commit = commit_measurement(&value, &item, Some(1)).unwrap();
    assert!(commit.wrote);
    assert!(commit.replaced);
    assert_eq!(commit.index, 1);
    assert_eq!(
        row(&commit.value, 1),
        (
            30.0,
            LengthUnit::Centimetre.code(),
            [2, 3],
            [11.0, 22.5, 30.0, 40.0],
            String::new()
        )
    );
    // The other rows as they were.
    assert_eq!(row(&commit.value, 0), row(&value, 0));
    assert_eq!(row(&commit.value, 2), row(&value, 2));
    // The base is the same allocation, so the points, the tracks and the
    // patch bitmaps are shared rather than copied, and no point moved.
    assert!(Arc::ptr_eq(&commit.value.base, &value.base));
    assert!(std::ptr::eq(
        commit.value.base.point_set.points.as_ptr(),
        value.base.point_set.points.as_ptr()
    ));
    assert_eq!(commit.value.point_count(), value.point_count());
    // The value committed into is unchanged.
    assert_eq!(row(&value, 1).0, 31.5);
    assert_eq!(commit.label("dino"), "Committed distance #1 in dino");
}

#[test]
fn a_commit_with_no_row_appends_one() {
    let value = measured();
    let mut item = item_of(&value, 0);
    item.origin = None;
    item.label = "copy".into();
    let commit = commit_measurement(&value, &item, None).unwrap();
    assert!(commit.wrote);
    assert!(!commit.replaced);
    assert_eq!(commit.index, 3);
    assert_eq!(commit.value.measurements().distance_measurement_count(), 4);
    let (d, u, images, pixels, label) = row(&commit.value, 3);
    assert_eq!((d, u, images, pixels), {
        let (d, u, images, pixels, _) = row(&value, 0);
        (d, u, images, pixels)
    });
    assert_eq!(label, "copy");
    assert_eq!(commit.label("dino"), "Added distance #3 in dino");
}

#[test]
fn a_commit_onto_a_reconstruction_with_none_starts_the_kind() {
    let none = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
    let mut item = item_of(&measured(), 2);
    item.origin = None;
    let commit = commit_measurement(&none, &item, None).unwrap();
    assert_eq!(commit.index, 0);
    assert_eq!(commit.value.measurements().distance_measurement_count(), 1);
}

#[test]
fn a_commit_of_what_the_row_holds_writes_nothing() {
    let value = measured();
    let commit = commit_measurement(&value, &item_of(&value, 0), Some(0)).unwrap();
    assert!(!commit.wrote);
    assert_eq!(commit.index, 0);
    assert_eq!(commit.value, value);
    assert_eq!(
        commit.label("dino"),
        "Committed distance #0 in dino: no effect, the row already holds it"
    );
}

#[test]
fn each_refusal_says_what_is_wrong() {
    let value = measured();
    let item = item_of(&value, 0);
    let refused = |item: &EditableMeasurement, row: Option<u32>| {
        let refusal = commit_measurement(&value, item, row).unwrap_err();
        // The preconditions and the step give the same answer. Compared as
        // sentences, since a pixel that is not a number is not equal to
        // itself.
        assert_eq!(
            commit_measurement_preconditions(&value, item, row)
                .unwrap_err()
                .to_string(),
            refusal.to_string()
        );
        refusal
    };
    let unplaced = with_distance(&item, |d| d.endpoints[1] = None);
    assert_eq!(
        refused(&unplaced, Some(0)),
        MeasurementCommitRefusal::EndpointNotPlaced { side: 1 }
    );
    assert_eq!(
        refused(&unplaced, Some(0)).to_string(),
        "endpoint B is not placed, and a distance needs both endpoints"
    );
    let unstated = with_distance(&item, |d| d.distance = None);
    assert_eq!(
        refused(&unstated, Some(0)),
        MeasurementCommitRefusal::NoStatedDistance
    );
    let negative = with_distance(&item, |d| d.distance = Some((-1.0, LengthUnit::Metre)));
    assert_eq!(
        refused(&negative, Some(0)),
        MeasurementCommitRefusal::DistanceNotPositive { distance: -1.0 }
    );
    let past = with_distance(&item, |d| {
        d.endpoints[0] = Some(MeasurementEndpoint {
            image: 8,
            pixel: [1.0, 1.0],
        })
    });
    assert_eq!(
        refused(&past, Some(0)),
        MeasurementCommitRefusal::NoSuchImage {
            side: 0,
            image: 8,
            image_count: 8,
        }
    );
    let outside = with_distance(&item, |d| {
        d.endpoints[1] = Some(MeasurementEndpoint {
            image: 1,
            pixel: [1920.0, 5.0],
        })
    });
    assert_eq!(
        refused(&outside, Some(0)),
        MeasurementCommitRefusal::PixelOutsideImage {
            side: 1,
            image: 1,
            pixel: [1920.0, 5.0],
            width: 1920,
            height: 1080,
        }
    );
    let not_a_number = with_distance(&item, |d| {
        d.endpoints[1] = Some(MeasurementEndpoint {
            image: 1,
            pixel: [f64::NAN, 5.0],
        })
    });
    assert!(matches!(
        refused(&not_a_number, Some(0)),
        MeasurementCommitRefusal::PixelOutsideImage { side: 1, .. }
    ));
    assert_eq!(
        refused(&item, Some(3)),
        MeasurementCommitRefusal::NoSuchRow(NoSuchMeasurement {
            kind: MeasurementKind::Distance,
            index: 3,
            count: 3,
        })
    );
    assert_eq!(
        commit_measurement_preconditions(&value, &item, Some(0)),
        Ok(())
    );
}

#[test]
fn a_committed_item_reseated_on_its_row_commits_to_no_effect() {
    let value = measured();
    let item = with_distance(&item_of(&value, 2), |d| {
        d.distance = Some((5.0, LengthUnit::Inch))
    });
    let commit = commit_measurement(&value, &item, Some(2)).unwrap();
    let seated = item.with_origin(9, commit.index);
    assert_eq!(seated.origin.unwrap().version, 9);
    let again = commit_measurement(&commit.value, &seated, Some(2)).unwrap();
    assert!(!again.wrote);
}

// ---- The direct delete ----------------------------------------------------

#[test]
fn deleting_a_measurement_renumbers_the_rows_after_it() {
    let value = measured();
    let (next, map) = delete_measurement(&value, MeasurementKind::Distance, 1).unwrap();
    assert_eq!(map, vec![Some(0), None, Some(1)]);
    assert_eq!(next.measurements().distance_measurement_count(), 2);
    assert_eq!(row(&next, 0), row(&value, 0));
    assert_eq!(row(&next, 1), row(&value, 2));
    assert!(Arc::ptr_eq(&next.base, &value.base));
}

#[test]
fn deleting_the_last_measurement_leaves_the_kind_absent() {
    let value = measured();
    let (one, _) = delete_measurement(&value, MeasurementKind::Distance, 0).unwrap();
    let (one, _) = delete_measurement(&one, MeasurementKind::Distance, 0).unwrap();
    let (none, map) = delete_measurement(&one, MeasurementKind::Distance, 0).unwrap();
    assert_eq!(map, vec![None]);
    assert!(none.measurements().distance.is_none());
    assert_eq!(
        delete_measurement(&none, MeasurementKind::Distance, 0).unwrap_err(),
        NoSuchMeasurement {
            kind: MeasurementKind::Distance,
            index: 0,
            count: 0,
        }
    );
}

// ---- The value's own measurements -----------------------------------------

#[test]
fn a_value_with_its_own_measurements_materializes_them() {
    let value = measured();
    assert!(!value.has_edits());
    let (next, _) = delete_measurement(&value, MeasurementKind::Distance, 0).unwrap();
    assert!(next.has_edits());
    // The base still holds three; the value holds two, and so does what it
    // materializes to, with the metadata count to match.
    assert_eq!(
        next.base
            .image_set
            .measurements
            .distance_measurement_count(),
        3
    );
    let (plain, _) = next.materialize();
    assert_eq!(plain.image_set.measurements.distance_measurement_count(), 2);
    assert_eq!(plain.metadata.distance_measurement_count, Some(2));
    assert_ne!(next, value);
    // A point edit on top keeps the measurements.
    let mut edited = next.clone();
    edited.delete_point(0).unwrap();
    assert_eq!(edited.measurements(), next.measurements());
}

/// A value built by hand whose columns disagree in length is read without a
/// panic: the row the short columns lack reads as an image past the set and
/// pixels that are not numbers, which the commit refuses, and a delete still
/// renumbers the rows.
#[test]
fn columns_of_different_lengths_are_read_without_a_panic() {
    let mut recon = SfmrReconstruction::demo(10);
    recon.image_set.measurements = MeasurementsData {
        distance: Some(DistanceMeasurementsData {
            distances: Array1::from(vec![1.0, 2.0]),
            units: Array1::from(vec![LengthUnit::Metre.code()]),
            image_indexes: Array2::from_shape_vec((1, 2), vec![0, 1]).unwrap(),
            endpoints_xy: Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            labels: vec!["only".into()],
        }),
    };
    let value = EditedReconstruction::new(Arc::new(recon));
    let short = item_of(&value, 1);
    let d = distance(&short);
    assert_eq!(d.endpoints[0].unwrap().image, u32::MAX);
    assert!(d.endpoints[1].unwrap().pixel[0].is_nan());
    assert_eq!(d.distance, Some((2.0, LengthUnit::Metre)));
    assert!(matches!(
        commit_measurement(&value, &short, Some(1)).unwrap_err(),
        MeasurementCommitRefusal::NoSuchImage { side: 0, .. }
    ));
    let (next, map) = delete_measurement(&value, MeasurementKind::Distance, 1).unwrap();
    assert_eq!(map, vec![Some(0), None]);
    assert_eq!(row(&next, 0).4, "only");
}

// ---- Setting the label and the stated distance ----------------------------

#[test]
fn setting_the_label_changes_the_label_and_nothing_else() {
    let value = measured();
    let item = item_of(&value, 0);
    let next = set_measurement_label(&item, "paw width");
    assert_eq!(next.label, "paw width");
    assert_eq!(item.label, "ear to ear", "the step changed its input");
    assert_eq!(next.origin, item.origin);
    assert_eq!(next.body, item.body);
    // The empty label is a label: the file allows an unlabelled measurement.
    assert_eq!(set_measurement_label(&item, "").label, "");
    // A commit writes the new label into the row and nothing else.
    let commit = commit_measurement(&value, &next, Some(0)).unwrap();
    assert!(commit.wrote);
    let (d, unit, images, pixels, label) = row(&commit.value, 0);
    assert_eq!(label, "paw width");
    assert_eq!((d, unit, images, pixels), {
        let (d, unit, images, pixels, _) = row(&value, 0);
        (d, unit, images, pixels)
    });
}

#[test]
fn setting_the_stated_distance_changes_the_distance_and_nothing_else() {
    let value = measured();
    let item = item_of(&value, 0);
    let next = set_stated_distance(&item, 4.2, LengthUnit::Inch).unwrap();
    assert_eq!(distance(&next).distance, Some((4.2, LengthUnit::Inch)));
    assert_eq!(distance(&next).endpoints, distance(&item).endpoints);
    assert_eq!(next.label, item.label);
    assert_eq!(next.origin, item.origin);
    assert_eq!(
        distance(&item).distance,
        Some((12.0, LengthUnit::Centimetre)),
        "the step changed its input"
    );
    // An item with no stated distance takes one.
    let unstated = with_distance(&item, |d| d.distance = None);
    assert_eq!(
        distance(&set_stated_distance(&unstated, 3.0, LengthUnit::Metre).unwrap()).distance,
        Some((3.0, LengthUnit::Metre))
    );
}

#[test]
fn a_stated_distance_that_is_not_finite_and_positive_is_refused() {
    let item = item_of(&measured(), 0);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300, 1e-60] {
        let refusal = set_stated_distance(&item, bad, LengthUnit::Metre).unwrap_err();
        assert!(
            matches!(refusal, StatedDistanceRefusal::NotPositive { .. }),
            "{bad}: {refusal:?}"
        );
    }
    // The sentence is the commit's, so the step and the commit read alike.
    assert_eq!(
        set_stated_distance(&item, -1.0, LengthUnit::Metre)
            .unwrap_err()
            .to_string(),
        MeasurementCommitRefusal::DistanceNotPositive { distance: -1.0 }.to_string()
    );
    assert_eq!(
        StatedDistanceRefusal::NotPositive { distance: -1.0 }.to_string(),
        "the stated distance is -1, which must be finite and strictly positive"
    );
}

#[test]
fn a_distance_reads_as_a_number_with_an_optional_unit() {
    let parse = |text: &str, bare: LengthUnit| parse_stated_distance(text, bare);
    // A suffix names the unit, with or without a space before it.
    assert_eq!(
        parse("12in", LengthUnit::Metre),
        Ok((12.0, LengthUnit::Inch))
    );
    assert_eq!(
        parse("30.48cm", LengthUnit::Metre),
        Ok((30.48, LengthUnit::Centimetre))
    );
    assert_eq!(
        parse(" 4.2 cm ", LengthUnit::Metre),
        Ok((4.2, LengthUnit::Centimetre))
    );
    assert_eq!(
        parse("0.3048m", LengthUnit::Foot),
        Ok((0.3048, LengthUnit::Metre))
    );
    assert_eq!(parse("2ft", LengthUnit::Metre), Ok((2.0, LengthUnit::Foot)));
    assert_eq!(
        parse("5mm", LengthUnit::Metre),
        Ok((5.0, LengthUnit::Millimetre))
    );
    // A bare number takes the unit the caller gives.
    assert_eq!(
        parse("304.8", LengthUnit::Millimetre),
        Ok((304.8, LengthUnit::Millimetre))
    );
    assert_eq!(
        parse("12", LengthUnit::Metre),
        Ok((12.0, LengthUnit::Metre))
    );
    // The number's other spellings.
    assert_eq!(parse(".5", LengthUnit::Metre), Ok((0.5, LengthUnit::Metre)));
    assert_eq!(
        parse("12.", LengthUnit::Metre),
        Ok((12.0, LengthUnit::Metre))
    );
    assert_eq!(parse("+3", LengthUnit::Metre), Ok((3.0, LengthUnit::Metre)));
    assert_eq!(
        parse("1e-3m", LengthUnit::Inch),
        Ok((0.001, LengthUnit::Metre))
    );
    assert_eq!(
        parse("2.5E2", LengthUnit::Metre),
        Ok((250.0, LengthUnit::Metre))
    );
}

#[test]
fn a_distance_that_does_not_read_is_refused_with_why() {
    let parse = |text: &str| parse_stated_distance(text, LengthUnit::Metre);
    for unreadable in [
        "",
        "cm",
        "abc",
        "inf",
        "nan",
        "12 IN",
        "1.2.3",
        "1e",
        "e5",
        "-",
        ".",
        // Text holding a multi-byte character is refused, not cut inside one.
        "12\u{e9}",
        "\u{e9}",
        "12m\u{e9}",
        "\u{661}\u{662}",
    ] {
        // `1e` reads as the number 1 with the unit `e`, which is unknown.
        let refusal = parse(unreadable).unwrap_err();
        assert!(
            matches!(
                refusal,
                StatedDistanceRefusal::Unreadable { .. }
                    | StatedDistanceRefusal::UnknownUnit { .. }
            ),
            "{unreadable:?}: {refusal:?}"
        );
    }
    assert_eq!(
        parse("12 IN"),
        Err(StatedDistanceRefusal::Unreadable {
            text: "12 IN".to_string()
        })
    );
    assert_eq!(
        parse("12 IN").unwrap_err().to_string(),
        "\"12 IN\" is not a distance: give a number with an optional unit suffix, such as \
         304.8, 12in or 0.3048 m"
    );
    // A space other than ASCII's between the number and the unit is a space.
    assert_eq!(parse("12\u{a0}cm"), Ok((12.0, LengthUnit::Centimetre)));
    assert_eq!(parse("12\u{2009}cm"), Ok((12.0, LengthUnit::Centimetre)));
    assert_eq!(
        parse("12yd"),
        Err(StatedDistanceRefusal::UnknownUnit {
            unit: "yd".to_string()
        })
    );
    assert_eq!(
        parse("12yd").unwrap_err().to_string(),
        "\"yd\" is not a unit: the units are mm, cm, m, in, ft"
    );
    for not_positive in ["0", "-2cm", "0.0m", "1e-60"] {
        assert!(
            matches!(
                parse(not_positive),
                Err(StatedDistanceRefusal::NotPositive { .. })
            ),
            "{not_positive:?}"
        );
    }
    // Too long for a `float32`, so the file could not hold it.
    let huge = format!("1{}", "0".repeat(40));
    assert!(matches!(
        parse(&huge),
        Err(StatedDistanceRefusal::NotPositive { .. })
    ));
}

// ---- Setting an endpoint --------------------------------------------------

#[test]
fn setting_an_endpoint_changes_that_endpoint_and_nothing_else() {
    let value = measured();
    let item = item_of(&value, 0);
    let moved = MeasurementEndpoint {
        image: 6,
        pixel: [1919.5, 0.0],
    };
    let next = set_measurement_endpoint(&value, &item, 0, moved).unwrap();
    assert_eq!(distance(&next).endpoints[0], Some(moved));
    assert_eq!(distance(&next).endpoints[1], distance(&item).endpoints[1]);
    assert_eq!(distance(&next).distance, distance(&item).distance);
    assert_eq!((&next.label, next.origin), (&item.label, item.origin));
    assert_eq!(
        distance(&item).endpoints[0].unwrap().image,
        0,
        "the step changed its input"
    );
    // An endpoint not placed is placed.
    let half = with_distance(&item, |d| d.endpoints[1] = None);
    let placed = set_measurement_endpoint(&value, &half, 1, moved).unwrap();
    assert_eq!(distance(&placed).endpoints[1], Some(moved));
    // A commit writes the new endpoint into the row.
    let commit = commit_measurement(&value, &next, Some(0)).unwrap();
    let (_, _, images, pixels, _) = row(&commit.value, 0);
    assert_eq!(images, [6, 3]);
    assert_eq!(&pixels[..2], &[1919.5, 0.0]);
}

#[test]
fn an_endpoint_outside_the_images_is_refused_in_the_commits_words() {
    let value = measured();
    let item = item_of(&value, 0);
    let at = |image: u32, pixel: [f64; 2]| MeasurementEndpoint { image, pixel };
    let refused = set_measurement_endpoint(&value, &item, 1, at(8, [1.0, 1.0])).unwrap_err();
    assert_eq!(
        refused,
        EndpointPlacementRefusal::Outside(MeasurementCommitRefusal::NoSuchImage {
            side: 1,
            image: 8,
            image_count: 8,
        })
    );
    // The commit of an item holding that endpoint says the same.
    let held = with_distance(&item, |d| d.endpoints[1] = Some(at(8, [1.0, 1.0])));
    let commit_says = commit_measurement(&value, &held, Some(0)).unwrap_err();
    assert_eq!(refused.to_string(), commit_says.to_string());

    for pixel in [[1920.0, 5.0], [5.0, -0.5], [f64::NAN, 5.0], [5.0, 1080.0]] {
        let refused = set_measurement_endpoint(&value, &item, 0, at(2, pixel)).unwrap_err();
        assert!(
            matches!(
                refused,
                EndpointPlacementRefusal::Outside(MeasurementCommitRefusal::PixelOutsideImage {
                    side: 0,
                    image: 2,
                    width: 1920,
                    height: 1080,
                    ..
                })
            ),
            "{pixel:?}: {refused:?}"
        );
        assert!(refused.to_string().starts_with("endpoint A's pixel ("));
    }
    let refused = set_measurement_endpoint(&value, &item, 2, at(0, [1.0, 1.0])).unwrap_err();
    assert_eq!(
        refused,
        EndpointPlacementRefusal::NoSuchEndpoint { side: 2 }
    );
    assert_eq!(
        refused.to_string(),
        "a distance has endpoints 0 (A) and 1 (B), so there is no endpoint 2"
    );
}
