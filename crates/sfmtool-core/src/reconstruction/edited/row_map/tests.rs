// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The row map's contract: a scan of an edit's input and output follows every
//! surviving point to its new row and reports the rows the edit created, and a
//! removal map closes up behind the indexes it dropped.

use std::sync::Arc;

use super::super::tests::{fixture, new_record};
use super::*;
use crate::reconstruction::edited::EditedReconstruction;

/// Every point of `before` that `map` kept lands on a distinct row of `after`
/// holding the same point, every surviving row is landed on once, and the two
/// directions agree. Returns which rows of `after` the map reports as created.
fn check_bijection(
    map: &RowMap,
    before: &SfmrReconstruction,
    after: &SfmrReconstruction,
) -> Vec<u32> {
    let mut landed = vec![false; after.point_count()];
    for old in 0..before.point_count() as u32 {
        let Some(new) = map.forward(old) else {
            continue;
        };
        assert!(
            !std::mem::replace(&mut landed[new as usize], true),
            "two points of the original landed on row {new}"
        );
        assert_eq!(
            map.inverse(new),
            Some(old),
            "the two directions disagree at {old} -> {new}"
        );
        assert_eq!(
            after.point_set.points[new as usize].position,
            before.point_set.points[old as usize].position,
            "row {new} is not the point row {old} held"
        );
    }
    (0..after.point_count() as u32)
        .filter(|&new| {
            let created = !landed[new as usize];
            assert_eq!(
                map.inverse(new).is_none(),
                created,
                "row {new} disagrees about whether it was created"
            );
            created
        })
        .collect()
}

#[test]
fn a_scan_over_an_image_subset_is_the_subsets_own_renumbering() {
    let before = fixture(200);
    let keep: Vec<u32> = (1..before.image_count() as u32).collect();
    let after = before.subset_by_image_indices(&keep, true).unwrap();
    let mut image_map = vec![None; before.image_count()];
    for (new, &old) in keep.iter().enumerate() {
        image_map[old as usize] = Some(new as u32);
    }

    let map = RowMap::by_scan(&before, &after, Some(&image_map)).unwrap();
    let created = check_bijection(&map, &before, &after);
    assert!(created.is_empty(), "an image subset creates no point");

    // The subset drops exactly the points image 0 was the only witness of, and
    // renumbers the rest in order: the same answer, computed the other way.
    let mut expected = 0u32;
    for old in 0..before.point_count() as u32 {
        let survives = before
            .point_set
            .tracks
            .iter()
            .any(|t| t.point_index == old && t.image_index != 0);
        if survives {
            assert_eq!(map.forward(old), Some(expected), "point {old}");
            expected += 1;
        } else {
            assert_eq!(map.forward(old), None, "point {old}");
        }
    }
    assert_eq!(expected as usize, after.point_count());
}

#[test]
fn a_scan_over_a_point_mask_is_the_mask() {
    let before = fixture(120);
    let mask: Vec<bool> = (0..before.point_count()).map(|i| i % 3 != 1).collect();
    let after = before.filter_points_by_mask(&mask);

    let map = RowMap::by_scan(&before, &after, None).unwrap();
    assert!(check_bijection(&map, &before, &after).is_empty());
    let mut expected = 0u32;
    for (old, &keep) in mask.iter().enumerate() {
        if keep {
            assert_eq!(map.forward(old as u32), Some(expected));
            expected += 1;
        } else {
            assert_eq!(map.forward(old as u32), None);
        }
    }
}

#[test]
fn a_scan_over_a_transform_is_the_identity() {
    let before = fixture(64);
    let transform = crate::Se3Transform {
        rotation: crate::RotQuaternion::from_nalgebra(nalgebra::UnitQuaternion::from_euler_angles(
            0.3, -0.2, 1.1,
        )),
        translation: nalgebra::Vector3::new(3.0, -1.0, 2.0),
        scale: 2.5,
    };
    let after = before.apply_se3_transform(&transform);

    let map = RowMap::by_scan(&before, &after, None).unwrap();
    for i in 0..before.point_count() as u32 {
        assert_eq!(map.forward(i), Some(i));
        assert_eq!(map.inverse(i), Some(i));
    }
}

#[test]
fn a_scan_reports_points_the_edit_created_and_maps_the_survivors_around_them() {
    // `after` is the full value and `before` is it with points removed, so
    // every removed point is one `after` holds and `before` does not: some
    // interleaved, and, since the mask ends on a dropped point, some appended.
    let after = fixture(90);
    let mask: Vec<bool> = (0..after.point_count())
        .map(|i| i % 4 != 2 && i + 1 != after.point_count())
        .collect();
    let before = after.filter_points_by_mask(&mask);

    let map = RowMap::by_scan(&before, &after, None).unwrap();
    let created = check_bijection(&map, &before, &after);

    let expected_created: Vec<u32> = mask
        .iter()
        .enumerate()
        .filter(|(_, &keep)| !keep)
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(created, expected_created);
    assert!(
        expected_created.len() > 1
            && *expected_created.last().unwrap() as usize == after.point_count() - 1,
        "the fixture must hold both an interleaved and an appended new point"
    );

    // Every survivor still names the row it came from, around the new ones.
    let mut old = 0u32;
    for (new, &keep) in mask.iter().enumerate() {
        if keep {
            assert_eq!(map.forward(old), Some(new as u32));
            old += 1;
        }
    }
}

#[test]
fn a_scan_of_a_materialisation_agrees_with_the_map_it_returned() {
    let base = Arc::new(fixture(80));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    edited.delete_point(3).unwrap();
    edited.delete_point(11).unwrap();
    let record = edited.point(20).unwrap().to_record();
    edited.replace_point(20, record).unwrap();
    edited
        .add_point(new_record(5, base.image_count() as u32))
        .unwrap();
    let (plain, materialised) = edited.materialize();

    let scanned = RowMap::by_scan(&base, &plain, None).unwrap();
    let created = check_bijection(&scanned, &base, &plain);

    // The two maps agree on every base point the overlay left alone: the two
    // deletions are gone from both, and everything else is where the
    // materialisation put it.
    for old in 0..base.point_count() as u32 {
        if old == 20 {
            continue;
        }
        assert_eq!(
            scanned.forward(old),
            materialised.forward(old),
            "the maps disagree on base point {old}"
        );
    }

    // Point 20 is where the two maps are answering different questions, and
    // both answers are right. The materialisation's map is over *edited*
    // indexes, and a replaced base index stopped resolving the moment the
    // overlay re-added the point under a new one, so it reports `None`. The
    // scan is over the base's own indexes and follows the point's identity, so
    // it reports the row the record landed in -- which is the base slot 20's
    // row, since a modified point goes back where it came from.
    assert_eq!(materialised.forward(20), None);
    let row = scanned.forward(20).expect("the point is still there");
    assert_eq!(materialised.inverse(row), Some(base.point_count() as u32));
    // The appended addition is the one row the base has no point for, and both
    // maps say so.
    assert_eq!(created, vec![plain.point_count() as u32 - 1]);
    assert_eq!(
        materialised.inverse(plain.point_count() as u32 - 1),
        Some(base.point_count() as u32 + 1),
        "the appended addition is the overlay's second added index"
    );
}

/// A removal map is exact in both directions: a dropped index resolves to
/// nothing, and a survivor moves down by the holes below it.
#[test]
fn a_removal_map_moves_the_survivors_down_by_the_holes_below_them() {
    let map = RowMap::by_removal(6, &[3, 0, 3]);
    let want = [None, Some(0), Some(1), None, Some(2), Some(3)];
    for (old, wanted) in want.iter().enumerate() {
        assert_eq!(map.forward(old as u32), *wanted, "point {old}");
        if let Some(new) = wanted {
            assert_eq!(map.inverse(*new), Some(old as u32));
        }
    }
    // Nothing was created, so every row of the answer has a source.
    assert_eq!(map.inverse_dense(4), [1, 2, 4, 5]);
    // An index past the point count named no point to begin with.
    assert_eq!(map.forward(6), None);
}

/// A removal map that removes nothing is the identity.
#[test]
fn a_removal_map_of_nothing_is_the_identity() {
    let map = RowMap::by_removal(3, &[]);
    for p in 0..3 {
        assert_eq!(map.forward(p), Some(p));
        assert_eq!(map.inverse(p), Some(p));
    }
}

#[test]
fn a_scan_refuses_an_image_map_of_the_wrong_length() {
    let before = fixture(8);
    let after = before.clone();
    let error = RowMap::by_scan(&before, &after, Some(&[Some(0)])).unwrap_err();
    assert_eq!(
        error,
        EditError::ScanImageMap {
            got: 1,
            expected: before.image_count()
        }
    );
}
