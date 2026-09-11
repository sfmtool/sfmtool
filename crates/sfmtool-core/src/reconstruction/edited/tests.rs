// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlay's contract: the base is never written, indexes are stable, a
//! read through the overlay is the read the materialisation gives under the row
//! map, and a materialised value's hash is the file's after a save.

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array2, Array3, Array4};

use sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_RANGED};

use super::*;
use crate::reconstruction::data::ObservationSource;

/// A demo reconstruction carrying *every* optional column, so a record that
/// omits or invents one is a failure the tests can see.
fn fixture(num_points: usize) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(num_points);
    let p = recon.point_set.points.len();
    let m = recon.point_set.tracks.len();

    let mut u = Array2::<f32>::zeros((p, 3));
    let mut v = Array2::<f32>::zeros((p, 3));
    for i in 0..p {
        u[[i, 0]] = 0.1 * (i + 1) as f32;
        v[[i, 1]] = 0.2 * (i + 1) as f32;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(Array4::<u8>::from_shape_fn(
        (p, 2, 2, 4),
        |(i, y, x, c)| ((i * 13 + y * 5 + x * 3 + c) % 256) as u8,
    )));
    recon.point_set.normal_confidence = Some((0..p).map(|i| (i * 7 % 256) as u8).collect());
    let mut constraints = PointConstraintColumns::all_free(p);
    if p > 1 {
        constraints.point_constraints[1] = POINT_CONSTRAINT_RANGED;
        constraints.constraint_distances[1] = 4.5;
        constraints.constraint_reference_images[1] = 2;
    }
    recon.point_set.point_constraints = Some(constraints);
    recon.point_set.observation_confidence = Some((0..m).map(|i| (i * 11 % 256) as u8).collect());
    if let ObservationSource::SiftFiles { keypoints_xy, .. } = &mut recon.point_set.observations {
        *keypoints_xy = Some(Array2::<f32>::from_shape_fn((m, 2), |(i, c)| {
            (i * 3 + c) as f32 * 0.5
        }));
    }
    recon.rebuild_derived_fields();
    recon
}

/// A record that is not any of the base's, for use as an addition.
fn new_record(seed: u32, image_count: u32) -> PointRecord {
    let a = seed % image_count;
    let b = (seed + 1) % image_count;
    let (first, second) = if a <= b { (a, b) } else { (b, a) };
    PointRecord {
        point: Point3D {
            position: nalgebra::Point3::new(seed as f64, 1.0, 2.0),
            w: 1.0,
            color: [seed as u8, 2, 3],
            error: 0.25,
            normal: nalgebra::Vector3::new(0.0, 0.0, 1.0),
        },
        observations: vec![
            RecordObservation {
                image_index: first,
                feature_index: Some(1000 + seed),
                keypoint_xy: Some([seed as f32, 7.0]),
                confidence: Some(200),
            },
            RecordObservation {
                image_index: second,
                feature_index: Some(2000 + seed),
                keypoint_xy: Some([3.0, seed as f32]),
                confidence: Some(201),
            },
        ],
        patch_u_halfvec: Some([1.0, 0.0, 0.0]),
        patch_v_halfvec: Some([0.0, 1.0, 0.0]),
        patch_bitmap: Some(Array3::<u8>::from_shape_fn((2, 2, 4), |(y, x, c)| {
            ((seed as usize + y + x + c) % 256) as u8
        })),
        normal_confidence: Some(42),
        constraint: Some((POINT_CONSTRAINT_FREE, f64::NAN, NO_REFERENCE_IMAGE)),
    }
}

/// A record's equality with `NaN` distances treated as equal, since the free
/// constraint's distance is `NaN` by definition and `NaN != NaN`.
fn records_agree(a: &PointRecord, b: &PointRecord) -> bool {
    let constraints_agree = match (a.constraint, b.constraint) {
        (Some((ka, da, ra)), Some((kb, db, rb))) => {
            ka == kb && ra == rb && (da == db || (da.is_nan() && db.is_nan()))
        }
        (x, y) => x.is_none() && y.is_none(),
    };
    constraints_agree
        && a.point == b.point
        && a.observations == b.observations
        && a.patch_u_halfvec == b.patch_u_halfvec
        && a.patch_v_halfvec == b.patch_v_halfvec
        && a.patch_bitmap == b.patch_bitmap
        && a.normal_confidence == b.normal_confidence
}

#[test]
fn a_fresh_overlay_is_its_base() {
    let base = Arc::new(fixture(6));
    let edited = EditedReconstruction::new(Arc::clone(&base));
    assert_eq!(edited.point_count(), base.point_count());
    assert_eq!(edited.image_count(), base.image_count());
    for i in 0..base.point_count() as u32 {
        let view = edited.point(i).expect("live");
        assert_eq!(view.point(), &base.point_set.points[i as usize]);
        assert_eq!(view.observations(), base.observations_for_point(i as usize));
    }
    let (mat, map) = edited.materialize();
    assert_eq!(mat.point_count(), base.point_count());
    for i in 0..base.point_count() as u32 {
        assert_eq!(map.forward(i), Some(i));
        assert_eq!(map.inverse(i), Some(i));
    }
}

#[test]
fn point_edits_leave_the_base_arc_alone() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    let record = edited.point(3).unwrap().to_record();
    edited.delete_point(0).unwrap();
    edited.replace_point(3, record).unwrap();
    edited.add_point(new_record(5, 8)).unwrap();
    assert!(Arc::ptr_eq(&edited.base, &base));
    // And the base still reads as it did: nothing was written through it.
    assert_eq!(base.point_count(), 6);
}

#[test]
fn indexes_are_stable_across_every_point_edit() {
    let base = Arc::new(fixture(6));
    let before: Vec<PointRecord> = (0..6)
        .map(|i| {
            EditedReconstruction::new(Arc::clone(&base))
                .point(i)
                .unwrap()
                .to_record()
        })
        .collect();

    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    edited.delete_point(1).unwrap();
    let moved = edited.replace_point(4, new_record(9, 8)).unwrap();
    edited.add_point(new_record(11, 8)).unwrap();

    for i in [0u32, 2, 3, 5] {
        let view = edited.point(i).expect("untouched indexes still resolve");
        assert!(records_agree(&view.to_record(), &before[i as usize]));
    }
    assert!(
        edited.point(1).is_none(),
        "a deleted index resolves to nothing"
    );
    assert!(
        edited.point(4).is_none(),
        "a replaced index resolves to nothing"
    );
    assert_eq!(moved, 6, "the replacement took the first addition index");
    assert_eq!(edited.point_count(), 6);
}

#[test]
fn a_replacement_goes_back_to_its_base_index() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    let replacement = new_record(3, 8);
    let moved = edited.replace_point(2, replacement.clone()).unwrap();
    let (mat, map) = edited.materialize();
    assert_eq!(map.forward(moved), Some(2), "back in its place");
    assert_eq!(map.forward(2), None, "the base index it replaced is gone");
    assert_eq!(map.inverse(2), Some(moved));
    let mat = EditedReconstruction::new(Arc::new(mat));
    assert!(records_agree(
        &mat.point(2).unwrap().to_record(),
        &replacement
    ));
}

#[test]
fn a_base_row_is_followed_to_wherever_its_point_now_lives() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    edited.delete_point(1).unwrap();
    let moved = edited.replace_point(4, new_record(9, 8)).unwrap();
    // A second replacement of the same point: the chain's middle link is now
    // dead, and only the last one is the answer.
    let moved_again = edited.replace_point(moved, new_record(10, 8)).unwrap();

    assert_eq!(
        edited.live_index_of_base(0),
        Some(0),
        "an untouched row is its own index"
    );
    assert_eq!(
        edited.live_index_of_base(1),
        None,
        "a deleted row has nowhere to be followed to"
    );
    assert_eq!(
        edited.live_index_of_base(4),
        Some(moved_again),
        "a twice-replaced row lands on the live end of its chain, not the dead middle"
    );
    assert_eq!(
        edited.live_index_of_base(6),
        None,
        "an addition is not a base row, so it is not this accessor's question"
    );

    // What the base rows follow to is exactly what is live and descended from
    // one, so the two accessors cannot disagree.
    for index in edited.live_indexes() {
        if index < base.point_count() as u32 {
            assert_eq!(edited.live_index_of_base(index), Some(index));
        }
    }
    assert!(edited.point(moved).is_none(), "the middle link is deleted");
}

#[test]
fn a_record_must_carry_exactly_the_base_columns() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(base);

    let mut no_frame = new_record(1, 8);
    no_frame.patch_u_halfvec = None;
    assert_eq!(
        edited.add_point(no_frame),
        Err(EditError::ColumnMismatch {
            column: "patch_u_halfvec_xyz",
            base_has: true
        })
    );

    let mut bad_image = new_record(1, 8);
    bad_image.observations[1].image_index = 99;
    assert_eq!(
        edited.add_point(bad_image),
        Err(EditError::ImageOutOfRange {
            observation: 1,
            image_index: 99,
            image_count: 8
        })
    );

    let mut empty = new_record(1, 8);
    empty.observations.clear();
    assert_eq!(edited.add_point(empty), Err(EditError::NoObservations));

    let mut wrong_patch = new_record(1, 8);
    wrong_patch.patch_bitmap = Some(Array3::<u8>::zeros((3, 3, 4)));
    assert_eq!(
        edited.add_point(wrong_patch),
        Err(EditError::PatchBitmapShape {
            got: (3, 3, 4),
            expected: (2, 2, 4)
        })
    );

    assert_eq!(edited.point_count(), 6, "a refused record adds nothing");
    assert!(edited.added.points.is_empty());
    assert!(edited.replaces.is_empty());

    assert_eq!(edited.delete_point(6), Err(EditError::NoSuchPoint(6)));
}

/// A deterministic pseudo-random edit sequence: the reads through the overlay
/// have to be the reads the materialisation gives, under the row map.
#[test]
fn overlay_reads_equal_materialised_reads_under_the_row_map() {
    for seed in 0u32..8 {
        let base = Arc::new(fixture(12));
        let mut edited = EditedReconstruction::new(Arc::clone(&base));
        let mut rng = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut next = move || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            rng >> 8
        };
        for step in 0..12u32 {
            let live: Vec<u32> = edited.live_indexes().collect();
            let pick = live[(next() as usize) % live.len()];
            match next() % 3 {
                0 => {
                    edited.delete_point(pick).unwrap();
                }
                1 => {
                    edited
                        .replace_point(pick, new_record(step + 100 * seed, 8))
                        .unwrap();
                }
                _ => {
                    edited.add_point(new_record(step + 200 * seed, 8)).unwrap();
                }
            }
        }

        let (mat, map) = edited.materialize();
        assert_eq!(mat.point_count(), edited.point_count());
        mat.validate_observation_columns().unwrap();
        mat.validate_point_columns().unwrap();
        let mat_view = EditedReconstruction::new(Arc::new(mat));

        let live: Vec<u32> = edited.live_indexes().collect();
        for &i in &live {
            let new = map.forward(i).expect("a live index lands somewhere");
            assert_eq!(map.inverse(new), Some(i), "the map inverts");
            let a = edited.point(i).unwrap().to_record();
            let b = mat_view.point(new).unwrap().to_record();
            assert!(records_agree(&a, &b), "index {i} -> {new} disagrees");
        }
        // A bijection: every survivor lands somewhere, no two land together,
        // and every materialised row has a survivor behind it.
        let landed: HashSet<u32> = live.iter().filter_map(|&i| map.forward(i)).collect();
        assert_eq!(landed.len(), live.len());
        assert_eq!(landed.len(), mat_view.point_count());
        for i in 0..edited.index_bound() {
            if edited.point(i).is_none() {
                assert_eq!(map.forward(i), None, "a dead index maps nowhere");
            }
        }
    }
}

#[test]
fn materialisation_is_deterministic_and_idempotent() {
    let base = Arc::new(fixture(8));
    let mut edited = EditedReconstruction::new(base);
    edited.delete_point(2).unwrap();
    edited.replace_point(5, new_record(1, 8)).unwrap();
    edited.add_point(new_record(2, 8)).unwrap();

    let (first, map_a) = edited.materialize();
    let (second, map_b) = edited.materialize();
    assert_eq!(map_a, map_b);
    let a = EditedReconstruction::new(Arc::new(first));
    let b = EditedReconstruction::new(Arc::new(second));
    assert_eq!(a, b, "the same value materialises to the same value");

    let (again, map_c) = a.materialize();
    assert_eq!(
        map_c.forward_dense(a.index_bound()),
        (0..a.index_bound()).map(Some).collect::<Vec<_>>(),
        "an empty overlay's row map is the identity"
    );
    assert_eq!(a, EditedReconstruction::new(Arc::new(again)));
}

#[test]
fn an_untouched_materialisation_shares_the_bitmaps() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    // A modification that leaves the patch alone: the bitmap column is the
    // base's, pointer and all.
    let mut record = edited.point(3).unwrap().to_record();
    record.point.color = [1, 2, 3];
    edited.replace_point(3, record).unwrap();
    let (mat, _) = edited.materialize();
    assert!(Arc::ptr_eq(
        mat.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap(),
        base.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap()
    ));
    assert_eq!(mat.point_set.points[3].color, [1, 2, 3]);
}

#[test]
fn the_materialised_hash_is_the_files_hash_after_a_save() {
    let base = Arc::new(fixture(8));
    let mut edited = EditedReconstruction::new(base);
    edited.delete_point(1).unwrap();
    edited.add_point(new_record(4, 8)).unwrap();
    let (mat, _) = edited.materialize();

    assert!(
        mat.content_hash.content_xxh128.is_empty(),
        "a materialised value is not the file its base came from, so it carries \
         no file's hashes"
    );
    assert!(mat.content_hash.rigs_xxh128.is_none());

    let computed = mat.content_xxh128().expect("hashable");
    let path = std::env::temp_dir().join(format!(
        "sfmtool-edited-hash-{}-{:?}.sfmr",
        std::process::id(),
        std::thread::current().id()
    ));
    mat.save(&path).expect("saved");
    let stored = sfmr_format::read_sfmr_content_hash(&path).expect("read back");
    let _ = std::fs::remove_file(&path);
    assert_eq!(computed.content_xxh128, stored.content_xxh128);
    assert_eq!(computed.points3d_xxh128, stored.points3d_xxh128);
    assert_eq!(computed.tracks_xxh128, stored.tracks_xxh128);
}

#[test]
fn a_point_edit_hash_is_a_function_of_the_records_and_the_base() {
    let base = Arc::new(fixture(6));
    let edited = EditedReconstruction::new(Arc::clone(&base));
    let one = new_record(1, 8);
    let two = new_record(2, 8);

    let h1 = edited.point_edit_hash(std::slice::from_ref(&one)).unwrap();
    assert_eq!(
        h1,
        edited.point_edit_hash(std::slice::from_ref(&one)).unwrap(),
        "the same addition on the same base hashes the same"
    );
    assert_ne!(
        h1,
        edited.point_edit_hash(std::slice::from_ref(&two)).unwrap()
    );
    assert_eq!(h1.len(), 32);

    // A different base is a different hash, because the base's own hash is the
    // first thing the edit hash covers.
    let other = EditedReconstruction::new(Arc::new(fixture(7)));
    assert_ne!(
        h1,
        other.point_edit_hash(std::slice::from_ref(&one)).unwrap()
    );
}

#[test]
fn the_hash_survives_the_clock() {
    // The write timestamp lives outside the content digest, so a value hashed
    // now and saved later agrees with the file, and two saves agree with each
    // other however much time passes between them.
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(base);
    edited.replace_point(0, new_record(6, 8)).unwrap();
    let (mat, _) = edited.materialize();
    let computed = mat.content_xxh128().expect("hashable");

    let dir = std::env::temp_dir().join(format!(
        "sfmtool-edited-clock-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let first = dir.join("first.sfmr");
    let second = dir.join("second.sfmr");
    mat.save(&first).expect("saved");
    std::thread::sleep(std::time::Duration::from_millis(20));
    mat.save(&second).expect("saved again");

    let a = sfmr_format::read_sfmr_content_hash(&first).expect("read back");
    let b = sfmr_format::read_sfmr_content_hash(&second).expect("read back");
    let stamp_a = sfmr_format::read_sfmr_metadata(&first).expect("metadata");
    let stamp_b = sfmr_format::read_sfmr_metadata(&second).expect("metadata");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(computed.content_xxh128, a.content_xxh128);
    assert_eq!(a.content_xxh128, b.content_xxh128);
    assert_ne!(stamp_a.timestamp, stamp_b.timestamp);
}

// ── The scanned row map ─────────────────────────────────────────────────

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
