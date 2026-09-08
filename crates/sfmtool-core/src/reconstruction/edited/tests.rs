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
