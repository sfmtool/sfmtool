// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The overlay's contract: the base is never written, indexes are stable, a
//! read through the overlay is the read the materialisation gives under the row
//! map, and a materialised value's hash is the file's after a save.

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array2, Array3, Array4};

use sfmtool_sfmr_format::{NO_REFERENCE_IMAGE, POINT_CONSTRAINT_FREE, POINT_CONSTRAINT_RANGED};

use super::*;
use crate::reconstruction::data::ObservationSource;

/// A demo reconstruction carrying *every* optional column, so a record that
/// omits or invents one is a failure the tests can see.
pub(super) fn fixture(num_points: usize) -> SfmrReconstruction {
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
    // Every demo point has two observations.
    recon.point_set.reference_observations = Some(
        (0..p)
            .map(|i| if i % 3 == 0 { -1 } else { (i % 2) as i32 })
            .collect(),
    );
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
pub(super) fn new_record(seed: u32, image_count: u32) -> PointRecord {
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
                reading: None,
            },
            RecordObservation {
                image_index: second,
                feature_index: Some(2000 + seed),
                keypoint_xy: Some([3.0, seed as f32]),
                confidence: Some(201),
                reading: None,
            },
        ],
        patch_u_halfvec: Some([1.0, 0.0, 0.0]),
        patch_v_halfvec: Some([0.0, 1.0, 0.0]),
        patch_bitmap: Some(Array3::<u8>::from_shape_fn((2, 2, 4), |(y, x, c)| {
            ((seed as usize + y + x + c) % 256) as u8
        })),
        normal_confidence: Some(42),
        constraint: Some((POINT_CONSTRAINT_FREE, f64::NAN, NO_REFERENCE_IMAGE)),
        reference_observation: Some(1),
        display_only_reference: false,
    }
}

/// The overlay's count of points at infinity follows every kind of edit: an
/// addition at infinity, a position replaced by a bearing, a bearing replaced
/// by a position, a deleted base bearing, and a deleted addition at infinity.
#[test]
fn the_infinity_count_reads_through_the_overlay() {
    let mut base = fixture(10);
    for p in [2usize, 3, 4] {
        let point = &mut base.point_set.points[p];
        point.position = nalgebra::Point3::from(point.position.coords.normalize());
        point.w = 0.0;
    }
    base.rebuild_derived_fields();
    let images = base.image_count() as u32;
    let mut edited = EditedReconstruction::new(Arc::new(base));
    assert_eq!(edited.infinity_point_count(), 3);

    let bearing = |seed| {
        let mut record = new_record(seed, images);
        record.point.position = nalgebra::Point3::new(0.0, 0.6, 0.8);
        record.point.w = 0.0;
        record
    };
    let added = edited.add_point(bearing(1)).unwrap();
    assert_eq!(edited.infinity_point_count(), 4);
    // A position replaced by a bearing.
    edited.replace_point(0, bearing(2)).unwrap();
    assert_eq!(edited.infinity_point_count(), 5);
    // A bearing replaced by a position.
    edited.replace_point(2, new_record(3, images)).unwrap();
    assert_eq!(edited.infinity_point_count(), 4);
    // A base bearing deleted, and an added one.
    edited.delete_point(3).unwrap();
    edited.delete_point(added).unwrap();
    assert_eq!(edited.infinity_point_count(), 2);
    let (materialised, _) = edited.materialize();
    assert_eq!(materialised.point_set.infinity_point_count, 2);
}

/// The record comparison: exact on every stored column, with `NaN` agreeing
/// with `NaN` so that a free point's distance -- or any other column a `NaN`
/// reached -- cannot make a record differ from a copy of itself forever.
#[test]
fn a_record_agrees_with_itself_and_with_nothing_that_moved() {
    let record = new_record(5, 8);
    assert!(record.agrees_with(&record.clone()));

    let mut nan = record.clone();
    nan.point.position.x = f64::NAN;
    nan.point.normal.y = f32::NAN;
    nan.observations[0].keypoint_xy = Some([f32::NAN, 7.0]);
    assert!(
        nan.agrees_with(&nan.clone()),
        "a NaN column has to agree with itself"
    );
    assert!(!nan.agrees_with(&record), "a NaN is not the number it was");

    // One column at a time, each moved by the smallest amount the stored
    // representation holds.
    let mut moved = record.clone();
    moved.point.position.z = f64::from_bits(record.point.position.z.to_bits() + 1);
    assert!(!moved.agrees_with(&record), "the position");
    let mut moved = record.clone();
    moved.point.error = f32::from_bits(record.point.error.to_bits() + 1);
    assert!(!moved.agrees_with(&record), "the error");
    let mut moved = record.clone();
    moved.point.w = 0.0;
    assert!(!moved.agrees_with(&record), "the w");
    let mut moved = record.clone();
    moved.point.color[1] += 1;
    assert!(!moved.agrees_with(&record), "the colour");
    let mut moved = record.clone();
    moved.patch_u_halfvec = Some([f32::from_bits(1.0f32.to_bits() + 1), 0.0, 0.0]);
    assert!(!moved.agrees_with(&record), "the frame");
    let mut moved = record.clone();
    moved.patch_bitmap.as_mut().expect("the column")[[0, 0, 0]] ^= 1;
    assert!(!moved.agrees_with(&record), "the bitmap");
    let mut moved = record.clone();
    moved.normal_confidence = Some(41);
    assert!(!moved.agrees_with(&record), "the normal confidence");
    let mut moved = record.clone();
    moved.constraint = Some((POINT_CONSTRAINT_RANGED, 1.0, 0));
    assert!(!moved.agrees_with(&record), "the constraint");
    let mut moved = record.clone();
    moved.observations[1].confidence = Some(202);
    assert!(!moved.agrees_with(&record), "an observation's confidence");
    let mut moved = record.clone();
    moved.observations[1].keypoint_xy = Some([f32::from_bits(3.0f32.to_bits() + 1), 5.0]);
    assert!(!moved.agrees_with(&record), "a keypoint");
    let mut moved = record.clone();
    moved.observations.remove(1);
    assert!(!moved.agrees_with(&record), "an observation dropped");
    let mut moved = record.clone();
    moved.observations.swap(0, 1);
    assert!(!moved.agrees_with(&record), "the stored order");
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

/// The observation count is the version's, not the base's: an edit adds and
/// removes whole tracks, and a reader asking the base would be told what the
/// file held.
#[test]
fn the_observation_count_follows_every_point_edit() {
    let base = Arc::new(fixture(6));
    let mut edited = EditedReconstruction::new(Arc::clone(&base));
    let stored = base.point_set.tracks.len();
    assert_eq!(edited.observation_count(), stored);

    let deleted = edited.point(0).unwrap().observations().len();
    edited.delete_point(0).unwrap();
    assert_eq!(edited.observation_count(), stored - deleted);

    // An addition brings its own track, and a replacement brings its own and
    // takes the replaced point's.
    let added = new_record(5, 8);
    let brought = added.observations.len();
    edited.add_point(added).unwrap();
    assert_eq!(edited.observation_count(), stored - deleted + brought);

    let replaced = edited.point(3).unwrap().observations().len();
    edited.replace_point(3, new_record(7, 8)).unwrap();
    assert_eq!(
        edited.observation_count(),
        stored - deleted + 2 * brought - replaced
    );
    // Deleting an addition takes back the track it brought.
    let addition = base.point_count() as u32;
    edited.delete_point(addition).unwrap();
    assert_eq!(
        edited.observation_count(),
        stored - deleted + brought - replaced
    );
    // The materialised value agrees, which is the answer with no overlay in it.
    let (mat, _) = edited.materialize();
    assert_eq!(mat.point_set.tracks.len(), edited.observation_count());
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
        assert!(view.to_record().agrees_with(&before[i as usize]));
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
    assert!(mat.point(2).unwrap().to_record().agrees_with(&replacement));
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
            assert!(a.agrees_with(&b), "index {i} -> {new} disagrees");
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
    let stored = sfmtool_sfmr_format::read_sfmr_content_hash(&path).expect("read back");
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

    let a = sfmtool_sfmr_format::read_sfmr_content_hash(&first).expect("read back");
    let b = sfmtool_sfmr_format::read_sfmr_content_hash(&second).expect("read back");
    let stamp_a = sfmtool_sfmr_format::read_sfmr_metadata(&first).expect("metadata");
    let stamp_b = sfmtool_sfmr_format::read_sfmr_metadata(&second).expect("metadata");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(computed.content_xxh128, a.content_xxh128);
    assert_eq!(a.content_xxh128, b.content_xxh128);
    assert_ne!(stamp_a.timestamp, stamp_b.timestamp);
}

// ── The base's identity ─────────────────────────────────────────────────

/// A base that arrived with hashes is taken at its word, and never re-derives
/// them.
///
/// Asserted the only way it can be observed from outside: the value's content
/// is changed underneath the stored hash, and the hash does not move. That is
/// the contract rather than an accident of caching -- a file hands over its own
/// hashes, and re-deriving them would spend a serialisation of the whole value
/// to answer a question already answered. Checking them against the bytes is
/// `verify_sfmr`'s job, asked for on purpose.
#[test]
fn a_base_that_came_with_a_hash_is_taken_at_its_word() {
    let mut recon = fixture(6);
    recon.content_hash.content_xxh128 = "0".repeat(32);
    // Something a re-derivation could not possibly agree with.
    recon.point_set.points[0].position.x += 1234.5;

    let edited = EditedReconstruction::new(Arc::new(recon));

    assert_eq!(
        edited
            .base_content_hash()
            .expect("a stored hash")
            .content_xxh128,
        "0".repeat(32),
        "the stored hash was recomputed rather than believed",
    );
}

/// A base with no stored hash is a value materialised from an edit, and it gets
/// the hash a write of it would store.
#[test]
fn a_base_with_no_stored_hash_is_computed() {
    let mut recon = fixture(6);
    recon.content_hash = sfmtool_sfmr_format::ContentHash::default();
    let expected = recon.content_xxh128().expect("hashable").content_xxh128;

    let edited = EditedReconstruction::new(Arc::new(recon));

    let got = &edited.base_content_hash().expect("hashable").content_xxh128;
    assert_eq!(*got, expected);
    assert_eq!(got.len(), 32, "a real hash, not the empty placeholder");
}

// ── The point map ───────────────────────────────────────────────────────

#[test]
fn a_removal_map_keeps_every_surviving_index_where_it_was() {
    let map = PointMap::Removed(vec![2, 5]);
    assert_eq!(map.forward(1), Some(1));
    assert_eq!(map.forward(2), None);
    assert_eq!(map.forward(6), Some(6));
    assert_eq!(map.inverse(6), Some(6));
    assert_eq!(map.inverse(5), None);
}

/// A point-mask filter over `recon`, and the map for it: the one
/// [`RowMap::by_scan`] reads off the filter's input and output, which is how a
/// bulk edit's map is made.
fn dropping(recon: &SfmrReconstruction, keep: &[bool]) -> (SfmrReconstruction, PointMap) {
    let after = recon.filter_points_by_mask(keep);
    let map = RowMap::by_scan(recon, &after, None).expect("a filter keeps point order");
    (after, PointMap::Rows(map))
}

#[test]
fn a_row_map_closes_up_behind_what_it_removed_and_inverts() {
    let removed = [0u32, 2, 3];
    let before = SfmrReconstruction::demo(6);
    let keep: Vec<bool> = (0..6).map(|i| !removed.contains(&i)).collect();
    let (_, map) = dropping(&before, &keep);
    // Survivors 1, 4, 5 become 0, 1, 2.
    assert_eq!(map.forward(1), Some(0));
    assert_eq!(map.forward(4), Some(1));
    assert_eq!(map.forward(5), Some(2));
    assert_eq!(map.forward(0), None);
    assert_eq!(map.forward(3), None);
    // And back, landing on a live slot every time.
    for (new, old) in [(0u32, 1u32), (1, 4), (2, 5)] {
        assert_eq!(map.inverse(new), Some(old), "inverse of {new}");
        assert!(!removed.contains(&old));
    }
}

#[test]
fn a_chain_applies_its_steps_in_order_and_inverts_in_reverse() {
    // Remove 1 of 5, then remove what was 3 (2 after the first step).
    let before = SfmrReconstruction::demo(5);
    let (middle, first) = dropping(&before, &[true, false, true, true, true]);
    let (_, second) = dropping(&middle, &[true, true, false, true]);
    let chain = PointMap::Chain(vec![first, second]);
    assert_eq!(chain.forward(0), Some(0));
    assert_eq!(chain.forward(1), None);
    assert_eq!(chain.forward(2), Some(1));
    assert_eq!(chain.forward(3), None);
    assert_eq!(chain.forward(4), Some(2));
    for (new, old) in [(0u32, 0u32), (1, 2), (2, 4)] {
        assert_eq!(chain.inverse(new), Some(old));
    }
}

#[test]
fn a_replacement_map_moves_the_named_index_and_no_other() {
    // Delete-and-re-add gives a modified point a new index while it stays the
    // same point, so this is what carries a selection across such an edit.
    let map = PointMap::Replaced(vec![(3, 40), (7, 41)]);
    assert_eq!(map.forward(3), Some(40));
    assert_eq!(map.forward(7), Some(41));
    assert_eq!(map.inverse(40), Some(3));
    assert_eq!(map.inverse(41), Some(7));
    // Everything not named is unchanged, which is what makes the map the size
    // of the edit rather than the size of the reconstruction.
    for index in [0, 4, 39, 42] {
        assert_eq!(map.forward(index), Some(index));
        assert_eq!(map.inverse(index), Some(index));
    }
}

#[test]
fn a_replacement_map_round_trips_through_a_chain() {
    let map = PointMap::Chain(vec![
        PointMap::Replaced(vec![(3, 40)]),
        PointMap::Replaced(vec![(40, 41)]),
    ]);
    assert_eq!(map.forward(3), Some(41));
    assert_eq!(map.inverse(41), Some(3));
}

#[test]
fn a_creation_map_is_the_identity_forward_and_has_no_inverse_for_what_it_made() {
    let map = PointMap::Created(vec![40, 41]);
    for index in [0, 3, 40, 41, 42] {
        assert_eq!(map.forward(index), Some(index));
    }
    // The created indexes are what an undo has no answer for, which is how it
    // drops a selection sitting on one rather than carrying it back to an
    // index that held nothing.
    assert_eq!(map.inverse(40), None);
    assert_eq!(map.inverse(41), None);
    assert_eq!(map.inverse(39), Some(39));
    assert_eq!(map.inverse(42), Some(42));
}
