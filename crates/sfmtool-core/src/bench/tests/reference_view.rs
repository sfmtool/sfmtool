// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view readings an evaluation writes, on a real track: the
//! seoul_bull ground truth and its photographs, which are checked in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array3;

use crate::bench::evaluate::open_localizer;
use crate::bench::{
    apply_thresholds, bitmap_target, commit, create_track, evaluate, evaluate_rendering_bitmap,
    fit, pin_verdicts, render_bitmap_in_place, score_bitmap, set_reference, set_stage, set_verdict,
    sight_observation, split, tilt_patch, unpin_verdicts, verdicts_if_unpinned, Bench, BenchItem,
    CreateTrackOptions, EditableTrack, EvaluateOptions, FitOptions, Provenance, StageKind,
    TrackEditError, Unmeasured, Verdict,
};
use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::render_tile;
use crate::camera::warp_map::patch_grid_jacobian;
use crate::camera::{PhotographCache, WarpMap};
use crate::geometry::RigidTransform;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
use crate::patch::keypoint_localize::{localize_patch_keypoints, KeypointLocalizeParams};
use crate::patch::keypoint_subpixel::{
    fuse_patch_bitmap, refine_patch_keypoints, KeypointSubpixelParams,
};
use crate::patch::member_coherence::{member_zncc_matrix, MemberCoherenceParams};
use crate::patch::normal_refine::{refine_patch_normal, NormalRefineParams};
use crate::patch::normal_refine::{PatchWindow, ProjectedImage};
use crate::patch::reference_view::{
    finite_middle, render_view_tile, ReferenceFallback, ReferenceTest, ViewTile,
    REFERENCE_MAX_VIEWING_ANGLE_DEG, REFERENCE_MIN_COVERAGE,
};
use crate::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityEllipseUnits, SelfSimilarityParams,
};
use crate::patch::stored_bitmap::{bitmap_from_tile, bitmap_planes, BitmapScorer};
use crate::patch::stored_bitmap::{render_patch_bitmap, render_reference};
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;
use crate::reconstruction::SfmrReconstruction;

/// The point of the ground truth the tests read: the first with at least
/// eight observations.
const MIN_TRACK: usize = 8;

/// The seoul_bull ground truth with its decoded photographs and poses.
struct GroundTruth {
    recon: SfmrReconstruction,
    pyramids: Vec<Arc<ImageU8Pyramid>>,
    poses: Vec<RigidTransform>,
}

impl GroundTruth {
    fn load() -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-data/images/seoul_bull_sculpture");
        let recon = SfmrReconstruction::load(
            &dir.join("seoul_bull_sculpture_ground_truth.sfmr"),
            &Progress::none(),
        )
        .expect("the ground truth loads");
        let images = &recon.image_table.images;
        let paths: Vec<PathBuf> = images.iter().map(|i| dir.join(&i.name)).collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (pyramids, _) = PhotographCache::new(0, 8)
            .get_many(&refs, &Progress::none())
            .expect("nothing cancels the decode");
        let pyramids = pyramids
            .into_iter()
            .map(|p| p.expect("every photograph is checked in"))
            .collect();
        let poses = images
            .iter()
            .map(|image| {
                let q = image.quaternion_wxyz;
                let t = image.translation_xyz;
                RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
            })
            .collect();
        Self {
            recon,
            pyramids,
            poses,
        }
    }

    fn views(&self) -> Vec<ProjectedImage<'_>> {
        self.recon
            .image_table
            .images
            .iter()
            .zip(&self.poses)
            .zip(&self.pyramids)
            .map(|((image, cam_from_world), pyramid)| ProjectedImage {
                camera: &self.recon.image_table.cameras[image.camera_index as usize],
                cam_from_world,
                pyramid,
            })
            .collect()
    }

    /// The first point with at least [`MIN_TRACK`] observations, put on the
    /// bench and evaluated with the default options.
    fn evaluated_track(&self) -> (EditableTrack, EditedReconstruction) {
        self.evaluated_track_with(&EvaluateOptions::default())
    }

    /// [`Self::evaluated_track`] with `options`.
    fn evaluated_track_with(
        &self,
        options: &EvaluateOptions,
    ) -> (EditableTrack, EditedReconstruction) {
        let offsets = &self.recon.point_set.observation_offsets;
        let point = (0..self.recon.point_count())
            .find(|&p| offsets[p + 1] - offsets[p] >= MIN_TRACK)
            .expect("the ground truth has a long track") as u32;
        let edited = EditedReconstruction::new(Arc::new(self.recon.clone()));
        let (bench, report) = create_track(
            &Bench::new(),
            &edited,
            point,
            &CreateTrackOptions::default(),
        )
        .expect("the point is live");
        let track = bench.track(&report.label).expect("just put on");
        let (read, _) = evaluate(track, &edited, &self.views(), options, &Progress::none())
            .expect("the track reads");
        (read, edited)
    }
}

/// The keypoint a track-stage row sits at.
fn keypoint_of(track: &EditableTrack, i: usize) -> [f64; 2] {
    let k = track.observations[i]
        .track
        .as_ref()
        .and_then(|m| m.keypoint)
        .expect("every row of a ground-truth track has a keypoint");
    [f64::from(k[0]), f64::from(k[1])]
}

#[test]
fn an_evaluation_picks_a_whole_view_facing_the_patch_on_a_ground_truth_track() {
    let truth = GroundTruth::load();
    let (read, _) = truth.evaluated_track();

    let mut picked = Vec::new();
    for (i, observation) in read.observations.iter().enumerate() {
        let m = observation.track.as_ref().expect("every row is read");
        assert_eq!(observation.verdict, Verdict::In);
        // Every row carries the per-view readings and a standing.
        assert!(m.viewing_angle_deg.is_some(), "row {i}");
        assert!(m.coverage.is_some(), "row {i}");
        assert!(m.clipped_share.is_some(), "row {i}");
        assert!(m.pair_zncc_grid.is_some(), "row {i}");
        let standing = m.reference_view.expect("an in row has a standing");
        assert_eq!(standing.fallback, ReferenceFallback::None);
        if standing.is_reference() {
            picked.push(i);
        }
    }
    assert_eq!(picked.len(), 1, "one reference view");
    let reference = read.observations[picked[0]].track.as_ref().unwrap();
    assert!(reference.coverage.unwrap() >= REFERENCE_MIN_COVERAGE);
    assert!(reference.viewing_angle_deg.unwrap() <= REFERENCE_MAX_VIEWING_ANGLE_DEG);
    assert!(reference.pair_zncc.is_some());
    // The reference is the sharpest of the views that passed every other test.
    let radius = |i: usize| {
        read.observations[i]
            .track
            .as_ref()
            .and_then(|m| m.zncc_self_similarity_radius)
            .unwrap()
    };
    for (i, observation) in read.observations.iter().enumerate() {
        let standing = observation.track.as_ref().unwrap().reference_view.unwrap();
        if standing.rejected_by == Some(ReferenceTest::Sharpness) {
            assert!(radius(i) >= radius(picked[0]), "row {i}");
        }
    }
}

/// The self-similarity readings come from the tile the reference view's
/// readings share. They are the readings of the tile rendered the way they
/// were before that tile was shared: the patch anchored on the keypoint, the
/// sampler the sampler rule picks from that placement's Jacobian, black off
/// the photograph, and the self-similarity read the overlap way over the
/// samples on the photograph.
#[test]
fn the_shared_tile_leaves_the_self_similarity_readings_as_a_direct_render_gives_them() {
    let truth = GroundTruth::load();
    let (read, edited) = truth.evaluated_track();
    let views = truth.views();
    let options = EvaluateOptions::default();
    let resolution = options.patch_resolution(&edited.base) as usize;
    let frame = read
        .track()
        .and_then(|t| t.placement.clone())
        .expect("a track-stage frame");
    for (i, observation) in read.observations.iter().enumerate() {
        let view = &views[observation.image as usize];
        let anchored =
            frame.anchored_at_keypoint(view.camera, view.cam_from_world, keypoint_of(&read, i));
        let placement = anchored.as_ref().unwrap_or(&frame);
        let jacobian = patch_grid_jacobian(placement, view.camera, view.cam_from_world, resolution);
        let sampler = options.localize.sampler.for_jacobian(jacobian);
        let mut map = WarpMap::from_patch(
            placement,
            view.camera,
            view.cam_from_world,
            resolution as u32,
        );
        let rendered = render_tile(view.pyramid, &mut map, sampler);
        let valid: Vec<bool> = (0..resolution * resolution)
            .map(|k| map.is_valid((k % resolution) as u32, (k / resolution) as u32))
            .collect();
        let channels = view.pyramid.level(0).channels() as usize;
        let samples = Array3::from_shape_fn((resolution, resolution, channels), |(r, c, k)| {
            if k >= 3 {
                u8::MAX
            } else if rendered.channels() >= 3 {
                rendered.get_pixel(c as u32, r as u32, k as u32)
            } else {
                rendered.get_pixel(c as u32, r as u32, 0)
            }
        });
        let values: Vec<f32> = samples.iter().map(|&v| f32::from(v)).collect();
        let (planes, colour) =
            PatchTile::planes_from_interleaved(&values, resolution, resolution, channels);
        let tile = PatchTile {
            values: &planes,
            channels: colour,
            width: resolution,
            height: resolution,
        };
        let parts =
            zncc_self_similarity_parts(&tile, Some(&valid), &SelfSimilarityParams::default());
        let ellipse =
            SelfSimilarityEllipseUnits::read(&parts.whole, jacobian, Some(placement), resolution);

        // Compared as printed, so that a `NaN` (a circle's major angle, a
        // textureless cell) matches itself.
        let same = |a: &dyn std::fmt::Debug, b: &dyn std::fmt::Debug| {
            assert_eq!(format!("{a:?}"), format!("{b:?}"), "row {i}");
        };
        let m = observation.track.as_ref().expect("every row is read");
        same(&m.zncc_self_similarity_radius, &Some(parts.whole.radius));
        same(
            &m.zncc_self_similarity_radius_middle,
            &Some(parts.middle.radius),
        );
        let grid = parts
            .grid
            .each_ref()
            .map(|row| row.each_ref().map(|cell| cell.radius));
        same(&m.zncc_self_similarity_radius_grid, &Some(grid));
        same(&m.zncc_self_similarity_ellipse, &ellipse);
    }
}

/// The pair ZNCC the bench reports is the median of the row's entries in
/// member coherence's matrix, built over the track's `in` rows anchored at
/// their keypoints with the evaluation's resolution and sampler.
#[test]
fn the_pair_zncc_is_the_median_of_member_coherence_s_row_called_directly() {
    let truth = GroundTruth::load();
    let (read, edited) = truth.evaluated_track();
    let views = truth.views();
    let options = EvaluateOptions::default();
    let frame = read
        .track()
        .and_then(|t| t.placement.clone())
        .expect("a track-stage frame");
    let rows: Vec<usize> = (0..read.observations.len())
        .filter(|&i| read.observations[i].verdict == Verdict::In)
        .collect();
    let members: Vec<u32> = rows.iter().map(|&i| read.observations[i].image).collect();
    let keypoints: Vec<Option<[f64; 2]>> =
        rows.iter().map(|&i| Some(keypoint_of(&read, i))).collect();
    let params = MemberCoherenceParams {
        resolution: options.patch_resolution(&edited.base),
        sampler: options.localize.sampler,
        ..MemberCoherenceParams::default()
    };
    let matrix = member_zncc_matrix(&frame, &views, &members, Some(&keypoints), &params);
    assert_eq!(matrix.members, members);
    for (k, &i) in rows.iter().enumerate() {
        let others: Vec<f64> = (0..matrix.len())
            .filter(|&j| j != k)
            .map(|j| matrix.get(k, j))
            .collect();
        let median = finite_middle(&others);
        let reported = read.observations[i]
            .track
            .as_ref()
            .and_then(|m| m.pair_zncc);
        assert_eq!(reported, median.is_finite().then_some(median), "row {i}");
    }
}

/// The track-stage tile of row `i`, rendered as the evaluation renders it.
fn tile_of_row(
    read: &EditableTrack,
    views: &[ProjectedImage<'_>],
    i: usize,
    resolution: usize,
) -> ViewTile {
    let frame = read
        .track()
        .and_then(|t| t.placement.clone())
        .expect("a track-stage frame");
    render_view_tile(
        &frame,
        &views[read.observations[i].image as usize],
        Some(keypoint_of(read, i)),
        resolution,
        EvaluateOptions::default().localize.sampler,
        &Progress::none(),
    )
}

/// A track whose bitmap names no reference observation (a fused mean, or a
/// bitmap from before the reference was recorded) scores every row against
/// that bitmap, plain and blur-matched, as the scorer does directly.
#[test]
fn every_row_is_scored_against_the_stored_bitmap() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (read, edited) = truth.evaluated_track();
    // The minimal ground truth stores no bitmaps; render one and forget which
    // row it came from.
    let mut rendered = render_bitmap_in_place(&read, &edited, &views, &FitOptions::default());
    if let crate::bench::Stage::Track(payload) = &mut rendered.stage {
        payload.reference = None;
    }
    let (read, _) = evaluate(
        &rendered,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    let payload = read.track().expect("a track stage");
    let bitmap = payload.bitmap.as_ref().expect("a rendered bitmap");
    let resolution = EvaluateOptions::default().patch_resolution(&edited.base) as usize;
    let samples: Vec<u8> = bitmap.iter().copied().collect();
    let planes = bitmap_planes(&samples, resolution);
    let mut scorer = BitmapScorer::new(&planes, PatchWindow::GaussianDisk { sigma: 0.6 });
    for i in 0..read.observations.len() {
        let m = read.observations[i].track.as_ref().unwrap();
        let ellipse = m.zncc_self_similarity_ellipse.map(|e| e.grid_px.matrix);
        let direct = scorer.score(&tile_of_row(&read, &views, i, resolution).planes(), ellipse);
        assert_eq!(m.zncc, Some(direct.zncc), "row {i}");
        assert_eq!(
            m.blur_matched_zncc,
            Some(direct.blur_matched_zncc),
            "row {i}"
        );
        assert_eq!(m.bitmap_blur_sigma, Some(direct.blur_sigma), "row {i}");
        assert_eq!(
            m.sharper_than_bitmap,
            Some(direct.sharper_than_bitmap),
            "row {i}"
        );
        // A view is at least as close to a bitmap blurred to its sharpness.
        if direct.blur_sigma > 0.0 {
            assert!(direct.blur_matched_zncc > direct.zncc - 0.02, "row {i}");
        }
    }
}

/// Rendering the bitmap where the track stands stores the tile of the row the
/// evaluation's reference-view rule picks, names that row, and the next
/// evaluation scores it 1 without computing it; a commit writes its place in
/// the stored track.
#[test]
fn the_rendered_bitmap_is_the_picked_row_s_tile_and_the_commit_records_it() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (read, edited) = truth.evaluated_track();
    let picked = read
        .observations
        .iter()
        .position(|o| {
            o.track
                .as_ref()
                .and_then(|m| m.reference_view)
                .is_some_and(|s| s.is_reference())
        })
        .expect("the rule picks one");
    let rendered = render_bitmap_in_place(&read, &edited, &views, &FitOptions::default());
    let payload = rendered.track().unwrap();
    assert_eq!(payload.reference, Some(picked));
    let resolution = EvaluateOptions::default().patch_resolution(&edited.base) as usize;
    let tile = tile_of_row(&read, &views, picked, resolution);
    let rgba = bitmap_from_tile(&tile);
    let stored: Vec<u8> = payload.bitmap.as_ref().unwrap().iter().copied().collect();
    assert_eq!(stored, rgba);
    // Alpha marks the samples on the photograph, which the stored bitmap's
    // readers take as data.
    assert!(rgba
        .as_chunks::<4>()
        .0
        .iter()
        .all(|p| p[3] == 255 || p[3] == 0));

    let (again, _) = evaluate(
        &rendered,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    for (i, o) in again.observations.iter().enumerate() {
        let m = o.track.as_ref().unwrap();
        if i == picked {
            assert_eq!(m.zncc, Some(1.0));
            assert_eq!(m.sharper_than_bitmap, None);
        } else {
            let z = m.zncc.expect("a scored row");
            assert!(z < 1.0 && z > 0.0, "row {i}: {z}");
        }
    }

    let (committed, report) = commit(&edited, &again).expect("the track commits");
    let view = committed.point(report.point).expect("the committed point");
    let images: Vec<u32> = view.observations().iter().map(|o| o.image_index).collect();
    let at = view.reference_observation().expect("the column is carried") as usize;
    assert_eq!(images[at], again.observations[picked].image);
}

/// The bench reads a point's reference with or without a bitmap: the column
/// names the observation the bitmap is, or is to be, rendered from. With a
/// column rendered for display, a pick only the display render made reaches
/// the bench like a stored reference, a save writes it as `-1`, and a commit
/// makes it the committed point's own.
#[test]
fn a_reference_reaches_the_bench_with_or_without_a_bitmap() {
    let truth = GroundTruth::load();
    let mut recon = truth.recon.clone();
    assert!(recon.point_set.patch_bitmaps_y_x_rgba.is_none());
    let offsets = &recon.point_set.observation_offsets;
    let mut long = (0..recon.point_count()).filter(|&p| offsets[p + 1] - offsets[p] >= MIN_TRACK);
    let stored = long.next().expect("the ground truth has a long track");
    let picked = long.next().expect("the ground truth has two long tracks");
    let mut references = vec![sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION; recon.point_count()];
    references[stored] = 2;
    recon.point_set.reference_observations = Some(references.clone());

    let put_on = |edited: &EditedReconstruction, point: usize| {
        let (bench, report) = create_track(
            &Bench::new(),
            edited,
            point as u32,
            &CreateTrackOptions::default(),
        )
        .expect("the point is live");
        bench.track(&report.label).expect("just put on").clone()
    };

    // With no bitmap the reference is still read: a later render renders
    // from it.
    let bare = EditedReconstruction::new(Arc::new(recon.clone()));
    let track = put_on(&bare, stored);
    assert_eq!(track.track().unwrap().reference, Some(2));
    assert!(track.track().unwrap().bitmap.is_none());
    let (committed, report) = commit(&bare, &track).expect("the track commits");
    let view = committed.point(report.point).expect("the point");
    assert_eq!(view.reference_observation(), Some(2));

    // A display column: `picked` was -1 in the file, and the display render
    // picked observation 1 for it.
    references[picked] = 1;
    let mut marks = vec![false; recon.point_count()];
    marks[picked] = true;
    recon.point_set.reference_observations = Some(references);
    recon.point_set.display_only_references = Some(marks);
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(ndarray::Array4::zeros((
        recon.point_count(),
        4,
        4,
        4,
    ))));
    recon.point_set.patch_bitmaps_for_display = true;
    let edited = EditedReconstruction::new(Arc::new(recon));
    assert_eq!(put_on(&edited, stored).track().unwrap().reference, Some(2));
    let track = put_on(&edited, picked);
    assert_eq!(track.track().unwrap().reference, Some(1));
    assert!(track.track().unwrap().bitmap.is_some());

    // A save writes the file's references, not the display pick, and no
    // bitmaps.
    let saved = edited.materialize().0.to_sfmr_data();
    assert!(saved.patch_bitmaps_y_x_rgba.is_none());
    let column = saved.reference_observations.expect("framed");
    assert_eq!(column[stored], 2);
    assert_eq!(
        column[picked],
        sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION
    );

    // A commit makes the bench's reference the committed point's own.
    let (committed, report) = commit(&edited, &track).expect("the track commits");
    assert_eq!(
        committed
            .point(report.point)
            .expect("the point")
            .reference_observation(),
        Some(1)
    );
    let saved = committed.materialize().0.to_sfmr_data();
    let column = saved.reference_observations.expect("framed");
    let mut named: Vec<i32> = column.iter().copied().filter(|&r| r >= 0).collect();
    named.sort_unstable();
    assert_eq!(named, vec![1, 2]);
}

#[test]
fn committing_an_unedited_display_pick_saves_it() {
    let truth = GroundTruth::load();
    let mut recon = truth.recon.clone();
    let offsets = &recon.point_set.observation_offsets;
    let point = (0..recon.point_count())
        .find(|&p| offsets[p + 1] - offsets[p] >= MIN_TRACK)
        .expect("the ground truth has a long track");
    let mut references = vec![sfmtool_sfmr_format::NO_REFERENCE_OBSERVATION; recon.point_count()];
    references[point] = 2;
    recon.point_set.reference_observations = Some(references);

    // A first commit settles the point on what the bench writes.
    let edited = EditedReconstruction::new(Arc::new(recon));
    let (bench, report) = create_track(
        &Bench::new(),
        &edited,
        point as u32,
        &CreateTrackOptions::default(),
    )
    .expect("the point is live");
    let track = bench.track(&report.label).expect("just put on").clone();
    assert_eq!(track.track().unwrap().reference, Some(2));
    let (next, first) = commit(&edited, &track).expect("the track commits");
    let (mut settled_recon, map) = next.materialize();
    let settled_point = map.forward(first.point).expect("the point is live");
    let settled = track.with_origin(1, settled_point);

    // A stored reference: committing the unedited track again writes nothing.
    let unmarked = EditedReconstruction::new(Arc::new(settled_recon.clone()));
    let (_, report) = commit(&unmarked, &settled).expect("the track commits");
    assert!(!report.changed);

    // The same reference picked only by the display render: the commit saves
    // the reference the bench holds, rather than leaving the mark to save -1.
    let mut marks = vec![false; settled_recon.point_count()];
    marks[settled_point as usize] = true;
    settled_recon.point_set.display_only_references = Some(marks);
    let marked = EditedReconstruction::new(Arc::new(settled_recon));
    let saved = marked.materialize().0.to_sfmr_data();
    let column = saved.reference_observations.expect("framed");
    assert!(column.iter().all(|&r| r < 0), "the mark saves -1");
    let (committed, report) = commit(&marked, &settled).expect("the track commits");
    assert!(report.changed);
    let view = committed.point(report.point).expect("the point");
    assert!(!view.display_only_reference());
    let saved = committed.materialize().0.to_sfmr_data();
    let column = saved.reference_observations.expect("framed");
    let named: Vec<i32> = column.iter().copied().filter(|&r| r >= 0).collect();
    assert_eq!(named, vec![2]);
}

// ---- The last fallback: the fused mean stands ------------------------------

/// A ground-truth point the reference-view rule reaches only through its last
/// fallback ([`ReferenceFallback::WithoutAny`]), with a fused mean of its views
/// that renders, so the fused mean is stored with no reference.
const WITHOUT_ANY_POINT: usize = 53;

/// Point `p`'s patch, its images and its stored keypoints.
fn point_inputs(recon: &SfmrReconstruction, p: usize) -> (OrientedPatch, Vec<u32>, Vec<[f64; 2]>) {
    let cloud = PatchCloud::from_stored_frames(recon).expect("patch frames");
    let kxy = recon.keypoints_xy().expect("inline keypoints");
    let offsets = &recon.point_set.observation_offsets;
    let rows = offsets[p]..offsets[p + 1];
    let images = rows
        .clone()
        .map(|o| recon.point_set.tracks[o].image_index)
        .collect();
    let keypoints = rows
        .map(|o| [f64::from(kxy[[o, 0]]), f64::from(kxy[[o, 1]])])
        .collect();
    (cloud.patch(p).clone(), images, keypoints)
}

fn anchors(keypoints: &[[f64; 2]]) -> Vec<Option<[f64; 2]>> {
    keypoints.iter().map(|&k| Some(k)).collect()
}

#[test]
fn a_last_fallback_pick_stores_the_fused_mean_with_no_reference() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (patch, images, keypoints) = point_inputs(&truth.recon, WITHOUT_ANY_POINT);
    let params = KeypointSubpixelParams {
        resolution: 24,
        ..Default::default()
    };
    let render = render_reference(
        &patch,
        &views,
        &images,
        &anchors(&keypoints),
        24,
        params.sampler,
        &Progress::none(),
    );
    assert_eq!(
        render.reading.choice.fallback,
        ReferenceFallback::WithoutAny
    );
    assert!(render.reading.choice.reference.is_some(), "the rule picks");
    assert_eq!(render.stored_reference(), None);
    let fused = fuse_patch_bitmap(&patch, &views, &images, &keypoints, &params)
        .expect("a fused mean renders");
    let stored = render_patch_bitmap(
        &patch,
        &views,
        &images,
        &keypoints,
        &params,
        &Progress::none(),
    )
    .expect("a bitmap");
    assert_eq!(stored.reference, None);
    assert_eq!(stored.rgba, fused);
}

#[test]
fn the_sub_pixel_refiner_stores_the_fused_mean_for_a_last_fallback_pick() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (patch, images, keypoints) = point_inputs(&truth.recon, WITHOUT_ANY_POINT);
    let params = KeypointSubpixelParams {
        resolution: 24,
        render_bitmaps: true,
        ..Default::default()
    };
    let out = refine_patch_keypoints(
        &patch,
        &views,
        &images,
        Some(&anchors(&keypoints)),
        None,
        &params,
    );
    // The rule runs over the views the refiner kept, at their final keypoints.
    let render = render_reference(
        &patch,
        &views,
        &out.views,
        &anchors(&out.keypoints),
        24,
        params.sampler,
        &Progress::none(),
    );
    assert_eq!(
        render.reading.choice.fallback,
        ReferenceFallback::WithoutAny
    );
    assert_eq!(out.reference, None);
    assert!(out.representative.is_some(), "the fused mean");
}

#[test]
fn the_localizer_aligns_to_the_fused_mean_for_a_last_fallback_pick() {
    // With no reference given and a rule that reaches its pick only by its
    // last fallback, the template is the fused mean: no view is the
    // reference, so none scores the 1.0 of a view against its own render, and
    // every kept view is scored against the mean.
    let truth = GroundTruth::load();
    let views = truth.views();
    let (patch, images, keypoints) = point_inputs(&truth.recon, WITHOUT_ANY_POINT);
    let params = KeypointLocalizeParams {
        resolution: 24,
        ..open_localizer()
    };
    let out = localize_patch_keypoints(
        &patch,
        &views,
        &images,
        Some(&anchors(&keypoints)),
        None,
        &params,
    );
    assert_eq!(out.reference, None);
    assert!(out.views.len() >= 2, "{:?}", out.views);
    for (&v, &z) in out.views.iter().zip(&out.zncc) {
        assert!(z.is_finite() && z < 1.0, "view {v} scores {z}");
    }
}

#[test]
fn normal_refinement_stores_the_bitmap_its_reference_names() {
    let truth = GroundTruth::load();
    let all = truth.views();
    let (patch, images, keypoints) = point_inputs(&truth.recon, WITHOUT_ANY_POINT);
    let views: Vec<ProjectedImage<'_>> = images
        .iter()
        .map(|&i| {
            let v = &all[i as usize];
            ProjectedImage {
                camera: v.camera,
                cam_from_world: v.cam_from_world,
                pyramid: v.pyramid,
            }
        })
        .collect();
    let params = NormalRefineParams {
        render_bitmap: true,
        min_views: 2,
        ..Default::default()
    };
    let out = refine_patch_normal(&patch, &views, 24, &params, Some(&anchors(&keypoints)));
    let rgba = out.representative.as_ref().expect("a bitmap");
    // The stored bitmap agrees with its reference: the named view's tile at
    // the refined frame, and no view named where the rule's pick does not
    // stand.
    let all_views: Vec<u32> = (0..views.len() as u32).collect();
    let render = render_reference(
        &out.patch,
        &views,
        &all_views,
        &anchors(&keypoints),
        24,
        KeypointSubpixelParams::default().sampler,
        &Progress::none(),
    );
    assert_eq!(out.reference, render.stored_reference());
    if let Some(r) = out.reference {
        assert_eq!(rgba, &bitmap_from_tile(&render.tiles[r]));
    }
}

/// [`score_bitmap`] after [`render_bitmap_in_place`] writes the scores the
/// next [`evaluate`] reads against that bitmap, here for a fused mean, which
/// names no reference, so every row is scored.
#[test]
fn rendering_then_scoring_matches_an_evaluation_of_the_rendered_track() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let edited = EditedReconstruction::new(Arc::new(truth.recon.clone()));
    let (bench, report) = create_track(
        &Bench::new(),
        &edited,
        WITHOUT_ANY_POINT as u32,
        &CreateTrackOptions::default(),
    )
    .expect("the point is live");
    let track = bench.track(&report.label).expect("just put on");
    let options = EvaluateOptions::default();
    let (read, _) = evaluate(track, &edited, &views, &options, &Progress::none()).unwrap();
    let rendered = render_bitmap_in_place(&read, &edited, &views, &FitOptions::default());
    let payload = rendered.track().unwrap();
    assert!(payload.bitmap.is_some());
    assert_eq!(payload.reference, None, "the fused mean names no row");
    let scored = score_bitmap(&rendered, &edited, &views, &options, &Progress::none()).unwrap();
    let (again, _) = evaluate(&rendered, &edited, &views, &options, &Progress::none()).unwrap();
    for (i, (a, b)) in scored
        .observations
        .iter()
        .zip(&again.observations)
        .enumerate()
    {
        let (a, b) = (a.track.as_ref().unwrap(), b.track.as_ref().unwrap());
        assert!(a.zncc.is_some(), "row {i} is scored");
        assert_eq!(a.zncc, b.zncc, "row {i}");
        assert_eq!(a.blur_matched_zncc, b.blur_matched_zncc, "row {i}");
        assert_eq!(a.sharper_than_bitmap, b.sharper_than_bitmap, "row {i}");
    }
    // The combined call gives the same answer.
    let (combined, _) = evaluate_rendering_bitmap(
        track,
        &edited,
        &views,
        &options,
        &FitOptions::default(),
        &Progress::none(),
    )
    .unwrap();
    assert_same_bitmap_and_scores(&combined, &scored);
}

/// After a patch step, [`evaluate_rendering_bitmap`] reads the bitmap off the
/// evaluation's own tiles, and gives what [`evaluate`],
/// [`render_bitmap_in_place`] and [`score_bitmap`] give in turn.
#[test]
fn rendering_from_the_evaluation_matches_the_three_calls() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (read, edited) = truth.evaluated_track();
    let p = read.track().unwrap().placement.clone().unwrap();
    let aim = (p.normal() + p.u_axis.normalize() * 0.05).normalize();
    let (tilted, _) = tilt_patch(&read, &edited, aim).expect("a small turn");
    assert!(tilted.track().unwrap().bitmap.is_none());
    let options = EvaluateOptions::default();
    let fit = FitOptions::default();

    let (combined, _) =
        evaluate_rendering_bitmap(&tilted, &edited, &views, &options, &fit, &Progress::none())
            .unwrap();
    let (measured, _) = evaluate(&tilted, &edited, &views, &options, &Progress::none()).unwrap();
    let rendered = render_bitmap_in_place(&measured, &edited, &views, &fit);
    let scored = score_bitmap(&rendered, &edited, &views, &options, &Progress::none()).unwrap();
    assert!(scored.track().unwrap().reference.is_some(), "a picked row");
    assert_same_bitmap_and_scores(&combined, &scored);
}

fn assert_same_bitmap_and_scores(a: &EditableTrack, b: &EditableTrack) {
    let (pa, pb) = (a.track().unwrap(), b.track().unwrap());
    assert_eq!(pa.reference, pb.reference);
    assert_eq!(pa.bitmap, pb.bitmap);
    assert_eq!(pa.color, pb.color);
    for (i, (x, y)) in a.observations.iter().zip(&b.observations).enumerate() {
        let (x, y) = (x.track.as_ref().unwrap(), y.track.as_ref().unwrap());
        assert_eq!(x.zncc, y.zncc, "row {i}");
        assert_eq!(x.blur_matched_zncc, y.blur_matched_zncc, "row {i}");
        assert_eq!(x.bitmap_blur_sigma, y.bitmap_blur_sigma, "row {i}");
        assert_eq!(x.sharper_than_bitmap, y.sharper_than_bitmap, "row {i}");
    }
}

// ---- A defined reference is rendered from ----------------------------------

/// The row the last evaluation's reference-view rule picked.
fn rule_pick(track: &EditableTrack) -> Option<usize> {
    track.observations.iter().position(|o| {
        o.track
            .as_ref()
            .and_then(|m| m.reference_view)
            .is_some_and(|s| s.is_reference())
    })
}

/// The ground truth with the long track's stored reference set to an `in` row
/// the rule does not pick, that track put on the bench, the rule's pick and
/// the stored reference.
fn track_with_another_reference(
    truth: &GroundTruth,
) -> (EditableTrack, EditedReconstruction, usize, usize) {
    let (read, _) = truth.evaluated_track();
    let picked = rule_pick(&read).expect("the rule picks one");
    // The best-covered other row, so a small step leaves it `in`.
    let coverage = |i: usize| {
        read.observations[i]
            .track
            .as_ref()
            .and_then(|m| m.coverage)
            .unwrap_or(0.0)
    };
    let other = (0..read.observations.len())
        .filter(|&i| i != picked && read.observations[i].verdict == Verdict::In)
        .max_by(|&a, &b| coverage(a).total_cmp(&coverage(b)))
        .expect("another row is in");
    let point = read.origin.as_ref().expect("put on from a point").point;
    let mut recon = truth.recon.clone();
    let mut references = recon
        .point_set
        .reference_observations
        .clone()
        .expect("the ground truth has patch frames");
    references[point as usize] = other as i32;
    recon.point_set.reference_observations = Some(references);
    let edited = EditedReconstruction::new(Arc::new(recon));
    let (bench, report) = create_track(
        &Bench::new(),
        &edited,
        point,
        &CreateTrackOptions::default(),
    )
    .expect("the point is live");
    let track = (**bench.track(&report.label).expect("just put on")).clone();
    (track, edited, picked, other)
}

/// Check that `track`'s bitmap is the tile of `row` at its current keypoint,
/// that the bitmap names it, and that the row scores 1 against it.
fn assert_rendered_from(
    track: &EditableTrack,
    views: &[ProjectedImage<'_>],
    edited: &EditedReconstruction,
    row: usize,
    step: &str,
) {
    let payload = track.track().expect("a track stage");
    assert_eq!(payload.reference, Some(row), "{step}");
    let resolution = EvaluateOptions::default().patch_resolution(&edited.base) as usize;
    let tile = bitmap_from_tile(&tile_of_row(track, views, row, resolution));
    let stored: Vec<u8> = payload
        .bitmap
        .as_ref()
        .unwrap_or_else(|| panic!("{step}: a bitmap"))
        .iter()
        .copied()
        .collect();
    assert_eq!(stored, tile, "{step}");
    let m = track.observations[row].track.as_ref().unwrap();
    assert_eq!(m.zncc, Some(1.0), "{step}");
    assert_eq!(m.zncc_middle, Some(1.0), "{step}");
    assert_eq!(m.zncc_grid, Some([[1.0; 3]; 3]), "{step}");
    assert_eq!(m.blur_matched_zncc, Some(1.0), "{step}");
    // Every other row that has a tile is scored against the bitmap.
    for (i, o) in track.observations.iter().enumerate() {
        if i != row && o.verdict == Verdict::In {
            let m = o.track.as_ref().unwrap();
            assert!(
                m.zncc.is_some_and(|z| z < 1.0),
                "{step}: row {i} {:?}",
                m.zncc
            );
        }
    }
}

/// [`evaluate_rendering_bitmap`] with the default options.
fn render_with(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
) -> EditableTrack {
    evaluate_rendering_bitmap(
        track,
        edited,
        views,
        &EvaluateOptions::default(),
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads")
    .0
}

/// A point's track comes on with every row pinned, so its stored reference
/// is held against the rule's pick, render after render, and pinning another
/// row does not make that row the reference.
#[test]
fn a_pinned_reference_row_keeps_the_reference_against_the_rule_s_pick() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    assert!(track.observations.iter().all(|o| o.pinned));
    let first = render_with(&track, &edited, &views);
    assert_rendered_from(&first, &views, &edited, other, "opened");
    assert_eq!(rule_pick(&first), Some(picked));
    let again = render_with(&first, &edited, &views);
    assert_rendered_from(&again, &views, &edited, other, "rendered again");

    // Unpinning the rule's pick and pinning it again leaves the reference.
    let (loose, _) = unpin_verdicts(&again, &[picked]).expect("a live row");
    let (pinned, report) = pin_verdicts(&loose, &[picked]).expect("a live row");
    assert!(report.changed);
    let pinned = render_with(&pinned, &edited, &views);
    assert_rendered_from(&pinned, &views, &edited, other, "pinned another row");
}

/// Unpinning the reference row hands the reference to the rule: the next
/// render renders the bitmap from the rule's pick and scores every row
/// against it.
#[test]
fn unpinning_the_reference_row_moves_the_reference_to_the_rule_s_pick() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    assert_rendered_from(&first, &views, &edited, other, "opened");

    let (unpinned, _) = unpin_verdicts(&first, &[other]).expect("a live row");
    assert_eq!(unpinned.track().unwrap().reference, Some(other));
    assert_eq!(unpinned.held_reference(), None);
    let moved = render_with(&unpinned, &edited, &views);
    let pick = rule_pick(&moved).expect("the rule picks one");
    assert_eq!(pick, picked);
    assert_rendered_from(&moved, &views, &edited, picked, "unpinned");
    let m = moved.observations[other].track.as_ref().unwrap();
    assert!(
        m.zncc.is_some_and(|z| z < 1.0),
        "the old reference is scored against the new bitmap: {:?}",
        m.zncc
    );
}

/// While the reference row is unpinned the reference follows the rule's
/// pick at every render, and stays on it while the pick stands.
#[test]
fn the_reference_follows_the_rule_s_pick_while_its_row_is_unpinned() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, other) = track_with_another_reference(&truth);
    let every: Vec<usize> = (0..track.observations.len()).collect();
    let (loose, _) = unpin_verdicts(&track, &every).expect("live rows");
    assert_eq!(loose.track().unwrap().reference, Some(other));
    let first = render_with(&loose, &edited, &views);
    let pick = rule_pick(&first).expect("the rule picks one");
    assert_ne!(pick, other);
    assert_rendered_from(&first, &views, &edited, pick, "first render");
    let again = render_with(&first, &edited, &views);
    assert_eq!(rule_pick(&again), Some(pick));
    assert_rendered_from(&again, &views, &edited, pick, "second render");
}

/// *Set as reference* makes a row the reference and pins it; the next render
/// renders the bitmap from it and scores every row against it, and the rule's
/// pick stays reported beside it. It is refused on an `out` row and at the
/// cluster stage.
#[test]
fn set_as_reference_pins_the_row_and_renders_from_it() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let third = (0..first.observations.len())
        .find(|&i| i != other && i != picked && first.observations[i].verdict == Verdict::In)
        .expect("a third row");
    let (loose, _) = unpin_verdicts(&first, &[third]).expect("a live row");
    let loose = render_with(&loose, &edited, &views);
    assert_eq!(loose.track().unwrap().reference, Some(other));

    let (set, report) = set_reference(&loose, third).expect("an in row with a keypoint");
    assert_eq!((report.observation, report.was), (third, Some(other)));
    assert!(report.changed);
    assert!(set.observations[third].pinned);
    assert_eq!(set.held_reference(), Some(third));
    assert!(
        set.track().unwrap().bitmap.is_none(),
        "the old bitmap is stale"
    );
    assert!(
        set.observations
            .iter()
            .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none())),
        "so is every score read against it"
    );
    // Unpinning another row while the held reference waits for its render
    // leaves the render where it was: from the held row, not the pick.
    let fourth = (0..set.observations.len())
        .find(|&i| i != third && set.observations[i].pinned)
        .expect("another pinned row");
    let (waiting, report) = unpin_verdicts(&set, &[fourth]).expect("a live row");
    assert!(!report.bitmap_pending, "the held reference still names it");
    assert_eq!(waiting.held_reference(), Some(third));
    let rendered = render_with(&set, &edited, &views);
    assert_rendered_from(&rendered, &views, &edited, third, "set as reference");
    assert_eq!(rule_pick(&rendered), Some(picked));

    // Setting it again changes nothing.
    let (_, again) = set_reference(&rendered, third).expect("still in");
    assert!(!again.changed);

    // Refusals.
    let (out, _) = set_verdict(&rendered, other, Verdict::Out).expect("a verdict");
    assert_eq!(
        set_reference(&out, other).unwrap_err(),
        TrackEditError::NotIn { observation: other }
    );
    let (cluster, _) = set_stage(
        &rendered,
        &edited,
        &views,
        StageKind::Cluster,
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("a track goes down to a cluster");
    assert!(matches!(
        set_reference(&cluster, 0),
        Err(TrackEditError::WrongStage { .. })
    ));
}

/// A track opened from a file whose reference is not the row the rule picks
/// renders from the file's reference, and keeps rendering from it after a
/// patch step, a sighting of another row, a sighting of the reference row
/// itself and a fit; the evaluation still reports the rule's pick, and a
/// commit saves the reference the bench holds. The render that reuses the
/// evaluation's tiles gives what the three separate calls give.
#[test]
fn a_defined_reference_is_rendered_from_through_the_bench_s_steps() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    assert_eq!(track.track().unwrap().reference, Some(other));
    assert!(track.track().unwrap().bitmap.is_none());
    let options = EvaluateOptions::default();
    let fit_options = FitOptions::default();
    let render = |t: &EditableTrack| {
        evaluate_rendering_bitmap(
            t,
            &edited,
            &views,
            &options,
            &fit_options,
            &Progress::none(),
        )
        .expect("the track reads")
        .0
    };

    // Opened from the file: the first render is from the file's reference,
    // while the rule's pick is reported as information.
    let first = render(&track);
    assert_rendered_from(&first, &views, &edited, other, "opened");
    assert_eq!(rule_pick(&first), Some(picked));

    // A patch step drops the bitmap; the render keeps the reference, and the
    // combined call matches the three separate calls.
    let p = first.track().unwrap().placement.clone().unwrap();
    let aim = (p.normal() + p.u_axis.normalize() * 0.05).normalize();
    let (tilted, _) = tilt_patch(&first, &edited, aim).expect("a small turn");
    assert!(tilted.track().unwrap().bitmap.is_none());
    let after_tilt = render(&tilted);
    assert_rendered_from(&after_tilt, &views, &edited, other, "tilted");
    let (measured, _) = evaluate(&tilted, &edited, &views, &options, &Progress::none()).unwrap();
    let rendered = render_bitmap_in_place(&measured, &edited, &views, &fit_options);
    let scored = score_bitmap(&rendered, &edited, &views, &options, &Progress::none()).unwrap();
    assert_same_bitmap_and_scores(&after_tilt, &scored);

    // Sighting another row keeps the bitmap and the reference.
    let third = (0..after_tilt.observations.len())
        .find(|&i| i != other && after_tilt.observations[i].verdict == Verdict::In)
        .expect("a third row");
    let [x, y] = keypoint_of(&after_tilt, third);
    let (sighted, _) =
        sight_observation(&after_tilt, &edited, third, [x + 0.5, y + 0.5]).expect("sighted");
    let sighted = render(&sighted);
    assert_eq!(sighted.track().unwrap().reference, Some(other));
    assert!(sighted.track().unwrap().bitmap.is_some());

    // Sighting the reference row itself drops the bitmap; the next render is
    // from the same row at its new keypoint.
    let [x, y] = keypoint_of(&sighted, other);
    let (moved, _) =
        sight_observation(&sighted, &edited, other, [x + 0.5, y - 0.5]).expect("sighted");
    assert!(moved.track().unwrap().bitmap.is_none());
    assert!(unscored(&moved), "no score outlives the bitmap");
    let moved = render(&moved);
    assert_rendered_from(&moved, &views, &edited, other, "reference sighted");

    // A fit renders from it too.
    let (fitted, _) = fit(&moved, &edited, &views, &fit_options, &Progress::none()).expect("fits");
    assert_rendered_from(&fitted, &views, &edited, other, "fitted");

    // A commit saves the reference the bench holds.
    let (committed, report) = commit(&edited, &fitted).expect("the track commits");
    let view = committed.point(report.point).expect("the committed point");
    let images: Vec<u32> = view.observations().iter().map(|o| o.image_index).collect();
    let at = view.reference_observation().expect("the column is carried") as usize;
    assert_eq!(images[at], fitted.observations[other].image);
}

/// Turning the reference row `out`, or deleting its image, leaves the track
/// with no reference, and the next render sets one by the rule.
#[test]
fn the_rule_sets_the_reference_again_after_its_row_goes() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, other) = track_with_another_reference(&truth);
    let options = EvaluateOptions::default();
    let fit_options = FitOptions::default();
    let render = |t: &EditableTrack| {
        evaluate_rendering_bitmap(
            t,
            &edited,
            &views,
            &options,
            &fit_options,
            &Progress::none(),
        )
        .expect("the track reads")
        .0
    };
    let first = render(&track);
    assert_eq!(first.track().unwrap().reference, Some(other));

    let (out, _) = set_verdict(&first, other, Verdict::Out).expect("a verdict");
    assert_eq!(out.track().unwrap().reference, None);
    assert!(unscored(&out), "no score outlives the bitmap");
    let again = render(&out);
    let pick = rule_pick(&again).expect("the rule picks one");
    assert_ne!(pick, other);
    assert_rendered_from(&again, &views, &edited, pick, "turned out");

    let image = first.observations[other].image;
    let (deleted, _) = first.delete_image(image).expect("the track sees it");
    assert_eq!(deleted.track().unwrap().reference, None);
    assert!(unscored(&deleted), "no score outlives the bitmap");
    let again = render(&deleted);
    let pick = rule_pick(&again).expect("the rule picks one");
    assert_rendered_from(&again, &views, &edited, pick, "deleted");
}

// ---- The bars judge scores against the bitmap the track ends with ----------

/// Whether no row of `track` carries a score against the bitmap.
fn unscored(track: &EditableTrack) -> bool {
    track
        .observations
        .iter()
        .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none()))
}

/// Whether the bars would move none of `track`'s verdicts: every unpinned
/// verdict is what the bars say about the scores the track carries.
fn verdicts_agree_with_the_bars(track: &EditableTrack) -> bool {
    !apply_thresholds(track).1.changed
}

/// Unpinning the row that holds the reference, where the rule picks another
/// row, judges nothing against the outgoing bitmap: every score is cleared, the
/// bitmap is kept until the render replaces it, and the report says the
/// verdicts wait for the render.
/// The evaluation that follows renders from the pick and judges every row
/// against it. The held row's would-be verdict is unknown until then.
#[test]
fn unpinning_the_held_reference_waits_for_the_new_bitmap() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    assert_rendered_from(&first, &views, &edited, other, "opened");
    assert_eq!(
        verdicts_if_unpinned(&first)[other],
        None,
        "the held row's score against its own render says nothing"
    );

    let (unpinned, report) = unpin_verdicts(&first, &[other]).expect("a live row");
    assert!(report.bitmap_pending);
    assert_eq!((report.turned_in, report.turned_out), (0, 0));
    let payload = unpinned.track().unwrap();
    assert!(
        payload.bitmap.is_some(),
        "the outgoing bitmap stays until the render replaces it"
    );
    assert_eq!(payload.reference, Some(other));
    assert!(unpinned
        .observations
        .iter()
        .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none())));
    for (a, b) in first.observations.iter().zip(&unpinned.observations) {
        assert_eq!(
            a.verdict, b.verdict,
            "no verdict is judged on a stale score"
        );
    }
    assert!(unpinned.bitmap_pending());
    for o in &unpinned.observations {
        let m = o.track.as_ref().unwrap();
        if m.seed_shift_px.is_some() {
            assert_eq!(m.reason, Some(Unmeasured::BitmapPending));
        }
    }

    // A commit before the render writes the outgoing bitmap with the row it
    // is the render of.
    let mut recon = (*edited.base).clone();
    let r = payload.bitmap.as_ref().unwrap().shape()[0];
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(ndarray::Array4::zeros((
        recon.point_count(),
        r,
        r,
        4,
    ))));
    let storing = EditedReconstruction::new(Arc::new(recon));
    let (committed, report) = commit(&storing, &unpinned).expect("the track commits");
    let view = committed.point(report.point).expect("the committed point");
    let images: Vec<u32> = view.observations().iter().map(|o| o.image_index).collect();
    let at = view.reference_observation().expect("the column is carried") as usize;
    assert_eq!(images[at], first.observations[other].image);
    assert_eq!(
        view.patch_bitmap().expect("the column is carried"),
        first.track().unwrap().bitmap.as_ref().unwrap().view(),
        "the outgoing bitmap is written"
    );

    // A reading that renders nothing scores no row against the bitmap the
    // render is to replace, and the bars judge none.
    let (plain, report) = evaluate(
        &unpinned,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");
    assert_eq!(report.scored, 0, "{report}");
    assert!(unscored(&plain));
    assert!(plain.bitmap_pending());
    assert_eq!(plain.track().unwrap().bitmap, payload.bitmap);
    for (a, b) in first.observations.iter().zip(&plain.observations) {
        assert_eq!(a.verdict, b.verdict);
    }
    let pending = plain.observations.iter().any(|o| {
        o.track
            .as_ref()
            .is_some_and(|m| m.reason == Some(Unmeasured::BitmapPending))
    });
    assert!(pending, "the rows say the bitmap is to be rendered again");

    // Pinning the row again holds the reference: the bitmap is no longer
    // pending, and a plain reading scores against it.
    let (held, _) = pin_verdicts(&unpinned, &[other]).expect("a live row");
    assert!(!held.bitmap_pending());
    assert!(!held.track().unwrap().bitmap_pending, "the mark is cleared");
    let (_, report) = evaluate(
        &held,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");
    assert!(report.scored > 0, "{report}");

    let moved = render_with(&unpinned, &edited, &views);
    assert!(!moved.bitmap_pending());
    assert_rendered_from(&moved, &views, &edited, picked, "after the unpin");
    assert!(verdicts_agree_with_the_bars(&moved));

    // Unpinning a row that is not the reference's leaves the bitmap.
    let third = (0..first.observations.len())
        .find(|&i| i != other && i != picked)
        .expect("a third row");
    let (_, report) = unpin_verdicts(&first, &[third]).expect("a live row");
    assert!(!report.bitmap_pending);
}

/// The ground truth with point `p` stored at `-1`: its reference observation
/// names no row and its stored bitmap is the fused mean of its views, as a
/// file stored before the reference was recorded holds it. The point's track
/// put on the bench, and the reconstruction.
fn track_at_minus_one(truth: &GroundTruth, p: usize) -> (EditableTrack, EditedReconstruction) {
    let views = truth.views();
    let mut recon = truth.recon.clone();
    let mut references = recon
        .point_set
        .reference_observations
        .clone()
        .expect("the ground truth has patch frames");
    references[p] = -1;
    recon.point_set.reference_observations = Some(references);
    // The ground truth stores no bitmaps: a column of blank ones, with the
    // fused mean in point `p`'s row.
    let r = EvaluateOptions::default().patch_resolution(&truth.recon) as usize;
    let mut bitmaps = ndarray::Array4::<u8>::zeros((recon.point_count(), r, r, 4));
    let (patch, images, keypoints) = point_inputs(&truth.recon, p);
    let params = KeypointSubpixelParams {
        resolution: r as u32,
        ..Default::default()
    };
    let fused = fuse_patch_bitmap(&patch, &views, &images, &keypoints, &params)
        .expect("a fused mean renders");
    for (stored, value) in bitmaps
        .index_axis_mut(ndarray::Axis(0), p)
        .iter_mut()
        .zip(fused)
    {
        *stored = value;
    }
    recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(bitmaps));
    let edited = EditedReconstruction::new(Arc::new(recon));
    let (bench, report) = create_track(
        &Bench::new(),
        &edited,
        p as u32,
        &CreateTrackOptions::default(),
    )
    .expect("the point is live");
    let track = (**bench.track(&report.label).expect("just put on")).clone();
    (track, edited)
}

/// The long track's point, the one [`GroundTruth::evaluated_track`] reads.
fn long_track_point(truth: &GroundTruth) -> usize {
    let offsets = &truth.recon.point_set.observation_offsets;
    (0..truth.recon.point_count())
        .find(|&p| offsets[p + 1] - offsets[p] >= MIN_TRACK)
        .expect("the ground truth has a long track")
}

/// A track at `-1` comes on with its stored fused mean and no reference.
/// Pins play no part: a reading that renders nothing scores the rows against
/// the mean, and unpinning and pinning rows hands nothing on, while the
/// first evaluation that renders renders the bitmap from the rule's pick and
/// names it, and a commit saves the pick as the point's reference.
#[test]
fn a_track_at_minus_one_is_rendered_from_the_rule_s_pick_at_its_first_evaluation() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited) = track_at_minus_one(&truth, long_track_point(&truth));
    assert!(track.observations.iter().all(|o| o.pinned));
    let stored = track.track().unwrap().bitmap.clone();
    assert!(stored.is_some());
    assert_eq!(track.track().unwrap().reference, None);

    // A plain reading keeps the mean and scores against it; the pins and the
    // proposals are those of any track that holds no reference.
    let (plain, _) = evaluate(
        &track,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");
    assert_eq!(plain.track().unwrap().bitmap, stored);
    assert!(!unscored(&plain));
    let picked = rule_pick(&plain).expect("the rule picks one");
    assert_eq!(bitmap_target(&plain), Some(Some(picked)));
    assert!(verdicts_if_unpinned(&plain).iter().any(Option::is_some));
    let row = (0..plain.observations.len())
        .find(|&i| i != picked)
        .expect("another row");
    let (unpinned, report) = unpin_verdicts(&plain, &[row]).expect("a live row");
    assert!(!report.bitmap_pending);
    assert!(!unpinned.bitmap_pending());
    assert!(!unscored(&unpinned), "an unpin clears no score");
    assert_eq!(bitmap_target(&unpinned), Some(Some(picked)));

    // The first evaluation that renders moves the bitmap to the pick.
    let first = render_with(&track, &edited, &views);
    assert_eq!(rule_pick(&first), Some(picked));
    assert_rendered_from(&first, &views, &edited, picked, "the first evaluation");
    assert_eq!(bitmap_target(&first), None);

    // A commit saves the pick as the point's reference.
    let (committed, report) = commit(&edited, &first).expect("the track commits");
    let view = committed.point(report.point).expect("the committed point");
    let images: Vec<u32> = view.observations().iter().map(|o| o.image_index).collect();
    let at = view.reference_observation().expect("the column is carried");
    assert!(at >= 0);
    assert_eq!(images[at as usize], first.observations[picked].image);
}

/// Where the rule reaches its pick only through its last fallback, the render
/// gives the fused mean the track at `-1` already stores, so the mean and
/// `-1` stay; once a reading reaches a pick another way, the bitmap moves to
/// it, whatever the pins.
#[test]
fn a_track_at_minus_one_keeps_its_fused_mean_until_the_rule_picks_without_its_last_fallback() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited) = track_at_minus_one(&truth, WITHOUT_ANY_POINT);
    let first = render_with(&track, &edited, &views);
    let stored = track.track().unwrap().bitmap.clone();
    assert_eq!(first.track().unwrap().reference, None);
    assert_eq!(first.track().unwrap().bitmap, stored);
    let pick = rule_pick(&first).expect("the rule picks one");
    let standing = |t: &EditableTrack| {
        t.observations[pick]
            .track
            .as_ref()
            .and_then(|m| m.reference_view)
            .expect("a reading")
    };
    assert_eq!(standing(&first).fallback, ReferenceFallback::WithoutAny);
    assert_eq!(bitmap_target(&first), None);

    // A reading that reaches the same pick without the last fallback, as a
    // later step's reading can.
    let mut reread = first.clone();
    reread.observations[pick]
        .track
        .as_mut()
        .and_then(|m| m.reference_view.as_mut())
        .expect("a reading")
        .fallback = ReferenceFallback::None;
    assert!(reread.observations.iter().all(|o| o.pinned));
    assert_eq!(bitmap_target(&reread), Some(Some(pick)));

    // A pick with no keypoint is a row no render takes the tile of, so the
    // mean stands rather than being rendered again at every evaluation.
    let mut unkeyed = reread.clone();
    unkeyed.observations[pick].track.as_mut().unwrap().keypoint = None;
    assert_eq!(bitmap_target(&unkeyed), None);
}

/// A track built on the bench whose bitmap names no row follows the same
/// rule as a point at `-1`: the first evaluation renders from the pick.
#[test]
fn a_bench_built_track_with_a_fused_mean_is_rendered_from_the_pick() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (mut track, edited) = track_at_minus_one(&truth, long_track_point(&truth));
    track.origin = None;
    for o in &mut track.observations {
        o.provenance = Provenance::Pixel;
    }
    let first = render_with(&track, &edited, &views);
    let pick = rule_pick(&first).expect("the rule picks one");
    assert_rendered_from(&first, &views, &edited, pick, "a bench-built track");
}

/// Unpinning the held reference, pinning it again and unpinning it once more
/// is decided from the pins as they stand: where the rule by then picks that
/// same row, the second unpin waits for no render and keeps every score, as
/// for a track that was never unpinned.
#[test]
fn a_re_pinned_reference_is_unpinned_again_as_if_never_unpinned() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);

    // Turn the rule's picks `out`, by hand, until it picks `other`.
    let until_the_rule_picks_other = |mut t: EditableTrack| {
        for _ in 0..t.observations.len() {
            let pick = rule_pick(&t).expect("the rule picks one");
            if pick == other {
                return t;
            }
            let (out, _) = set_verdict(&t, pick, Verdict::Out).expect("a live row");
            t = render_with(&out, &edited, &views);
        }
        panic!("the rule never picks row {other}");
    };
    let scored = |t: &EditableTrack| {
        t.observations
            .iter()
            .filter(|o| o.track.as_ref().is_some_and(|m| m.zncc.is_some()))
            .count()
    };

    let (unpinned, report) = unpin_verdicts(&first, &[other]).expect("a live row");
    assert!(report.bitmap_pending);
    let (held, _) = pin_verdicts(&unpinned, &[other]).expect("a live row");
    assert!(!held.track().unwrap().bitmap_pending, "the mark is cleared");
    let cycled = until_the_rule_picks_other(render_with(&held, &edited, &views));
    assert_eq!(cycled.track().unwrap().reference, Some(other));
    assert_eq!(verdicts_if_unpinned(&cycled)[other], Some(Verdict::In));
    let (again, report) = unpin_verdicts(&cycled, &[other]).expect("a live row");
    assert!(!report.bitmap_pending, "the rule picks the held row");
    assert!(!again.bitmap_pending());
    assert_eq!(scored(&again), scored(&cycled), "every score is kept");
    assert!(scored(&again) > 0);

    // The same as for the track never unpinned.
    let control = until_the_rule_picks_other(first);
    let (_, report) = unpin_verdicts(&control, &[other]).expect("a live row");
    assert!(!report.bitmap_pending);
}

/// A repaint inside one evaluation that turns the reference row out moves the
/// bitmap to the rule's new pick in the same call, and the rows are scored
/// and judged against that bitmap; the evaluation that reads the marked track
/// again settles without moving anything.
#[test]
fn a_repaint_that_moves_the_pick_rerenders_and_rejudges_in_the_same_evaluation() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, _) = track_with_another_reference(&truth);
    let every: Vec<usize> = (0..track.observations.len()).collect();
    let (loose, _) = unpin_verdicts(&track, &every).expect("live rows");
    let first = render_with(&loose, &edited, &views);
    let pick = rule_pick(&first).expect("the rule picks one");
    assert_eq!(first.track().unwrap().reference, Some(pick));

    // A projection bar just under the pick's own reprojection error turns the
    // pick out, and the rows whose error is larger with it. The bitmap is the
    // pick's render when the repaint judges that, so the repaint drops it with
    // its reference, and the rule picks again among the rows left.
    let error = |t: &EditableTrack, i: usize| {
        t.observations[i]
            .track
            .as_ref()
            .and_then(|m| m.reprojection_error)
            .expect("a triangulated track")
    };
    let bar = error(&first, pick) - 1e-3;
    let left = (0..first.observations.len())
        .filter(|&i| error(&first, i) <= bar)
        .count();
    assert!(left >= 2, "two rows stay to render from");
    let mut tight = first.clone();
    tight.thresholds.max_projection_error_px = bar;
    let (read, report) = evaluate_rendering_bitmap(
        &tight,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");
    assert_eq!(read.observations[pick].verdict, Verdict::Out);
    assert!(report.turned_out >= 1);
    let payload = read.track().unwrap();
    assert!(payload.bitmap.is_some(), "rendered again within the call");
    let reference = payload.reference;
    assert_ne!(reference, Some(pick), "the bitmap left the row turned out");
    match reference {
        // The rule's new pick among the rows left.
        Some(row) => {
            assert_eq!(rule_pick(&read), Some(row));
            assert_rendered_from(&read, &views, &edited, row, "moved within the call");
        }
        // The rule reached no pick it stores among the rows left, so the
        // bitmap is the fused mean.
        None => {
            let stored = read.observations.iter().position(|o| {
                o.track
                    .as_ref()
                    .and_then(|m| m.reference_view)
                    .is_some_and(|s| {
                        s.is_reference() && s.fallback != ReferenceFallback::WithoutAny
                    })
            });
            assert_eq!(stored, None);
        }
    }
    let scored = read.observations[pick].track.as_ref().unwrap().zncc;
    assert!(
        scored.is_some_and(|z| z < 1.0),
        "the row turned out is scored against the new bitmap: {scored:?}"
    );
    assert!(verdicts_agree_with_the_bars(&read));
    assert!(read.repainted());

    // The marked track is read once more and settles where it stands.
    let settled = render_with(&read, &edited, &views);
    assert!(!settled.repainted());
    assert_eq!(settled.track().unwrap().reference, reference);
    for (a, b) in read.observations.iter().zip(&settled.observations) {
        assert_eq!(a.verdict, b.verdict);
    }
}

/// A bar no row clears turns every row `out`, and the bitmap goes with its
/// reference row. The next render cannot render from an `in` row, so it
/// renders a bitmap for judging from the rows with a keypoint: the rows are
/// scored against it, a commit is refused, and loosening the bar turns rows
/// back `in`, after which the render by the usual rule replaces it.
#[test]
fn a_track_whose_rows_all_go_out_is_judged_against_a_bitmap_for_judging() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, _) = track_with_another_reference(&truth);
    let every: Vec<usize> = (0..track.observations.len()).collect();
    let (loose, _) = unpin_verdicts(&track, &every).expect("live rows");
    let first = render_with(&loose, &edited, &views);
    let defaults = first.thresholds.clone();

    // With every row pinned and the reference the rule's pick, unpinning them
    // all under a bar no row clears turns them all out, and the bitmap goes
    // with the reference row: the report says a render is pending.
    let (mut pinned_tight, _) = pin_verdicts(&first, &every).expect("live rows");
    pinned_tight.thresholds.max_projection_error_px = 1e-12;
    let (unpinned, report) = unpin_verdicts(&pinned_tight, &every).expect("live rows");
    assert!(unpinned
        .observations
        .iter()
        .all(|o| o.verdict == Verdict::Out));
    assert!(unpinned.track().unwrap().bitmap.is_none());
    assert!(report.bitmap_pending, "{report:?}");
    let judged = render_with(&unpinned, &edited, &views);
    assert!(judged.track().unwrap().bitmap_for_judging);

    let mut tight = first.clone();
    tight.thresholds.max_projection_error_px = 1e-12;
    let (out, report) = apply_thresholds(&tight);
    assert!(report.turned_out >= 2);
    assert!(out.observations.iter().all(|o| o.verdict == Verdict::Out));
    assert!(
        out.track().unwrap().bitmap.is_none(),
        "it went with its row"
    );
    assert!(
        out.observations
            .iter()
            .all(|o| o.track.as_ref().is_none_or(|m| m.zncc.is_none())),
        "no score outlives the bitmap it was read against"
    );

    let judged = render_with(&out, &edited, &views);
    let payload = judged.track().unwrap();
    assert!(payload.bitmap.is_some() && payload.bitmap_for_judging);
    assert_eq!(payload.reference, None, "a bitmap for judging names no row");
    assert!(payload.committable_bitmap().is_none());
    assert!(judged
        .observations
        .iter()
        .all(|o| o.verdict == Verdict::Out));
    assert!(
        judged
            .observations
            .iter()
            .filter(|o| o.track.as_ref().is_some_and(|m| m.keypoint.is_some()))
            .all(|o| o.track.as_ref().unwrap().zncc.is_some()),
        "every row with a keypoint is scored"
    );
    assert!(commit(&edited, &judged).is_err(), "no row is in");
    // Read again with no row `in`, the bitmap for judging stands.
    let again = render_with(&judged, &edited, &views);
    assert!(again.track().unwrap().bitmap_for_judging);
    assert_eq!(again.track().unwrap().bitmap, payload.bitmap);

    let mut loosened = judged.clone();
    loosened.thresholds = defaults;
    let (back, report) = apply_thresholds(&loosened);
    assert!(report.turned_in >= 2, "{report:?}");
    assert!(
        back.track().unwrap().bitmap_for_judging,
        "a verdict leaves the bitmap for judging until the next render"
    );
    if edited.has_patch_bitmaps() {
        assert!(commit(&edited, &back).is_err(), "it is not committed");
    }

    let mut settled = render_with(&back, &edited, &views);
    for _ in 0..settled.observations.len() {
        if !settled.repainted() {
            break;
        }
        settled = render_with(&settled, &edited, &views);
    }
    let payload = settled.track().unwrap();
    assert!(!payload.bitmap_for_judging, "rendered by the usual rule");
    assert!(payload.bitmap.is_some());
    assert!(settled.in_observations().len() >= 2);
    if let Some(row) = payload.reference {
        assert_eq!(settled.observations[row].verdict, Verdict::In);
    }
    assert!(verdicts_agree_with_the_bars(&settled));
}

/// A track with one `in` row gets a bitmap for judging whichever way the
/// render runs: the `in` rows give no bitmap of their own, so two evaluations
/// in a row give the same bitmap and the same verdicts. Turning a second row
/// `in` by hand gets the render by the usual rule; a sighting or a deleted
/// image drops the bitmap for judging, which can be that row's tile.
#[test]
fn a_track_with_one_in_row_is_judged_against_the_same_bitmap_for_judging() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, _) = track_with_another_reference(&truth);
    let every: Vec<usize> = (0..track.observations.len()).collect();
    let (loose, _) = unpin_verdicts(&track, &every).expect("live rows");
    let first = render_with(&loose, &edited, &views);
    let ins = first.in_observations();
    assert!(ins.len() >= 3);
    let (kept, second) = (ins[0], ins[1]);
    // Every other row out by hand, and the kept one pinned `in`.
    let mut one = first.clone();
    for i in 0..one.observations.len() {
        let verdict = if i == kept { Verdict::In } else { Verdict::Out };
        one = set_verdict(&one, i, verdict).expect("a verdict").0;
    }
    assert_eq!(one.in_observations(), vec![kept]);

    let a = render_with(&one, &edited, &views);
    let payload = a.track().unwrap();
    assert!(payload.bitmap_for_judging, "one `in` row holds no bitmap");
    assert_eq!(payload.reference, None);
    // Read again, with the reading's rule now naming the kept row: the same
    // bitmap for judging, the same scores and the same verdicts.
    let b = render_with(&a, &edited, &views);
    assert!(b.track().unwrap().bitmap_for_judging);
    assert_eq!(b.track().unwrap().bitmap, payload.bitmap);
    for (x, y) in a.observations.iter().zip(&b.observations) {
        assert_eq!(x.verdict, y.verdict);
        let (mx, my) = (x.track.as_ref().unwrap(), y.track.as_ref().unwrap());
        assert_eq!(mx.zncc, my.zncc);
    }
    // A plain reading scores the rows against it.
    let (plain, report) = evaluate(
        &a,
        &edited,
        &views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .expect("the track reads");
    assert!(report.scored > 0, "{report}");
    assert_eq!(plain.track().unwrap().bitmap, payload.bitmap);

    // A second row turned `in` by hand: the usual render replaces it.
    let (two, _) = set_verdict(&a, second, Verdict::In).expect("a verdict");
    assert!(two.track().unwrap().bitmap_for_judging, "until the render");
    let rendered = render_with(&two, &edited, &views);
    let payload = rendered.track().unwrap();
    assert!(!payload.bitmap_for_judging);
    let reference = payload.reference.expect("the render names its row");
    assert!([kept, second].contains(&reference));

    // Set as reference on such a track renders from the row named.
    let (set, _) = set_reference(&two, second).expect("an in row with a keypoint");
    assert!(set.track().unwrap().bitmap.is_none());
    let rendered = render_with(&set, &edited, &views);
    assert_rendered_from(&rendered, &views, &edited, second, "set as reference");

    // A sighting drops a bitmap for judging with every score against it, and
    // so does deleting an image the track sees.
    let [x, y] = keypoint_of(&a, kept);
    let (sighted, _) = sight_observation(&a, &edited, kept, [x + 0.5, y]).expect("sighted");
    assert!(sighted.track().unwrap().bitmap.is_none());
    assert!(unscored(&sighted));
    let out_row = (0..a.observations.len())
        .find(|&i| i != kept)
        .expect("another row");
    let image = a.observations[out_row].image;
    let (deleted, _) = a.delete_image(image).expect("the track sees it");
    assert!(deleted.track().unwrap().bitmap.is_none());
    assert!(unscored(&deleted));
}

/// *Set as reference* refuses a row with no keypoint and an index past the
/// end.
#[test]
fn set_reference_refuses_a_row_with_no_keypoint_and_an_index_past_the_end() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, _) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let mut unplaced = first.clone();
    unplaced.observations[picked]
        .track
        .as_mut()
        .unwrap()
        .keypoint = None;
    assert_eq!(
        set_reference(&unplaced, picked).unwrap_err(),
        TrackEditError::NoPlace {
            observation: picked
        }
    );
    let count = first.observations.len();
    assert!(matches!(
        set_reference(&first, count),
        Err(TrackEditError::NoSuchObservation { observation, .. }) if observation == count
    ));
}

/// A fit renders the bitmap from the reference the track holds, not from the
/// rule's pick.
#[test]
fn a_fit_renders_from_the_held_reference_rather_than_the_pick() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let (fitted, _) = fit(
        &first,
        &edited,
        &views,
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("the track fits");
    assert_eq!(fitted.held_reference(), Some(other));
    assert_eq!(fitted.track().unwrap().reference, Some(other));
    let read = render_with(&fitted, &edited, &views);
    // The fit aligned the other rows to the held reference, which moves them
    // by tenths of a pixel and can move the rule's pick among them; the pick
    // is still another row than the one the bitmap is rendered from.
    let pick = rule_pick(&read).expect("the rule picks one");
    assert_ne!(pick, other, "the rule still picks {picked} or another row");
    assert_rendered_from(&read, &views, &edited, other, "after a fit");
}

/// The rows' keypoints after a fit of `track` with the default options.
fn fitted(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
) -> EditableTrack {
    fit(
        track,
        edited,
        views,
        &FitOptions::default(),
        &Progress::none(),
    )
    .expect("the track fits")
    .0
}

/// Whether row `i` of `a` and `b` sits at the same keypoint, to the `f32`
/// the column stores.
fn same_keypoint(a: &EditableTrack, b: &EditableTrack, i: usize) -> bool {
    let k = |t: &EditableTrack| t.observations[i].track.as_ref().and_then(|m| m.keypoint);
    k(a) == k(b)
}

/// A fit aligns every other row to the render of the reference the track
/// holds and leaves the reference row's keypoint where it was: the reading
/// after it finds each other row at its correlation peak against that render.
#[test]
fn a_fit_aligns_the_rows_to_the_held_reference_and_leaves_it_unmoved() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let fit_once = fitted(&first, &edited, &views);
    assert!(
        same_keypoint(&first, &fit_once, other),
        "the reference is the anchor"
    );
    let m = fit_once.observations[other].track.as_ref().unwrap();
    assert_eq!(m.seed_shift_px, Some(0.0), "aligned to its own render");
    assert_eq!(m.zncc, Some(1.0));
    let ins: Vec<usize> = (0..fit_once.observations.len())
        .filter(|&i| i != other && fit_once.observations[i].verdict == Verdict::In)
        .collect();
    assert!(ins.len() >= 2);
    assert!(
        ins.iter().any(|&i| !same_keypoint(&first, &fit_once, i)),
        "the other rows are aligned"
    );
    for &i in &ins {
        let m = fit_once.observations[i].track.as_ref().unwrap();
        let shift = m.seed_shift_px.expect("read");
        assert!(shift < 0.5, "row {i} sits {shift} grid px off its peak");
    }
    // Fitting again aligns to the same reference; only the patch the first
    // fit placed again differs, so no row moves by more than a fraction of a
    // pixel.
    let twice = fitted(&fit_once, &edited, &views);
    assert!(same_keypoint(&fit_once, &twice, other));
    for &i in &ins {
        let [a, b] = [keypoint_of(&fit_once, i), keypoint_of(&twice, i)];
        let moved = (a[0] - b[0]).hypot(a[1] - b[1]);
        assert!(moved < 0.25, "row {i} moved {moved} px on a second fit");
    }
}

/// After *Set as reference* on another row, the next fit aligns the rows to
/// that row's render: it is the one left unmoved, and the others move.
#[test]
fn a_fit_after_set_as_reference_aligns_to_the_new_reference() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, picked, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let fit_once = fitted(&first, &edited, &views);
    let (set, _) = set_reference(&fit_once, picked).expect("an in row with a keypoint");
    let set = render_with(&set, &edited, &views);
    assert_eq!(set.held_reference(), Some(picked));
    let fit_again = fitted(&set, &edited, &views);
    assert_eq!(fit_again.held_reference(), Some(picked));
    assert!(
        same_keypoint(&set, &fit_again, picked),
        "the new anchor stays"
    );
    assert!(
        !same_keypoint(&set, &fit_again, other),
        "the old reference is aligned to the new one"
    );
    let m = fit_again.observations[other].track.as_ref().unwrap();
    assert!(
        m.seed_shift_px.is_some_and(|s| s < 0.5),
        "{:?}",
        m.seed_shift_px
    );
    assert_rendered_from(&fit_again, &views, &edited, picked, "after the second fit");
}

/// Splitting the held reference's row off drops the first track's bitmap with
/// its reference, and the next render renders from the rule's pick among the
/// rows left.
#[test]
fn splitting_the_held_reference_off_renders_from_the_pick_of_the_rows_left() {
    let truth = GroundTruth::load();
    let views = truth.views();
    let (track, edited, _, other) = track_with_another_reference(&truth);
    let first = render_with(&track, &edited, &views);
    let (bench, label) = Bench::new().put("T", BenchItem::Track(Arc::new(first)));
    let (bench, report) = split(&bench, &edited, &label, &[other]).expect("a split");
    assert_eq!(report.moved, 1);
    let left = bench.track(&label).expect("the first half");
    assert!(left.track().unwrap().bitmap.is_none());
    assert_eq!(left.track().unwrap().reference, None);
    assert!(unscored(left), "no score outlives the bitmap");
    let read = render_with(left, &edited, &views);
    let pick = rule_pick(&read).expect("the rule picks one");
    assert_rendered_from(&read, &views, &edited, pick, "after the split");
}
