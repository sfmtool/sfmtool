// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view readings an evaluation writes, on a real track: the
//! seoul_bull ground truth and its photographs, which are checked in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array3;

use crate::bench::{
    commit, create_track, evaluate, evaluate_rendering_bitmap, render_bitmap_in_place,
    score_bitmap, tilt_patch, Bench, CreateTrackOptions, EditableTrack, EvaluateOptions,
    FitOptions, Verdict,
};
use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::render_tile;
use crate::camera::warp_map::patch_grid_jacobian;
use crate::camera::{PhotographCache, WarpMap};
use crate::geometry::RigidTransform;
use crate::patch::cloud::{OrientedPatch, PatchCloud};
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
        assert_eq!(m.bitmap_zncc, Some(direct.zncc), "row {i}");
        assert_eq!(
            m.blur_matched_bitmap_zncc,
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
            assert_eq!(m.bitmap_zncc, Some(1.0));
            assert_eq!(m.sharper_than_bitmap, None);
        } else {
            let z = m.bitmap_zncc.expect("a scored row");
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
    let out = refine_patch_keypoints(&patch, &views, &images, Some(&anchors(&keypoints)), &params);
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
        assert!(a.bitmap_zncc.is_some(), "row {i} is scored");
        assert_eq!(a.bitmap_zncc, b.bitmap_zncc, "row {i}");
        assert_eq!(
            a.blur_matched_bitmap_zncc, b.blur_matched_bitmap_zncc,
            "row {i}"
        );
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
        assert_eq!(x.bitmap_zncc, y.bitmap_zncc, "row {i}");
        assert_eq!(
            x.blur_matched_bitmap_zncc, y.blur_matched_bitmap_zncc,
            "row {i}"
        );
        assert_eq!(x.bitmap_blur_sigma, y.bitmap_blur_sigma, "row {i}");
        assert_eq!(x.sharper_than_bitmap, y.sharper_than_bitmap, "row {i}");
    }
}
