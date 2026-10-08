// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reference-view readings an evaluation writes, on a real track: the
//! seoul_bull ground truth and its photographs, which are checked in.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::Array3;

use crate::bench::{
    create_track, evaluate, set_verdict, Bench, CreateTrackOptions, EditableTrack, EvaluateOptions,
    ReferenceViewOptions, Verdict,
};
use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::render_tile;
use crate::camera::warp_map::patch_grid_jacobian;
use crate::camera::{PhotographCache, WarpMap};
use crate::geometry::RigidTransform;
use crate::patch::member_coherence::{member_zncc_matrix, MemberCoherenceParams};
use crate::patch::normal_refine::{PatchWindow, ProjectedImage};
use crate::patch::pair_sharpness::PairMatching;
use crate::patch::reference_view::{
    blur_matched_agreement, finite_middle, render_view_tile, PairZnccReading, ReferenceFallback,
    ReferenceRuleInputs, ReferenceTest, ViewTile, REFERENCE_MAX_VIEWING_ANGLE_DEG,
    REFERENCE_MIN_COVERAGE,
};
use crate::patch::self_similarity::{
    zncc_self_similarity_parts, PatchTile, SelfSimilarityEllipseUnits, SelfSimilarityParams,
};
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

/// Options that take blur-matched readings and have both tests read them.
fn blur_matched_options(matching: PairMatching) -> EvaluateOptions {
    EvaluateOptions {
        reference_view: ReferenceViewOptions {
            matching,
            agreement: PairZnccReading::BlurMatched,
            cells: PairZnccReading::BlurMatched,
        },
        ..EvaluateOptions::default()
    }
}

/// The blur-matched readings are taken where the options ask for them, are
/// those of the tiles blur-matched directly, leave the plain readings as they
/// were, and the rule's standing says which readings it read.
#[test]
fn blur_matched_readings_are_those_of_the_tiles_blur_matched_directly() {
    let truth = GroundTruth::load();
    let views = truth.views();
    // Plain matching takes no blur-matched readings, whatever the tests are
    // set to read.
    let none = blur_matched_options(PairMatching::Plain);
    let (plain, _) = truth.evaluated_track_with(&none);
    let options = blur_matched_options(PairMatching::BlurMatched);
    let (read, edited) = truth.evaluated_track_with(&options);
    let resolution = options.patch_resolution(&edited.base) as usize;
    let frame = read
        .track()
        .and_then(|t| t.placement.clone())
        .expect("a track-stage frame");
    let rows: Vec<usize> = (0..read.observations.len())
        .filter(|&i| read.observations[i].verdict == Verdict::In)
        .collect();
    let tiles: Vec<ViewTile> = rows
        .iter()
        .map(|&i| {
            render_view_tile(
                &frame,
                &views[read.observations[i].image as usize],
                Some(keypoint_of(&read, i)),
                resolution,
                options.localize.sampler,
                &Progress::none(),
            )
        })
        .collect();
    let ellipses: Vec<Option<[[f64; 2]; 2]>> = rows
        .iter()
        .map(|&i| {
            read.observations[i]
                .track
                .as_ref()
                .and_then(|m| m.zncc_self_similarity_ellipse)
                .map(|e| e.grid_px.matrix)
        })
        .collect();
    let refs: Vec<&ViewTile> = tiles.iter().collect();
    let direct = blur_matched_agreement(
        &refs,
        &ellipses,
        PairMatching::BlurMatched,
        PatchWindow::GaussianDisk { sigma: 0.6 },
        &Progress::none(),
    );
    for (k, &i) in rows.iter().enumerate() {
        let m = read.observations[i].track.as_ref().unwrap();
        let p = plain.observations[i].track.as_ref().unwrap();
        let finite = |v: f64| v.is_finite().then_some(v);
        assert_eq!(
            m.blur_matched_pair_zncc,
            finite(direct.pair_zncc[k]),
            "row {i}"
        );
        assert_eq!(
            format!("{:?}", m.blur_matched_pair_zncc_grid),
            format!("{:?}", Some(direct.cells.pair_zncc_grid[k])),
            "row {i}"
        );
        assert_eq!(m.blur_matched_cell_deficit, finite(direct.cells.deficit[k]));
        // The plain readings are the same whether or not the blur-matched
        // ones are taken.
        assert_eq!(m.pair_zncc, p.pair_zncc, "row {i}");
        assert_eq!(m.cell_deficit, p.cell_deficit, "row {i}");
        let standing = m.reference_view.expect("an in row has a standing");
        assert_eq!(
            standing.inputs,
            ReferenceRuleInputs {
                agreement: PairZnccReading::BlurMatched,
                cells: PairZnccReading::BlurMatched,
            }
        );
        // Without blur-matched readings the rule reads the plain ones,
        // whatever the options name.
        assert_eq!(p.blur_matched_pair_zncc, None);
        assert_eq!(p.blur_matched_pair_zncc_grid, None);
        assert_eq!(p.blur_matched_cell_deficit, None);
        assert_eq!(p.reference_view.unwrap().inputs, ReferenceRuleInputs::PLAIN);
    }
}

/// Above a ratio, a pair whose ellipses differ by less is read plain, which
/// is what the blur-matched reading of a pair left plain is.
#[test]
fn a_ratio_above_every_difference_reads_every_pair_plain() {
    let truth = GroundTruth::load();
    let (read, _) = truth.evaluated_track_with(&blur_matched_options(
        PairMatching::BlurMatchedAboveRatio(1e6),
    ));
    for (i, observation) in read.observations.iter().enumerate() {
        let m = observation.track.as_ref().unwrap();
        assert_eq!(
            format!("{:?}", m.blur_matched_pair_zncc_grid),
            format!("{:?}", m.pair_zncc_grid),
            "row {i}: a pair read plain over each ninth reads the plain grid"
        );
        assert_eq!(m.blur_matched_cell_deficit, m.cell_deficit, "row {i}");
    }
}

/// A verdict moved after the reading keeps the rule on the readings it read:
/// the standings after the step name the same inputs.
#[test]
fn a_verdict_set_after_a_blur_matched_reading_keeps_the_rule_on_its_inputs() {
    let truth = GroundTruth::load();
    let (read, _) = truth.evaluated_track_with(&blur_matched_options(PairMatching::BlurMatched));
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
    let (after, _) = set_verdict(&read, picked, Verdict::Out).expect("a live row");
    let m = after.observations[picked].track.as_ref().unwrap();
    assert_eq!(m.blur_matched_pair_zncc, None);
    assert_eq!(m.blur_matched_cell_deficit, None);
    let mut picked_again = 0;
    for observation in &after.observations {
        if observation.verdict != Verdict::In {
            continue;
        }
        let standing = observation.track.as_ref().unwrap().reference_view.unwrap();
        assert_eq!(standing.inputs.agreement, PairZnccReading::BlurMatched);
        assert_eq!(standing.inputs.cells, PairZnccReading::BlurMatched);
        picked_again += usize::from(standing.is_reference());
    }
    assert_eq!(picked_again, 1);
}
