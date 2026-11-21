// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster refinement, the view selection and the kernels that read
//! through the reference view (the stored bitmaps, member coherence, the
//! observation readings, the keypoint localizer and sub-pixel refiner, the
//! candidate spawn and *Add image to tracks*) give the same result whichever
//! path a read takes: every read copying its regions into the pool,
//! every read going through the photographs' tiles, and a pool too small for
//! all of a call's regions, where the two paths mix within one call. Each pool
//! holds no more than its capacity and has every byte back once the call
//! returns. A photograph that lacks the pixels a read needs fails the call.
//!
//! The data is the seoul_bull ground truth and its photographs, which are
//! checked in: each point's patch is selected over its track, and each point's
//! track is a cluster whose members sit at their keypoints with the projected
//! patch frame as their shape.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{Array2, Array3};

use crate::camera::image::{ImageU8, ImageU8Pyramid, PyramidLevel};
use crate::camera::sampler::{RegionPool, RegionPoolStats, SamplerChoice, ViewRegions};
use crate::camera::{PhotographCache, PhotographReadError, TiledPhotograph};
use crate::geometry::RigidTransform;
use crate::patch::cloud::PatchCloud;
use crate::patch::cluster_refine::{
    refine_cluster_patches, ClusterRefineParams, ClusterRefineResult, FeatureGeometry,
    PiecewiseParams,
};
use crate::patch::keypoint_localize::{
    localize_patch_cloud_keypoints, KeypointLocalizeParams, LocalizeError, TrackReferences,
};
use crate::patch::keypoint_subpixel::{refine_patch_cloud_keypoints, KeypointSubpixelParams};
use crate::patch::member_coherence::{
    validate_patch_cloud_member_coherence, MemberCoherenceParams,
};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::observation_reading::read_cloud_observations;
use crate::patch::spawn::{spawn_candidate_tracks, SpawnError, SpawnParams};
use crate::patch::stored_bitmap::{
    render_patch_cloud_bitmaps, render_patch_cloud_bitmaps_in, CloudRenderError, UnreferencedPoints,
};
use crate::patch::view_selection::{
    projected_patch_frame, select_patch_cloud_views, select_patch_views, ViewSelectParams,
};
use crate::patch::{track_keypoints_from_reconstruction, track_views_from_reconstruction};
use crate::progress::Progress;
use crate::reconstruction::add_image_to_tracks::{
    add_image_to_tracks, AddImageToTracksError, AddImageToTracksOptions,
};
use crate::reconstruction::bulk_point_order::tests::assert_the_point_order_changes_nothing;
use crate::reconstruction::bulk_point_order::BulkPointOrder;
use crate::reconstruction::SfmrReconstruction;

/// The tile side the photographs are cut into for the reads through tiles:
/// the photographs are 270 x 480, so a read crosses several tiles.
const TILE: u32 = 32;

/// The points of the ground truth the tests read.
const POINTS: usize = 40;

/// Stored patches, with each one's track views and their keypoints.
type Patches = (PatchCloud, Vec<Vec<u32>>, Vec<Vec<Option<[f64; 2]>>>);

/// The ground truth, its poses, and its photographs twice: as decoded, and
/// cut into [`TILE`] px tiles.
struct Truth {
    recon: SfmrReconstruction,
    poses: Vec<RigidTransform>,
    pyramids: Vec<Arc<ImageU8Pyramid>>,
    whole: Vec<TiledPhotograph>,
    tiled: Vec<TiledPhotograph>,
}

impl Truth {
    fn load() -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-data/images/seoul_bull_sculpture");
        let recon = SfmrReconstruction::load(
            &dir.join("seoul_bull_sculpture_ground_truth_v01.sfmr"),
            &Progress::none(),
        )
        .expect("the ground truth loads");
        let images = &recon.image_set.images;
        let paths: Vec<PathBuf> = images.iter().map(|i| dir.join(&i.name)).collect();
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (pyramids, _) = PhotographCache::new(0)
            .get_many(&refs, &Progress::none())
            .expect("nothing cancels the decode");
        let pyramids: Vec<_> = pyramids
            .into_iter()
            .map(|p| p.expect("every photograph is checked in"))
            .collect();
        let whole = pyramids
            .iter()
            .map(|p| TiledPhotograph::from_pyramid(p.clone()))
            .collect();
        let tiled = tiled(&pyramids);
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
            poses,
            pyramids,
            whole,
            tiled,
        }
    }

    /// One view per image, reading `sources`.
    fn views<'a>(&'a self, sources: &'a [TiledPhotograph]) -> Vec<ProjectedImage<'a>> {
        self.recon
            .image_set
            .images
            .iter()
            .zip(&self.poses)
            .zip(sources)
            .map(|((image, cam_from_world), source)| ProjectedImage {
                camera: &self.recon.image_set.cameras[image.camera_index as usize],
                cam_from_world,
                source,
            })
            .collect()
    }

    /// The first [`POINTS`] stored patches, their track views and their
    /// keypoints.
    fn patches(&self) -> Patches {
        let cloud = PatchCloud::from_stored_patch_frames(&self.recon)
            .unwrap()
            .expect("the ground truth has frames");
        let cloud = PatchCloud {
            patches: cloud.patches[..POINTS].to_vec(),
            point_indexes: cloud.point_indexes[..POINTS].to_vec(),
        };
        let tracks = track_views_from_reconstruction(&self.recon, &cloud);
        let keypoints = track_keypoints_from_reconstruction(&self.recon, &cloud);
        (cloud, tracks, keypoints)
    }
}

/// `pyramids`, each read in place and cut into [`TILE`] px tiles.
fn tiled(pyramids: &[Arc<ImageU8Pyramid>]) -> Vec<TiledPhotograph> {
    pyramids
        .iter()
        .map(|p| TiledPhotograph::from_pyramid(p.clone()).with_tile_size(TILE))
        .collect()
}

/// The clusters of [`Truth::patches`]: one per point, its members the
/// observations of its track at their keypoints, each shaped by the patch
/// frame projected into its image and scaled to the template's radius.
struct Clusters {
    positions: Vec<Array2<f32>>,
    shapes: Vec<Array3<f32>>,
    starts: Vec<u32>,
    images: Vec<u32>,
    features: Vec<u32>,
}

impl Clusters {
    fn of(truth: &Truth, params: &ClusterRefineParams) -> Self {
        let views = truth.views(&truth.whole);
        let (cloud, tracks, keypoints) = truth.patches();
        let r = params.radius;
        type Row = ([f64; 2], [[f64; 2]; 2]);
        let mut per_image: Vec<Vec<Row>> = vec![Vec::new(); views.len()];
        let (mut starts, mut images, mut features) = (vec![0u32], Vec::new(), Vec::new());
        for (i, track) in tracks.iter().enumerate() {
            for (t, &image) in track.iter().enumerate() {
                let (Some((_, f)), Some(keypoint)) = (
                    projected_patch_frame(cloud.patch(i), &views[image as usize]),
                    keypoints[i][t],
                ) else {
                    continue;
                };
                let shape = [[f[0][0] / r, -f[0][1] / r], [f[1][0] / r, -f[1][1] / r]];
                images.push(image);
                features.push(per_image[image as usize].len() as u32);
                per_image[image as usize].push((keypoint, shape));
            }
            starts.push(images.len() as u32);
        }
        let positions = per_image
            .iter()
            .map(|rows| Array2::from_shape_fn((rows.len(), 2), |(k, c)| rows[k].0[c] as f32))
            .collect();
        let shapes = per_image
            .iter()
            .map(|rows| {
                Array3::from_shape_fn((rows.len(), 2, 2), |(k, a, b)| rows[k].1[a][b] as f32)
            })
            .collect();
        Self {
            positions,
            shapes,
            starts,
            images,
            features,
        }
    }

    fn geometry(&self) -> Vec<FeatureGeometry<'_>> {
        self.positions
            .iter()
            .zip(&self.shapes)
            .map(|(p, s)| FeatureGeometry {
                positions_xy: p.view(),
                affine_shapes: s.view(),
            })
            .collect()
    }

    fn refine(
        &self,
        sources: &[TiledPhotograph],
        pool: &RegionPool,
        params: &ClusterRefineParams,
    ) -> Result<ClusterRefineResult, PhotographReadError> {
        refine_cluster_patches(
            sources,
            pool,
            &self.geometry(),
            &self.starts,
            &self.images,
            &self.features,
            params,
            None,
        )
    }
}

/// The default refinement with the piecewise stage moving the shape, so
/// every read the refinement makes is taken.
fn cluster_params() -> ClusterRefineParams {
    ClusterRefineParams {
        piecewise: Some(PiecewiseParams {
            move_shape: true,
            ..PiecewiseParams::default()
        }),
        ..ClusterRefineParams::default()
    }
}

/// Every array of a result as text, which shows each float in the shortest
/// form that reads back to the same bits and tells `-0.0` from `0.0`; a
/// result holds `NaN` for a reading that was not taken, and `NaN != NaN`.
fn refined_text(r: &ClusterRefineResult) -> String {
    let mut text = String::new();
    writeln!(
        text,
        "{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{} {}",
        r.reference_members,
        r.member_status,
        r.member_positions.iter().collect::<Vec<_>>(),
        r.member_affine_shapes.iter().collect::<Vec<_>>(),
        r.member_zncc,
        r.member_zncc_middle,
        r.member_zncc_grid,
        r.member_shift_px,
        r.cells,
        r.refined_zncc_self_similarity_radius,
        r.refined_zncc_self_similarity_radius_grid,
        r.elongated_members,
        r.elongated_evaluations,
    )
    .unwrap();
    text
}

/// Run `call` three ways on one rayon thread, so that the reads come in the
/// same order each time, and assert it returns the same text each time:
///
/// - (a) over the photographs as decoded, through a pool that holds any
///   read, so every read of a pixel copies its regions;
/// - (b) over the photographs cut into [`TILE`] px tiles, through a pool of no
///   bytes, so every read goes through the tiles;
/// - (c) over the tiled photographs, through a pool of half the most bytes
///   (a) held at once, so some reads copy their regions and others go through
///   the tiles.
///
/// Each pool holds no more than its capacity and holds nothing once the call
/// returns. With `reuses`, (a) reuses regions a read kept, as a call that
/// reads a view more than once does; a call that reads each view once
/// passes `false`.
fn assert_the_same_on_every_path(
    truth: &Truth,
    reuses: bool,
    call: impl Fn(&[TiledPhotograph], &RegionPool) -> String + Sync,
) {
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool of one thread");
    let run = |sources: &[TiledPhotograph], capacity: u64| -> (String, RegionPoolStats) {
        let pool = RegionPool::with_capacity(capacity);
        let text = one_thread.install(|| call(sources, &pool));
        let stats = pool.stats();
        assert!(stats.peak_bytes <= capacity, "{stats:?} over {capacity}");
        assert_eq!(pool.held_bytes(), 0, "{stats:?}");
        (text, stats)
    };

    let (a, a_stats) = run(&truth.whole, u64::MAX);
    assert!(
        a_stats.copied_reads > 0
            && (a_stats.reused_reads > 0) == reuses
            && a_stats.tiled_reads == 0,
        "the reads copy their regions, reusing them as the call reads: {a_stats:?}"
    );

    let (b, b_stats) = run(&truth.tiled, 0);
    assert!(
        b_stats.tiled_reads > 0 && b_stats.copied_reads + b_stats.reused_reads == 0,
        "every read goes through the tiles: {b_stats:?}"
    );
    assert_same("copied regions", &a, "tiles", &b);

    let (c, c_stats) = run(&truth.tiled, a_stats.peak_bytes / 2);
    assert!(
        c_stats.copied_reads > 0 && c_stats.tiled_reads > 0,
        "a pool of {} bytes copies some reads and takes others through the tiles: \
         {c_stats:?}",
        a_stats.peak_bytes / 2
    );
    assert_same("copied regions", &a, "a small pool", &c);
}

/// `a == b`, and on a difference, the first place the texts part.
fn assert_same(a_name: &str, a: &str, b_name: &str, b: &str) {
    if a != b {
        let at = a
            .bytes()
            .zip(b.bytes())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        let from = at.saturating_sub(200);
        panic!(
            "the results differ from byte {at}:\n{a_name}: ...{}\n{b_name}: ...{}",
            &a[from..(at + 200).min(a.len())],
            &b[from..(at + 200).min(b.len())],
        );
    }
}

#[test]
fn the_cluster_refinement_is_the_same_on_every_read_path() {
    let truth = Truth::load();
    let params = cluster_params();
    let clusters = Clusters::of(&truth, &params);
    assert_the_same_on_every_path(&truth, true, |sources, pool| {
        let result = clusters
            .refine(sources, pool, &params)
            .expect("photographs in memory read");
        assert!(
            result
                .cells
                .iter()
                .flatten()
                .any(|cells| cells.iterations >= 2),
            "the piecewise stage moves a shape and renders it again, so its reads are taken"
        );
        refined_text(&result)
    });
}

#[test]
fn the_view_selection_is_the_same_on_every_read_path() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    let params = ViewSelectParams::default();
    assert_the_same_on_every_path(&truth, true, |sources, pool| {
        let views = truth.views(sources);
        let batch = select_patch_cloud_views(
            &cloud,
            &views,
            pool,
            &tracks,
            Some(&keypoints),
            &params,
            None,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        assert!(
            batch
                .iter()
                .any(|selection| selection.admitted.len() > selection.track_view_count),
            "a selection admits a candidate, so candidates are scored"
        );
        let single = select_patch_views(
            cloud.patch(0),
            &views,
            pool,
            &tracks[0],
            None,
            &params,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        format!("{batch:?}\n{single:?}")
    });
}

/// `source` replaced by one that holds only an 8 px square in the corner of
/// its finest level and none of its coarser levels, so that a read anywhere
/// near a patch asks for pixels it does not hold.
fn without_pixels(source: &TiledPhotograph) -> TiledPhotograph {
    let (width, height) = source.layout().level_size(0);
    let channels = source.channels();
    let corner = PyramidLevel {
        image: ImageU8::new(8, 8, channels, vec![128; 64 * channels as usize]),
        origin: [0, 0],
        full_size: [width, height],
    };
    let mut levels: Vec<Option<PyramidLevel>> =
        (0..source.layout().levels()).map(|_| None).collect();
    levels[0] = Some(corner);
    let pyramid = ImageU8Pyramid::from_regions(levels).expect("a valid region pyramid");
    TiledPhotograph::from_pyramid(Arc::new(pyramid)).with_tile_size(TILE)
}

/// A photograph that lacks the pixels a read needs fails the cluster
/// refinement and the view selection with the read's error, on either path,
/// rather than reading as a refusal.
#[test]
fn a_read_of_pixels_a_photograph_lacks_fails_the_call() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    // An image the first point is seen in, which every one of its readers
    // reads.
    let image = tracks[0][0] as usize;
    let mut sources = tiled(&truth.pyramids);
    sources[image] = without_pixels(&truth.tiled[image]);
    let params = cluster_params();
    let clusters = Clusters::of(&truth, &params);
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let refused = clusters
            .refine(&sources, &pool, &params)
            .err()
            .expect("the image cannot be read");
        assert!(
            matches!(refused, PhotographReadError::OutOfRange { .. }),
            "pool of {capacity}: {refused}"
        );
        assert_eq!(pool.held_bytes(), 0);

        let views = truth.views(&sources);
        let refused = select_patch_cloud_views(
            &cloud,
            &views,
            &pool,
            &tracks,
            Some(&keypoints),
            &ViewSelectParams::default(),
            None,
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(
            matches!(refused, PhotographReadError::OutOfRange { .. }),
            "pool of {capacity}: {refused}"
        );
        assert_eq!(pool.held_bytes(), 0);
    }
}

/// The flat bytes of an array, whose `Debug` text elides a long one.
fn flat<D: ndarray::Dimension>(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = u8>, D>,
) -> Vec<u8> {
    a.iter().copied().collect()
}

/// `views`, each as `Some`, for the whole-cloud renders.
fn present<'a>(views: &[ProjectedImage<'a>]) -> Vec<Option<ProjectedImage<'a>>> {
    views.iter().map(|v| Some(*v)).collect()
}

/// The stored bitmaps' render over the points: each point's reference
/// observation's tile, or, with its references dropped, the reference-view
/// rule's pick, whose tiles and pair readings share each view's regions.
#[test]
fn the_stored_bitmaps_are_the_same_on_every_read_path() {
    let truth = Truth::load();
    let (cloud, _, _) = truth.patches();
    let mut unreferenced = truth.recon.clone();
    unreferenced.point_set.reference_observations = None;
    assert_the_same_on_every_path(&truth, true, |sources, pool| {
        let views = present(&truth.views(sources));
        let mut text = String::new();
        for recon in [&truth.recon, &unreferenced] {
            let column = render_patch_cloud_bitmaps(
                &cloud,
                recon,
                &views,
                pool,
                &KeypointSubpixelParams::default(),
                UnreferencedPoints::Pick,
                None,
                &Progress::none(),
            )
            .expect("photographs in memory read");
            writeln!(
                text,
                "{:?} {:?}",
                flat(&column.bitmaps),
                column.reference_observations
            )
            .unwrap();
        }
        text
    });
}

/// The whole-cloud render in the bulk kernels' point order gives the bitmaps
/// and references of a render over the patches by index, and reads through
/// the one pool its caller made. The ground truth stores `-1` for these
/// points, so each point is given a reference, a different observation of its
/// track from point to point, except every third point, which keeps none, so
/// the rule's picks are rendered too and those points sort last.
#[test]
fn the_stored_bitmaps_in_the_point_order_are_those_by_index() {
    let truth = Truth::load();
    let (cloud, _, _) = truth.patches();
    let mut recon = truth.recon.clone();
    let set = &mut recon.point_set;
    let mut references = vec![-1i32; set.points.len()];
    for &p in &cloud.point_indexes {
        let p = p as usize;
        if !p.is_multiple_of(3) {
            references[p] = (p as u32 % set.observation_counts[p]) as i32;
        }
    }
    set.reference_observations = Some(references);
    let views = present(&truth.views(&truth.tiled));
    let order = BulkPointOrder::of_points(&recon.point_set, &cloud.point_indexes);
    assert_the_point_order_changes_nothing(&order, |order, pool| {
        let column = render_patch_cloud_bitmaps_in(
            &cloud,
            &recon,
            &views,
            pool,
            &KeypointSubpixelParams::default(),
            UnreferencedPoints::Pick,
            None,
            order,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        format!(
            "{:?} {:?}",
            flat(&column.bitmaps),
            column.reference_observations
        )
    });
    // A point with a reference renders that one view, and the read is the
    // caller's pool's: rendering only those points counts one read each.
    let pool = RegionPool::with_capacity(u64::MAX);
    render_patch_cloud_bitmaps(
        &cloud,
        &recon,
        &views,
        &pool,
        &KeypointSubpixelParams::default(),
        UnreferencedPoints::Skip,
        None,
        &Progress::none(),
    )
    .expect("photographs in memory read");
    let stats = pool.stats();
    let referenced = cloud
        .point_indexes
        .iter()
        .filter(|&&p| !p.is_multiple_of(3))
        .count();
    assert!(referenced > 0);
    assert_eq!(
        stats.copied_reads + stats.reused_reads + stats.tiled_reads,
        referenced as u64,
        "{stats:?}"
    );
}

/// Member coherence's matrices and verdicts over the points' tracks.
#[test]
fn member_coherence_is_the_same_on_every_read_path() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    assert_the_same_on_every_path(&truth, false, |sources, pool| {
        let coherence = validate_patch_cloud_member_coherence(
            &cloud,
            &truth.views(sources),
            pool,
            &tracks,
            Some(&keypoints),
            &MemberCoherenceParams::default(),
            None,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        assert!(
            coherence.iter().any(|c| c.matrix.n_support > 0),
            "a matrix has support, so the members are rendered"
        );
        format!("{coherence:?}")
    });
}

/// Each observation's readings on its own tile, scored against its point's
/// stored bitmap.
#[test]
fn the_observation_readings_are_the_same_on_every_read_path() {
    let truth = Truth::load();
    let (cloud, _, _) = truth.patches();
    assert_the_same_on_every_path(&truth, false, |sources, pool| {
        let readings = read_cloud_observations(
            &cloud,
            &truth.recon,
            &present(&truth.views(sources)),
            pool,
            24,
            SamplerChoice::default(),
            None,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        format!("{readings:?}")
    });
}

/// The localizer and the sub-pixel refiner over the points' tracks, the
/// refiner rendering each point's stored bitmap, and a spawn of candidate
/// tracks beside the first points, which runs both.
#[test]
fn the_keypoint_kernels_are_the_same_on_every_read_path() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    let parents: Vec<u32> = (0..8).collect();
    let offsets: Vec<[f64; 2]> = parents
        .iter()
        .map(|&p| if p % 2 == 0 { [0.6, 0.0] } else { [0.0, -0.6] })
        .collect();
    let spawn_views: Vec<Vec<u32>> = parents
        .iter()
        .map(|&p| tracks[p as usize].clone())
        .collect();
    assert_the_same_on_every_path(&truth, true, |sources, pool| {
        let views = truth.views(sources);
        let localized = localize_patch_cloud_keypoints(
            &cloud,
            &views,
            pool,
            &tracks,
            Some(&keypoints),
            None,
            &KeypointLocalizeParams::default(),
            None,
            &Progress::none(),
        )
        .expect("photographs in memory read");
        let refined = refine_patch_cloud_keypoints(
            &cloud,
            &views,
            pool,
            &tracks,
            Some(&keypoints),
            None,
            &KeypointSubpixelParams {
                render_bitmaps: true,
                ..KeypointSubpixelParams::default()
            },
            &Progress::none(),
        )
        .expect("photographs in memory read");
        assert!(refined.iter().any(|r| r.representative.is_some()));
        let spawned = spawn_candidate_tracks(
            &views,
            pool,
            &cloud,
            &parents,
            &offsets,
            &spawn_views,
            &SpawnParams::default(),
        )
        .expect("photographs in memory read");
        format!("{localized:?}\n{refined:?}\n{spawned:?}")
    });
}

/// *Add image to tracks* for one image: its references' renders, the search,
/// the sub-pixel step and the scores of every candidate point.
#[test]
fn adding_an_image_to_tracks_is_the_same_on_every_read_path() {
    let truth = Truth::load();
    assert_the_same_on_every_path(&truth, true, |sources, pool| {
        let photographs: Vec<Option<&TiledPhotograph>> = sources.iter().map(Some).collect();
        let (next, report) = add_image_to_tracks(
            &truth.recon,
            5,
            &photographs,
            pool,
            &AddImageToTracksOptions::default(),
            &Progress::none(),
        )
        .expect("photographs in memory read");
        assert!(
            report.accepted > 0,
            "a candidate is accepted, so every read is taken"
        );
        format!(
            "{:?}\n{:?}\n{:?}",
            next.point_set.tracks, next.point_set.observation_readings, report
        )
    });
}

/// A photograph that lacks the pixels a read needs fails each kernel of the
/// reference view's family with the read's error, on either path: the reads
/// of each patch of a batch, and of each candidate of *Add image to tracks*,
/// turn a failure into that patch's value, and the batch returns the first.
#[test]
fn a_read_of_pixels_a_photograph_lacks_fails_each_reference_view_kernel() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    let image = tracks[0][0] as usize;
    let mut sources = tiled(&truth.pyramids);
    sources[image] = without_pixels(&truth.tiled[image]);
    let is_out_of_range =
        |e: &PhotographReadError| matches!(e, PhotographReadError::OutOfRange { .. });
    for capacity in [u64::MAX, 0] {
        let pool = RegionPool::with_capacity(capacity);
        let views = truth.views(&sources);
        let error = render_patch_cloud_bitmaps(
            &cloud,
            &truth.recon,
            &present(&views),
            &pool,
            &KeypointSubpixelParams::default(),
            UnreferencedPoints::Pick,
            None,
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(
            matches!(&error, CloudRenderError::Photograph(e) if is_out_of_range(e)),
            "pool of {capacity}: {error}"
        );
        let error = read_cloud_observations(
            &cloud,
            &truth.recon,
            &present(&views),
            &pool,
            24,
            SamplerChoice::default(),
            None,
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(
            matches!(&error, CloudRenderError::Photograph(e) if is_out_of_range(e)),
            "pool of {capacity}: {error}"
        );
        let error = validate_patch_cloud_member_coherence(
            &cloud,
            &views,
            &pool,
            &tracks,
            Some(&keypoints),
            &MemberCoherenceParams::default(),
            None,
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(is_out_of_range(&error), "pool of {capacity}: {error}");
        let error = localize_patch_cloud_keypoints(
            &cloud,
            &views,
            &pool,
            &tracks,
            Some(&keypoints),
            None,
            &KeypointLocalizeParams::default(),
            None,
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(is_out_of_range(&error), "pool of {capacity}: {error}");
        let error = refine_patch_cloud_keypoints(
            &cloud,
            &views,
            &pool,
            &tracks,
            Some(&keypoints),
            None,
            &KeypointSubpixelParams::default(),
            &Progress::none(),
        )
        .expect_err("the image cannot be read");
        assert!(is_out_of_range(&error), "pool of {capacity}: {error}");
        let error = spawn_candidate_tracks(
            &views,
            &pool,
            &cloud,
            &[0],
            &[[0.6, 0.0]],
            &[tracks[0].clone()],
            &SpawnParams::default(),
        )
        .expect_err("the image cannot be read");
        assert!(
            matches!(&error, SpawnError::Photograph(e) if is_out_of_range(e)),
            "pool of {capacity}: {error}"
        );
        // The image as the target, whose search reads it first, and as a
        // reference of the points another target sees, whose renders read it
        // first; on the next image no candidate reaches a reference render of
        // it, so the test takes the one after.
        let photographs: Vec<Option<&TiledPhotograph>> = sources.iter().map(Some).collect();
        let target = (image + 2) % sources.len();
        for added in [image, target] {
            let error = add_image_to_tracks(
                &truth.recon,
                added,
                &photographs,
                &pool,
                &AddImageToTracksOptions::default(),
                &Progress::none(),
            )
            .err()
            .unwrap_or_else(|| panic!("image {added} cannot be read"));
            assert!(
                matches!(&error, AddImageToTracksError::Photograph(e) if is_out_of_range(e)),
                "image {added}, pool of {capacity}: {error}"
            );
        }
        assert_eq!(pool.held_bytes(), 0);
    }
}

/// A point's references read from photographs that hold their pixels, and
/// the search and score of a view whose photograph lacks them: each fails
/// with the read's error rather than reading as no peak or no score.
#[test]
fn a_search_or_score_of_a_view_that_cannot_be_read_fails() {
    let truth = Truth::load();
    let (cloud, tracks, keypoints) = truth.patches();
    let views = truth.views(&truth.whole);
    let references = &tracks[0][1..];
    let reference_keypoints: Vec<[f64; 2]> = keypoints[0][1..]
        .iter()
        .map(|k| k.expect("a stored keypoint"))
        .collect();
    let pool = RegionPool::with_capacity(u64::MAX);
    let mut references: Vec<(u32, ViewRegions<'_>)> = references
        .iter()
        .map(|&v| (v, ViewRegions::new(views[v as usize].source, &pool)))
        .collect();
    let params = KeypointLocalizeParams::default();
    let built = TrackReferences::build(
        cloud.patch(0),
        &views,
        &mut references,
        &reference_keypoints,
        None,
        None,
        &params,
    )
    .expect("photographs in memory read")
    .expect("the references settle a template");
    let image = tracks[0][0] as usize;
    let broken = without_pixels(&truth.tiled[image]);
    let mut target = views[image];
    target.source = &broken;
    let mut target_regions = ViewRegions::new(&broken, &pool);
    let error = built
        .search(
            cloud.patch(0),
            &target,
            &mut target_regions,
            None,
            false,
            &params,
        )
        .expect_err("the target cannot be read");
    assert!(
        matches!(
            error,
            LocalizeError::Photograph(PhotographReadError::OutOfRange { .. })
        ),
        "{error}"
    );
    let keypoint = keypoints[0][0].expect("a stored keypoint");
    let error = built
        .score(
            cloud.patch(0),
            &target,
            &mut target_regions,
            keypoint,
            &params,
        )
        .expect_err("the target cannot be read");
    assert!(
        matches!(error, PhotographReadError::OutOfRange { .. }),
        "{error}"
    );
}
