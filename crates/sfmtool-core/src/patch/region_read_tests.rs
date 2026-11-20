// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The cluster refinement and the view selection give the same result
//! whichever path a read takes: every read copying its regions into the pool,
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
use crate::camera::sampler::{RegionPool, RegionPoolStats};
use crate::camera::{PhotographCache, PhotographReadError, TiledPhotograph};
use crate::geometry::RigidTransform;
use crate::patch::cloud::PatchCloud;
use crate::patch::cluster_refine::{
    refine_cluster_patches, ClusterRefineParams, ClusterRefineResult, FeatureGeometry,
    PiecewiseParams,
};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::view_selection::{
    projected_patch_frame, select_patch_cloud_views, select_patch_views, ViewSelectParams,
};
use crate::patch::{track_keypoints_from_reconstruction, track_views_from_reconstruction};
use crate::progress::Progress;
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
/// returns.
fn assert_the_same_on_every_path(
    truth: &Truth,
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
        a_stats.copied_reads > 0 && a_stats.reused_reads > 0 && a_stats.tiled_reads == 0,
        "the reads copy their regions and reuse them: {a_stats:?}"
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
    assert_the_same_on_every_path(&truth, |sources, pool| {
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
    assert_the_same_on_every_path(&truth, |sources, pool| {
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
