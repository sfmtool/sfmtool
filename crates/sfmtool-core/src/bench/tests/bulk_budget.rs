// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bulk kernels whose photographs are read through a photograph cache with a
//! budget smaller than their tiles: *Refit all points* (`refit_points`), whose
//! fit of each point renders every view of it, and the stored bitmaps'
//! whole-cloud render, which renders each point from its reference image only,
//! on photographs written to a temporary directory and fetched with one
//! `get_many`, as SfM Explorer's jobs fetch them.
//!
//! The scene is six cameras in a row over the textured plane, with nine
//! points to each reference image, spread over its photograph's tiles. Each
//! point is seen by its reference image and that image's neighbours, so the
//! photographs one reference image's points are seen in are three of the six.
//! The points' indexes take the reference images in turn, so the run over the
//! points by index is a shuffled order and the run in the bulk kernels' point
//! order takes each reference image's points together.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::Array2;

use super::scene::{fixture_points, with_columns, PLANE_Z};
use super::*;
use crate::bench::refit_points::refit_points_in;
use crate::camera::photograph_cache::SlotEvent;
use crate::camera::sampler::RegionPool;
use crate::camera::{PhotographCache, PhotographCacheOptions, TiledPhotograph};
use crate::patch::cloud::PatchCloud;
use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::stored_bitmap::{render_patch_cloud_bitmaps, UnreferencedPoints};
use crate::reconstruction::bulk_point_order::BulkPointOrder;
use crate::reconstruction::data::TrackObservation;

/// The bitmap side the value stores.
const BITMAP_R: usize = 8;

/// The cache's tile size: the 128 px photographs are 8 tiles a side at level
/// 0, so a render reads a few of a photograph's tiles.
const TILE: u32 = 16;

/// The six cameras' centres, in a row along `x`.
const ROW: [[f64; 3]; 6] = [
    [-0.5, 0.0, 0.0],
    [-0.3, 0.0, 0.0],
    [-0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.3, 0.0, 0.0],
    [0.5, 0.0, 0.0],
];

/// Each reference image's points, as offsets on the plane from the point
/// its camera looks at.
const OFFSETS: [[f64; 2]; 9] = [
    [-0.6, -0.6],
    [0.0, -0.6],
    [0.6, -0.6],
    [-0.6, 0.0],
    [0.0, 0.0],
    [0.6, 0.0],
    [-0.6, 0.6],
    [0.0, 0.6],
    [0.6, 0.6],
];

/// The images that see a point whose reference image is `r`: `r` and its
/// neighbours in the row.
fn seen_in(r: usize) -> std::ops::RangeInclusive<usize> {
    r.saturating_sub(1)..=(r + 1).min(ROW.len() - 1)
}

/// The scene and a value over it of nine points per reference image, point
/// `p`'s reference image being `p % 6`, each point seen by the images of
/// [`seen_in`] at its exact projections, with a bitmap column.
fn row_scene() -> (Scene, SfmrReconstruction) {
    let scene = Scene::from_centers(&ROW, PLANE_Z);
    let n = ROW.len();
    let points: Vec<Point3<f64>> = (0..n * OFFSETS.len())
        .map(|p| {
            let (r, [dx, dy]) = (p % n, OFFSETS[p / n]);
            Point3::new(ROW[r][0] + dx, ROW[r][1] + dy, PLANE_Z)
        })
        .collect();
    let mut recon = fixture_points(&scene, &points);
    let set = &mut recon.point_set;
    let mut tracks = Vec::new();
    let mut counts = Vec::new();
    let mut keypoints = Vec::new();
    let mut references = Vec::new();
    for (p, &world) in points.iter().enumerate() {
        let r = p % n;
        let images = seen_in(r);
        references.push((r - images.start()) as i32);
        counts.push(images.clone().count() as u32);
        for image in images {
            tracks.push(TrackObservation {
                image_index: image as u32,
                point_index: p as u32,
            });
            let px = scene.project(image, world);
            keypoints.extend([px[0] as f32, px[1] as f32]);
        }
    }
    set.keypoints_xy = Some(Array2::from_shape_vec((tracks.len(), 2), keypoints).expect("(N, 2)"));
    set.tracks = tracks;
    set.observation_counts = counts;
    set.reference_observations = Some(references);
    let recon = with_columns(recon, BITMAP_R);
    (scene, recon)
}

/// What a run through a cache did: the value's hash and the report, which no
/// budget or order changes, and each photograph's decodes in the read phase
/// and in the renders.
struct Run {
    answer: String,
    read_phase: BTreeMap<PathBuf, usize>,
    renders: BTreeMap<PathBuf, usize>,
    /// The cache's own count of its decodes.
    decodes: u64,
}

impl Run {
    fn render_decodes(&self) -> usize {
        self.renders.values().sum()
    }
}

/// Which kernel a run is.
#[derive(Clone, Copy)]
enum Kernel<'a> {
    /// `refit_points`, whose fit of each point renders every view of it, in
    /// its own point order, or in the order given.
    Refit(Option<&'a BulkPointOrder>),
    /// The stored bitmaps' whole-cloud render (`render_patch_cloud_bitmaps`),
    /// which renders each point from its reference observation only, as the
    /// display bitmaps do.
    Bitmaps,
}

/// Run `kernel` over `recon` on a rayon pool of one thread, reading `paths`,
/// the scene's photographs, fetched with one `get_many` through a cache of
/// `budget` bytes and tiles of [`TILE`] px.
fn run_through_cache(
    scene: &Scene,
    recon: &SfmrReconstruction,
    paths: &[PathBuf],
    budget: u64,
    kernel: Kernel<'_>,
) -> Run {
    let mut cache = PhotographCache::new(PhotographCacheOptions {
        budget_bytes: budget,
        tile_size: TILE,
        ..Default::default()
    });
    let decoded = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
    cache.set_slot_hook(Box::new({
        let decoded = Arc::clone(&decoded);
        move |_, event, path| {
            if event == SlotEvent::DecodeStarted {
                decoded.lock().unwrap().push(path.to_path_buf());
            }
        }
    }));
    let tally = |decoded: Vec<PathBuf>| {
        let mut counts = BTreeMap::new();
        for path in decoded {
            *counts.entry(path).or_insert(0) += 1;
        }
        counts
    };
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a pool");
    one_thread.install(|| {
        let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
        let (found, _) = cache
            .get_many(&refs, &Progress::none())
            .expect("not cancelled");
        let read_phase = tally(std::mem::take(&mut *decoded.lock().unwrap()));
        let sources: Vec<Arc<TiledPhotograph>> = found
            .into_iter()
            .map(|photograph| photograph.expect("readable"))
            .collect();
        let views = scene.views_of(&sources);
        let pool = RegionPool::for_kernel_call();
        let answer = match kernel {
            Kernel::Refit(order) => {
                let options = RefitPointsOptions {
                    fit: test_fit_options(),
                    ..RefitPointsOptions::default()
                };
                let (value, report) = match order {
                    None => refit_points(recon, &views, &pool, &options, &Progress::none()),
                    Some(order) => {
                        refit_points_in(recon, &views, &pool, &options, order, &Progress::none())
                    }
                }
                .expect("the run refits");
                assert_eq!(report.refit(), recon.point_count(), "{report:?}");
                format!(
                    "{:?} {report:?}",
                    value.content_xxh128().expect("hashes").content_xxh128
                )
            }
            Kernel::Bitmaps => {
                let cloud = PatchCloud::from_stored_patch_frames(recon)
                    .expect("readable frames")
                    .expect("frames");
                let views: Vec<Option<ProjectedImage<'_>>> = views.into_iter().map(Some).collect();
                let column = render_patch_cloud_bitmaps(
                    &cloud,
                    recon,
                    &views,
                    &pool,
                    &KeypointSubpixelParams::default(),
                    UnreferencedPoints::Pick,
                    None,
                    &Progress::none(),
                )
                .expect("the render completes");
                assert!(column.bitmaps.iter().any(|&v| v != 0));
                format!("{:?}", column.bitmaps)
            }
        };
        let renders = tally(std::mem::take(&mut *decoded.lock().unwrap()));
        Run {
            answer,
            read_phase,
            renders,
            decodes: cache.stats().decodes,
        }
    })
}

/// [`run_through_cache`] of `refit_points` in its own point order, or in
/// `order`.
fn refit_through_cache(
    scene: &Scene,
    recon: &SfmrReconstruction,
    paths: &[PathBuf],
    budget: u64,
    order: Option<&BulkPointOrder>,
) -> Run {
    run_through_cache(scene, recon, paths, budget, Kernel::Refit(order))
}

/// The bytes the cache keeps of one of the scene's photographs, every one of
/// which is the same size.
fn photograph_bytes(path: &Path) -> u64 {
    let cache = PhotographCache::new(PhotographCacheOptions {
        budget_bytes: 1 << 30,
        tile_size: TILE,
        ..Default::default()
    });
    cache.get(path, &Progress::none()).expect("readable");
    cache.stats().bytes
}

/// Under a budget that holds one photograph's tiles of the six, and under
/// one smaller than one photograph's level 0, the run completes with the
/// answer it gives when every tile fits, its renders decoding photographs
/// again, which the cache counts.
#[test]
fn a_run_under_a_budget_smaller_than_its_photographs_completes() {
    let (scene, recon) = row_scene();
    let dir = tempfile::tempdir().unwrap();
    let paths = scene.write_photographs(dir.path());
    let all = refit_through_cache(&scene, &recon, &paths, 1 << 30, None);
    assert_eq!(all.render_decodes(), 0, "every tile fits");

    let one = photograph_bytes(&paths[0]);
    let level_0 = u64::from(IMG_W * IMG_H) * 3;
    assert!(level_0 < one);
    for budget in [one, level_0 / 2] {
        let run = refit_through_cache(&scene, &recon, &paths, budget, None);
        assert_eq!(run.answer, all.answer, "budget {budget}");
        assert!(run.render_decodes() > 0, "budget {budget}");
        assert_eq!(
            run.decodes as usize,
            paths.len() + run.render_decodes(),
            "budget {budget}: the cache counts every decode"
        );
    }
}

/// On a rayon pool of one thread, under a budget smaller than the scene's
/// tiles but holding the tiles of the three photographs one reference image's
/// points are seen in, each photograph is decoded once in the read phase and
/// at most once more for each reference image whose points it sees; the same
/// points taken by index, a shuffled order, decode more.
///
/// The budget holds one photograph more than those three, four of the six:
/// the unused tiles of the photographs the current reference image's points
/// still read, kept from their earlier decodes, go before the used tiles of
/// the photograph the previous reference image's points read and the current
/// one's do not, so under three a photograph can be decoded twice for one
/// reference image.
#[test]
fn the_point_order_decodes_each_photograph_once_per_reference_image_that_it_sees() {
    let (scene, recon) = row_scene();
    let dir = tempfile::tempdir().unwrap();
    let paths = scene.write_photographs(dir.path());
    let one = photograph_bytes(&paths[0]);
    let budget = one * 4;
    assert!(budget < one * paths.len() as u64);

    let ordered = refit_through_cache(&scene, &recon, &paths, budget, None);
    for (i, path) in paths.iter().enumerate() {
        assert_eq!(ordered.read_phase.get(path), Some(&1), "image {i}");
        // The reference images whose points image `i` sees are its own and
        // its neighbours'.
        let sees = seen_in(i).count();
        let renders = ordered.renders.get(path).copied().unwrap_or(0);
        assert!(
            renders <= sees,
            "image {i}: {renders} decodes in the renders, {sees} reference images"
        );
    }

    let shuffled = refit_through_cache(
        &scene,
        &recon,
        &paths,
        budget,
        Some(&BulkPointOrder::identity(recon.point_count())),
    );
    assert_eq!(shuffled.answer, ordered.answer);
    assert!(
        shuffled.render_decodes() > ordered.render_decodes(),
        "shuffled {:?}, in the point order {:?}",
        shuffled.renders,
        ordered.renders
    );
}

/// A render's decode of a photograph whose tiles are under half the budget
/// keeps the whole photograph until the next decode; over half, only the
/// tiles asked for are kept as used. Under a budget of two and a half
/// photographs, the bitmaps' renders, which read each reference image's
/// tiles one after another, decode each photograph at most once, where
/// keeping only the tiles asked for decodes it again for the tiles dropped
/// before the used tiles of photographs already finished. Under a budget of
/// one photograph, the refit's fits, which read a few tiles of each of a
/// point's views in turn, decode fewer times than there are observations,
/// where keeping each decode whole would push out the other views' tiles
/// and decode at nearly every change of view.
#[test]
fn a_renders_decode_is_kept_whole_only_under_half_the_budget() {
    let (scene, recon) = row_scene();
    let dir = tempfile::tempdir().unwrap();
    let paths = scene.write_photographs(dir.path());
    let one = photograph_bytes(&paths[0]);

    let bitmaps = run_through_cache(&scene, &recon, &paths, one * 5 / 2, Kernel::Bitmaps);
    for (i, path) in paths.iter().enumerate() {
        let renders = bitmaps.renders.get(path).copied().unwrap_or(0);
        assert!(renders <= 1, "image {i}: {:?}", bitmaps.renders);
    }

    let refit = refit_through_cache(&scene, &recon, &paths, one, None);
    assert!(
        refit.render_decodes() < recon.point_set.tracks.len(),
        "{} decodes, {} observations",
        refit.render_decodes(),
        recon.point_set.tracks.len()
    );
}
