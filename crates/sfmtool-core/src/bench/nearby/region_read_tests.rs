// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The nearby-track search's grey reads, each of which makes grey only the
//! rectangle of level 0 it samples: a read gives the samples of the whole
//! photograph's blurred grey, bit for bit, at the photograph's edges too; the
//! patch read, the depth layers, the far-field sweep and the whole search give
//! the same result whichever path a read takes (every read copying its
//! rectangle into the pool, every read going through the photograph's tiles,
//! and a pool too small for some of a call's rectangles); and a photograph
//! that lacks the pixels a read needs fails each of them with the read's
//! error.

use std::fmt::Debug;
use std::sync::Arc;

use nalgebra::Point3;

use crate::bench::tests::scene::{edited, Scene};
use crate::bench::track_at_pixel::ViewCamera;
use crate::camera::image::{ImageU8, ImageU8Pyramid, PyramidLevel};
use crate::camera::sampler::RegionPool;
use crate::camera::{PhotographReadError, TiledPhotograph};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;

use super::grey::read_grey;
use super::*;

/// The tile side the scenes are cut into for the reads through tiles: a
/// quarter of the photographs' 128 px, so a read crosses several tiles.
const TILE: u32 = 32;

/// Four cameras a metre or so apart, all looking down world `+z`, the third
/// at half the others' distance from the plane, so that a read near the plane
/// takes a larger rectangle of its photograph than of the others', and a pool
/// of half the most bytes held at once copies some reads and takes others
/// through the tiles.
const CENTERS: [[f64; 3]; 4] = [
    [-0.5, -0.3, 0.0],
    [0.45, 0.25, 0.0],
    [0.1, -0.55, 2.0],
    [0.3, 0.4, 0.0],
];

/// The plane's distance along world `z`.
const PLANE: f64 = 4.0;

/// The pixel the tests query in image 0: the middle of the photograph.
const PIXEL: [f64; 2] = [64.0, 64.0];

/// The point the versions hold, on the plane.
const POINT: Point3<f64> = Point3::new(0.0, 0.0, PLANE);

fn scene() -> Scene {
    Scene::from_centers(&CENTERS, PLANE)
}

/// The distance along [`PIXEL`]'s ray in image 0 at which it meets the plane.
fn plane_distance(scene: &Scene) -> f64 {
    let views = scene.views();
    let camera = ViewCamera::new(&views[0]);
    let ray = camera.ray(PIXEL).normalize();
    (PLANE - camera.center.z) / ray.z
}

/// Run `kernel` three ways and assert it returns the same result each time:
///
/// - (a) over the scene `build` makes, through a pool that holds any read, so
///   every read copies its rectangle;
/// - (b) over the scene cut into [`TILE`] px tiles, through a pool of no
///   bytes, so every read goes through the tiles;
/// - (c) over the tiled scene, through a pool of half the most bytes (a)
///   held at once, so some reads copy their rectangles and others go through
///   the tiles.
///
/// The results are compared by their `{:#?}` text, which prints every element
/// of an array and every float in the shortest form that reads back to the
/// same bits, so `NaN`s compare equal and `-0.0` differs from `0.0`. Each pool
/// holds no more than its capacity and holds nothing once the call returns.
fn assert_the_same_on_every_path<R: Debug>(
    build: impl Fn() -> Scene,
    kernel: impl Fn(&Scene, &RegionPool) -> R,
) {
    let whole = build();
    let tiled = build().with_tile_size(TILE);

    let copying = RegionPool::with_capacity(u64::MAX);
    let a = format!("{:#?}", kernel(&whole, &copying));
    let a_stats = copying.stats();
    assert!(
        a_stats.copied_reads > 0 && a_stats.tiled_reads == 0,
        "every read copies its rectangle: {a_stats:?}"
    );
    assert_eq!(copying.held_bytes(), 0, "{a_stats:?}");
    assert!(!a.contains("..."), "the text elides: {a}");

    let on_tiles = RegionPool::with_capacity(0);
    let b = format!("{:#?}", kernel(&tiled, &on_tiles));
    let b_stats = on_tiles.stats();
    assert!(
        b_stats.tiled_reads > 0 && b_stats.copied_reads + b_stats.reused_reads == 0,
        "every read goes through the tiles: {b_stats:?}"
    );
    assert_eq!(b_stats.peak_bytes, 0);
    assert_eq!(on_tiles.held_bytes(), 0);
    assert_same_text("copied rectangles", &a, "tiles", &b);

    let small = RegionPool::with_capacity(a_stats.peak_bytes / 2);
    let c = format!("{:#?}", kernel(&tiled, &small));
    let c_stats = small.stats();
    assert!(
        c_stats.copied_reads > 0 && c_stats.tiled_reads > 0,
        "a pool of {} bytes copies some reads and takes others through the tiles: {c_stats:?}",
        small.capacity()
    );
    assert!(
        c_stats.peak_bytes <= small.capacity(),
        "{c_stats:?} over {}",
        small.capacity()
    );
    assert_eq!(small.held_bytes(), 0, "{c_stats:?}");
    assert_same_text("copied rectangles", &a, "a small pool", &c);
}

/// `a == b`, and on a difference, the first place the texts part.
fn assert_same_text(a_name: &str, a: &str, b_name: &str, b: &str) {
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

/// A read's samples are those [`sample_grey`] takes of [`blurred_grey`] of
/// the whole level 0, bit for bit, on either path: in the middle of the
/// photograph, and in each corner, where the blur reflects at two of its
/// edges. A set of positions with one off the photograph reads nothing.
#[test]
fn a_grey_read_samples_the_whole_photographs_blurred_grey() {
    let whole = scene();
    let tiled = scene().with_tile_size(TILE);
    let (w, h) = (128.0, 128.0);
    // An 11 x 11 grid 3 px apart from `(x0, y0)`.
    let grid = |x0: f64, y0: f64| -> Vec<[f64; 2]> {
        (0..121)
            .map(|k| [x0 + 3.0 * (k % 11) as f64, y0 + 3.0 * (k / 11) as f64])
            .collect()
    };
    let sets = [
        grid(48.3, 50.7),
        grid(0.5, 0.5),
        grid(w - 30.6, 0.75),
        grid(0.9, h - 30.55),
        grid(w - 30.51, h - 30.52),
    ];
    for image in 0..CENTERS.len() {
        let reference = blurred_grey(whole.pyramid(image).level(0));
        for (scene, capacity) in [(&whole, u64::MAX), (&tiled, 0)] {
            let pool = RegionPool::with_capacity(capacity);
            let views = scene.views();
            for positions in &sets {
                let got = read_grey(views[image].source, &pool, positions)
                    .expect("an in-memory photograph reads")
                    .expect("every position is on the photograph");
                let want: Vec<u32> = positions
                    .iter()
                    .map(|&[x, y]| {
                        sample_grey(&reference, x, y)
                            .expect("on the photograph")
                            .to_bits()
                    })
                    .collect();
                let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want, "image {image}, pool of {capacity}");
            }
            let mut off = sets[1].clone();
            off[60] = [0.49, 64.0];
            assert_eq!(read_grey(views[image].source, &pool, &off), Ok(None));
            assert_eq!(pool.held_bytes(), 0);
        }
    }
}

#[test]
fn the_patch_read_is_the_same_on_every_read_path() {
    assert_the_same_on_every_path(scene, |scene, pool| {
        let views = scene.views();
        let t = plane_distance(scene);
        // A patch in the middle and one by the photograph's left edge, at
        // two sizes, read at and off the plane and at infinity.
        [(PIXEL, 8.0), (PIXEL, 20.0), ([10.0, 70.0], 8.0)].map(|(pixel, radius_px)| {
            let patch = RayPatch {
                image: 0,
                pixel,
                radius_px,
            };
            read_patch_along_ray(
                &views,
                pool,
                &patch,
                &[t * 0.8, t, t * 1.25, f64::INFINITY],
                &[1, 2, 3],
                true,
            )
            .expect("every image is a view")
        })
    });
}

/// A candidate's sightings and distance range, owned, so a [`LayerCandidate`]
/// can borrow them.
struct Owned {
    sightings: Vec<(u32, [f64; 2])>,
    range: [f64; 2],
    class: RangeClass,
}

/// Two candidates, on the plane and a quarter of the way in front of it,
/// sighted in every image.
fn candidates(scene: &Scene) -> Vec<Owned> {
    let views = scene.views();
    let camera = ViewCamera::new(&views[0]);
    let t = plane_distance(scene);
    [t, t * 0.75]
        .iter()
        .map(|&t| {
            let x = Point3::from(camera.center + camera.ray(PIXEL).normalize() * t);
            let sightings: Vec<(u32, [f64; 2])> = (0..CENTERS.len() as u32)
                .map(|i| (i, scene.project(i as usize, x)))
                .collect();
            let range = distance_range(&views, 0, PIXEL, &sightings, t, 1.0).expect("valid");
            let class = classify_range(range, camera_spread(&views), &RangeOptions::default());
            Owned {
                sightings,
                range,
                class,
            }
        })
        .collect()
}

/// The depth layers of [`candidates`] at [`PIXEL`], read in `views`.
fn layers(
    scene: &Scene,
    views: &[ProjectedImage<'_>],
    pool: &RegionPool,
) -> Result<DepthLayers, DepthLayerError> {
    let owned = candidates(scene);
    let candidates: Vec<LayerCandidate<'_>> = owned
        .iter()
        .map(|c| LayerCandidate {
            source: NearbySource::Clusters,
            sightings: &c.sightings,
            distance_px: 0.0,
            max_ray_angle_deg: 10.0,
            range: c.range,
            class: c.class,
        })
        .collect();
    depth_layers(
        views,
        pool,
        0,
        PIXEL,
        &candidates,
        &DepthLayerOptions::default(),
    )
}

#[test]
fn the_depth_layers_are_the_same_on_every_read_path() {
    assert_the_same_on_every_path(scene, |scene, pool| {
        let found = layers(scene, &scene.views(), pool).expect("valid");
        assert!(found.layers.iter().all(|l| l.ranking.is_some()));
        found
    });
}

/// The far-field sweep's plane's distance along world `z`: where the widest
/// image, with its focal length of 160 px, moves the pixel 6 px from where it
/// lands at infinity.
fn far_depth() -> f64 {
    let widest = CENTERS[1..]
        .iter()
        .map(|c| 160.0 * (c[0] - CENTERS[0][0]).hypot(c[1] - CENTERS[0][1]))
        .fold(0.0, f64::max);
    widest / 6.0
}

/// The far-field sweep's capture: a plane far enough out to read at a
/// disparity of 6 px in the widest image, seen from [`CENTERS`] with the third
/// camera moved 40% of the way to the plane, so its rectangles are larger
/// than the others'.
fn far_scene() -> Scene {
    let depth = far_depth();
    let mut centers = CENTERS;
    centers[2][2] = 0.4 * depth;
    Scene::from_centers(&centers, depth)
}

#[test]
fn the_far_field_sweep_is_the_same_on_every_read_path() {
    assert_the_same_on_every_path(far_scene, |scene, pool| {
        let views = scene.views();
        // The version's one point, which only the refit's fit reads: where
        // the pixel's ray meets the plane.
        let camera = ViewCamera::new(&views[0]);
        let ray = camera.ray(PIXEL).normalize();
        let world = camera.center + ray * ((far_depth() - camera.center.z) / ray.z);
        let found = far_field_sweep(
            &edited(scene, Point3::from(world)),
            &views,
            pool,
            0,
            PIXEL,
            &FarFieldOptions::default(),
            &Progress::none(),
        )
        .expect("valid");
        assert!(!found.readings.is_empty(), "the plane reads as far");
        found
    });
}

/// The search's options: the existing points' source, then the far-field
/// sweep when `far_field_when` says so, building no bench track.
fn search_options(far_field_when: FarFieldWhen) -> NearbyTrackOptions {
    NearbyTrackOptions {
        sources: vec![NearbySource::Points],
        far_field_when,
        tracks: BenchTrackOptions {
            build: false,
            ..BenchTrackOptions::default()
        },
        ..NearbyTrackOptions::default()
    }
}

/// The pixel the searches query in image 0: a few pixels from where
/// [`POINT`] lands, so the points source finds it and the layers read the
/// patch about it.
fn query(scene: &Scene) -> [f64; 2] {
    let at = scene.project(0, POINT);
    [at[0] + 3.0, at[1] - 2.0]
}

fn search(
    scene: &Scene,
    pool: &RegionPool,
    far_field_when: FarFieldWhen,
) -> Result<NearbyTracks, NearbyTracksError> {
    find_nearby_tracks(
        &edited(scene, POINT),
        &scene.views(),
        pool,
        &NearbyTrackSources::default(),
        0,
        query(scene),
        &search_options(far_field_when),
        &Progress::none(),
    )
}

#[test]
fn the_nearby_track_search_is_the_same_on_every_read_path() {
    assert_the_same_on_every_path(scene, |scene, pool| {
        let mut found = search(scene, pool, FarFieldWhen::Always).expect("valid");
        assert!(!found.layers.is_empty());
        // The report's times are the only values that change from run to run.
        let report = &mut found.report;
        for source in &mut report.sources {
            (source.seconds, source.range_seconds) = (0.0, 0.0);
        }
        if let Some(run) = &mut report.far_field {
            (run.report.seconds, run.report.range_seconds) = (0.0, 0.0);
        }
        (report.layers_seconds, report.tracks_seconds) = (0.0, 0.0);
        found
    });
}

/// `scene`'s photograph 0 replaced by one that holds only an 8 px square in
/// the corner of its finest level and none of its coarser levels, so that a
/// read anywhere near the queried pixel asks for pixels it does not hold.
fn without_pixels(scene: &Scene) -> TiledPhotograph {
    let (width, height) = scene.views()[0].source.layout().level_size(0);
    let corner = PyramidLevel {
        image: ImageU8::new(8, 8, 1, vec![128; 64]),
        origin: [0, 0],
        full_size: [width, height],
    };
    let levels = vec![Some(corner), None, None, None];
    let pyramid = ImageU8Pyramid::from_regions(levels).expect("a valid region pyramid");
    TiledPhotograph::from_pyramid(Arc::new(pyramid)).with_tile_size(TILE)
}

/// A photograph that lacks the pixels a grey read needs fails each kernel
/// with the read's error, on either path, rather than reading as a patch
/// that does not land: the queried photograph, read for the patch's
/// template, and another photograph, read in the loop over the distances and
/// images.
#[test]
fn a_grey_read_of_pixels_a_photograph_lacks_fails_each_kernel() {
    let scene = scene();
    let broken = without_pixels(&scene);
    let t = plane_distance(&scene);
    let patch = RayPatch {
        image: 0,
        pixel: PIXEL,
        radius_px: 8.0,
    };
    let out_of_range =
        |e: &PhotographReadError| matches!(e, PhotographReadError::OutOfRange { .. });
    for capacity in [u64::MAX, 0] {
        for broken_image in [0, 1] {
            let mut views = scene.views();
            views[broken_image].source = &broken;
            let pool = RegionPool::with_capacity(capacity);
            let error = read_patch_along_ray(&views, &pool, &patch, &[t], &[1, 2, 3], false)
                .expect_err("the broken photograph is read");
            assert!(
                matches!(&error, PatchReadError::Photograph(e) if out_of_range(e)),
                "image {broken_image}, pool of {capacity}: {error}"
            );
            assert_eq!(pool.held_bytes(), 0);
        }

        // The other kernels, with photograph 1 broken: each reads it in its
        // loop over the images.
        let mut views = scene.views();
        views[1].source = &broken;
        let pool = RegionPool::with_capacity(capacity);
        let error = layers(&scene, &views, &pool).expect_err("photograph 1 is read");
        assert!(
            matches!(&error, DepthLayerError::Photograph(e) if out_of_range(e)),
            "pool of {capacity}: {error}"
        );

        let edited = edited(&scene, POINT);
        let error = far_field_sweep(
            &edited,
            &views,
            &pool,
            0,
            PIXEL,
            &FarFieldOptions::default(),
            &Progress::none(),
        )
        .expect_err("photograph 1 is read");
        assert!(
            matches!(&error, FarFieldError::Photograph(e) if out_of_range(e)),
            "pool of {capacity}: {error}"
        );

        // The search reads through the layers, and through the sweep when it
        // runs first.
        for far_field_when in [FarFieldWhen::Never, FarFieldWhen::Always] {
            let error = find_nearby_tracks(
                &edited,
                &views,
                &pool,
                &NearbyTrackSources::default(),
                0,
                query(&scene),
                &search_options(far_field_when),
                &Progress::none(),
            )
            .expect_err("photograph 1 is read");
            assert!(
                matches!(&error, NearbyTracksError::Photograph(e) if out_of_range(e)),
                "{far_field_when:?}, pool of {capacity}: {error}"
            );
        }
        assert_eq!(pool.held_bytes(), 0);
    }
}
