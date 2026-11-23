// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Mutex, OnceLock};

use super::render::{camera_rotation, dot, pixel_ray, BACKGROUND, CAMERA_CENTRES};
use super::texture::{
    octave_seed, octave_shift, perlin_fbm, perlin_noise, value_noise, FbmOptions, Surface, CUBE_A,
    CUBE_B, PERLIN_BOUND, TEXTURE_FBM, TEXTURE_PEAK,
};
use super::*;
use crate::features::sift::ImageToGray;
use crate::progress::Event;

/// The default scene, rendered once for the tests that only read it.
pub(super) fn scene() -> &'static DemoScene {
    static SCENE: OnceLock<DemoScene> = OnceLock::new();
    SCENE
        .get_or_init(|| DemoScene::render(&DemoSceneOptions::default(), &Progress::none()).unwrap())
}

/// Where a ray from `o` along `d` first meets a surface, found without the
/// slab test: each face's plane is intersected and the hit kept when it lies
/// on the face, and the nearest hit of the faces and the ground wins.
pub(super) fn reference_trace(o: [f64; 3], d: [f64; 3]) -> Option<(Surface, f64)> {
    const Z: [f64; 3] = [0.0, 0.0, 1.0];
    let neg = |v: [f64; 3]| v.map(|c| -c);
    let faces = [
        (Surface::MinusA, neg(CUBE_A), 0.5),
        (Surface::PlusA, CUBE_A, 0.5),
        (Surface::MinusB, neg(CUBE_B), 0.5),
        (Surface::PlusB, CUBE_B, 0.5),
        (Surface::Top, Z, 1.0),
        (Surface::Base, neg(Z), 0.0),
    ];
    let mut best: Option<(Surface, f64)> = None;
    let mut keep = |surface: Surface, lambda: f64| {
        if lambda > 0.0 && best.is_none_or(|(_, b)| lambda < b) {
            best = Some((surface, lambda));
        }
    };
    const SLACK: f64 = 1e-12;
    for (surface, normal, offset) in faces {
        let dn = dot(d, normal);
        if dn >= 0.0 {
            continue;
        }
        let lambda = (offset - dot(o, normal)) / dn;
        let x = [0, 1, 2].map(|k| o[k] + lambda * d[k]);
        let on_face = dot(x, CUBE_A).abs() <= 0.5 + SLACK
            && dot(x, CUBE_B).abs() <= 0.5 + SLACK
            && (-SLACK..=1.0 + SLACK).contains(&x[2]);
        if on_face {
            keep(surface, lambda);
        }
    }
    if d[2] < 0.0 {
        let lambda = -o[2] / d[2];
        let (x, y) = (o[0] + lambda * d[0], o[1] + lambda * d[1]);
        if x.abs() <= 5.0 && y.abs() <= 5.0 {
            keep(Surface::Ground, lambda);
        }
    }
    best
}

/// The world point at distance `lambda` along the ray.
fn at(o: [f64; 3], d: [f64; 3], lambda: f64) -> [f64; 3] {
    [0, 1, 2].map(|k| o[k] + lambda * d[k])
}

/// The rays of the four samples of pixel `(col, row)` of camera `image` in
/// the default scene, at offsets 0.25 and 0.75 along each axis, and the ray
/// through its centre.
fn pixel_rays(image: usize, col: u32, row: u32) -> ([[f64; 3]; 4], [f64; 3]) {
    let camera = demo_camera(&DemoSceneOptions::default());
    let rotation = camera_rotation(CAMERA_CENTRES[image]);
    let ray =
        |u: f64, v: f64| pixel_ray(&camera, &rotation, f64::from(col) + u, f64::from(row) + v);
    (
        [
            ray(0.25, 0.25),
            ray(0.75, 0.25),
            ray(0.25, 0.75),
            ray(0.75, 0.75),
        ],
        ray(0.5, 0.5),
    )
}

/// The surface all four samples of a pixel see by [`reference_trace`], with
/// their hits, or `None` for a pixel whose samples see more than one thing.
/// `Some((None, _))` is a pixel whose samples all see the background.
#[allow(clippy::type_complexity)]
fn uniform_pixel(image: usize, col: u32, row: u32) -> Option<(Option<Surface>, [[f64; 3]; 4])> {
    let (rays, _) = pixel_rays(image, col, row);
    let centre = CAMERA_CENTRES[image];
    let hits = rays.map(|d| reference_trace(centre, d).map(|(s, l)| (s, at(centre, d, l))));
    let first = hits[0].map(|(s, _)| s);
    if hits.iter().any(|h| h.map(|(s, _)| s) != first) {
        return None;
    }
    Some((first, hits.map(|h| h.map_or([0.0; 3], |(_, x)| x))))
}

fn pixel(image: &ImageU8, col: u32, row: u32) -> [u8; 3] {
    [0, 1, 2].map(|c| image.get_pixel(col, row, c))
}

/// The BT.709 luma of an RGB colour on `[0, 255]`, through the kernels'
/// grey conversion.
fn luma(rgb: [f32; 3]) -> f32 {
    ImageToGray::default().gray(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0) * 255.0
}

#[test]
fn rendered_photographs_are_the_same_on_two_runs_and_across_thread_counts() {
    // A quarter-size scene with the same field of view: the same code paths
    // at a sixteenth of the cost.
    let options = DemoSceneOptions {
        image_width: 160,
        image_height: 120,
        focal_length_px: 140.0,
        ..DemoSceneOptions::default()
    };
    let render_on = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| DemoScene::render(&options, &Progress::none()).unwrap())
    };
    let first = render_on(1);
    for threads in [1, 3, 8] {
        let again = render_on(threads);
        for (i, (a, b)) in first
            .rendered_photographs()
            .iter()
            .zip(again.rendered_photographs())
            .enumerate()
        {
            assert!(
                a.data() == b.data(),
                "image {i} differs on {threads} threads"
            );
        }
        assert_eq!(first.encoded_photographs(), again.encoded_photographs());
    }
    // The photographs hold the scene, not a blank: the sky, the ground and
    // the cube's faces each take pixels.
    let image = &first.rendered_photographs()[2];
    assert_eq!(pixel(image, 80, 2), BACKGROUND);
    assert_ne!(pixel(image, 80, 110), pixel(image, 80, 60));
}

/// The XXH128 of each JPEG of the default scene as `jpeg-encoder`'s AVX2 path
/// encodes it, which it takes on an x86 CPU with AVX2: computed on x86-64
/// Windows, and checked by CI's Linux and Windows runners.
const JPEG_XXH128_AVX2: [&str; 5] = [
    "2676ff94935814d7a9c791473689514b",
    "fb9c6357421a4f7857f052e3c7c8e288",
    "7b8a4e749c4706b3ad2954c9d40d7ed7",
    "60b576f9c973c4a0b7f09e600d991f9b",
    "06738cd2d67578ffa0b9e95ed0248f50",
];

/// The same as `jpeg-encoder`'s scalar path encodes it, which it takes on
/// every other CPU: computed on x86-64 Windows with the crate's `simd`
/// feature off, and checked by CI's macOS arm64 runner. Its DCT gives other
/// coefficients than the AVX2 one for some blocks, so these differ from
/// [`JPEG_XXH128_AVX2`].
const JPEG_XXH128_SCALAR: [&str; 5] = [
    "9ffaa8da80968aa67ee7fea0b81cc22c",
    "80e84e3aeb6b5037099f2f97048103bd",
    "02229b128c970781c69eaa6cb265e079",
    "7e1b45aa571c988b43828f05506bc85a",
    "e9f00c15962f49793383dad543609e20",
];

/// Whether `jpeg-encoder` takes its AVX2 path here, by the test it makes.
pub(super) fn encoder_takes_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[test]
fn the_jpegs_match_the_hashes_stored_for_the_encoders_path() {
    let hashes: Vec<String> = scene()
        .encoded_photographs()
        .iter()
        .map(|bytes| format!("{:032x}", xxhash_rust::xxh3::xxh3_128(bytes)))
        .collect();
    let expected = if encoder_takes_avx2() {
        JPEG_XXH128_AVX2
    } else {
        JPEG_XXH128_SCALAR
    };
    assert_eq!(hashes, expected);
}

#[test]
fn photographs_are_the_decoded_jpegs_and_differ_a_little_from_the_render() {
    let scene = scene();
    assert_eq!(scene.photographs().len(), 5);
    for (i, ((decoded, encoded), rendered)) in scene
        .photographs()
        .iter()
        .zip(scene.encoded_photographs())
        .zip(scene.rendered_photographs())
        .enumerate()
    {
        assert_eq!(&encoded[..3], [0xFF, 0xD8, 0xFF], "image {i} is a JPEG");
        let again = ImageU8::decode_rgb(encoded).unwrap();
        assert!(decoded.data() == again.data(), "image {i}");
        let (sum, max) =
            decoded
                .data()
                .iter()
                .zip(rendered.data())
                .fold((0u64, 0u8), |(sum, max), (a, b)| {
                    let d = a.abs_diff(*b);
                    (sum + u64::from(d), max.max(d))
                });
        let mean = sum as f64 / decoded.data().len() as f64;
        assert!(
            (0.5..=4.0).contains(&mean) && max <= 60,
            "image {i}: mean JPEG error {mean}, max {max}"
        );
    }
}

#[test]
fn a_facing_pixel_is_the_texture_at_its_centre_ray_hit_within_the_blur() {
    // Camera 0 sees the face -a 12 degrees off its normal. With one sample a
    // pixel its only ray is the centre one, so the pixel is the texture
    // there up to the rounding to 8 bits; with the default four it is that
    // to within the supersampling's blur.
    let (image, seed) = (0, DemoSceneOptions::default().seed);
    let one_ray = DemoScene::render(
        &DemoSceneOptions {
            supersampling: 1,
            ..DemoSceneOptions::default()
        },
        &Progress::none(),
    )
    .unwrap();
    let sharp = &one_ray.rendered_photographs()[image];
    let blurred = &scene().rendered_photographs()[image];
    let centre = CAMERA_CENTRES[image];
    let (mut count, mut total_error) = (0usize, 0f64);
    for row in 0..480 {
        for col in 0..640 {
            // Pixels wholly on the face, so its centre ray and its four
            // samples all meet it.
            let Some((Some(Surface::MinusA), samples)) = uniform_pixel(image, col, row) else {
                continue;
            };
            let (_, d) = pixel_rays(image, col, row);
            let Some((Surface::MinusA, lambda)) = reference_trace(centre, d) else {
                continue;
            };
            let expected = Surface::MinusA.colour(at(centre, d, lambda), seed);
            let (sharp, blurred) = (pixel(sharp, col, row), pixel(blurred, col, row));
            for c in 0..3 {
                assert!(
                    (f32::from(sharp[c]) - expected[c]).abs() <= 0.5 + 1e-3,
                    "pixel ({col}, {row}) channel {c}: {} at one ray, the texture {}",
                    sharp[c],
                    expected[c]
                );
                // The blur: the four-sample pixel is their mean, so it is no
                // further from the centre's colour than the farthest sample.
                let blur = samples
                    .iter()
                    .map(|x| (Surface::MinusA.colour(*x, seed)[c] - expected[c]).abs())
                    .fold(0f32, f32::max);
                let error = (f32::from(blurred[c]) - expected[c]).abs();
                assert!(
                    error <= blur + 0.5 + 1e-3,
                    "pixel ({col}, {row}) channel {c}: {} at four rays, the texture {}, \
                     blur {blur}",
                    blurred[c],
                    expected[c]
                );
                total_error += f64::from(error);
            }
            count += 1;
        }
    }
    assert!(count > 10_000, "{count} facing pixels");
    // Over the face the blur mostly averages out: the four samples sit
    // symmetrically about the centre, so the texture's slope cancels and its
    // curvature and the rounding are left, 0.74 levels on average.
    let mean = total_error / (3 * count) as f64;
    assert!(mean < 1.0, "mean error {mean} levels");
}

#[test]
fn each_surface_grey_is_its_tint_luma_times_the_noise_factor() {
    // The tints' lumas, as the draft's table gives them, within 5% of each
    // other.
    let lumas = [
        (Surface::Ground, 183.8),
        (Surface::Top, 185.8),
        (Surface::MinusA, 178.4),
        (Surface::MinusB, 178.7),
        (Surface::PlusA, 181.4),
        (Surface::PlusB, 179.4),
    ];
    for (surface, stated) in lumas {
        let l = luma(surface.tint().map(f32::from));
        assert!((l - stated).abs() < 0.05, "{surface:?}: {l}");
        assert!((l / 180.0 - 1.0).abs() < 0.05, "{surface:?}: {l}");
    }

    let seed = DemoSceneOptions::default().seed;
    let mut seen = std::collections::BTreeMap::new();
    for (image, photograph) in scene().rendered_photographs().iter().enumerate() {
        // Every third pixel each way: each surface's pixels number thousands.
        for row in (0..480).step_by(3) {
            for col in (0..640).step_by(3) {
                let Some((Some(surface), samples)) = uniform_pixel(image, col, row) else {
                    continue;
                };
                let factor = samples
                    .iter()
                    .map(|x| surface.noise_factor(*x, seed))
                    .sum::<f32>()
                    / 4.0;
                let tint_luma = luma(surface.tint().map(f32::from));
                let got = luma(pixel(photograph, col, row).map(f32::from));
                // The grey is linear in the channels with weights summing to
                // 1, so rounding each channel moves it by at most half a level.
                assert!(
                    (got - tint_luma * factor).abs() <= 0.5 + 1e-3,
                    "image {image} pixel ({col}, {row}) on {surface:?}: grey {got}, \
                     expected {tint_luma} x {factor}"
                );
                *seen.entry(surface as u8).or_insert(0usize) += 1;
            }
        }
    }
    for surface in [
        Surface::Ground,
        Surface::Top,
        Surface::MinusA,
        Surface::MinusB,
    ] {
        let n = seen.get(&(surface as u8)).copied().unwrap_or(0);
        assert!(n > 500, "{surface:?} has {n} pixels");
    }
}

#[test]
fn the_cube_silhouette_is_where_the_slab_test_puts_it() {
    let is_cube = |s: Option<Surface>| s.is_some_and(|s| s != Surface::Ground);
    for (image, photograph) in scene().rendered_photographs().iter().enumerate() {
        let mut kinds = vec![None; 640 * 480];
        for row in 0..480 {
            for col in 0..640 {
                let Some((surface, _)) = uniform_pixel(image, col, row) else {
                    continue;
                };
                let got = pixel(photograph, col, row);
                match surface {
                    None => assert_eq!(got, BACKGROUND, "image {image} ({col}, {row})"),
                    Some(surface) => {
                        // The colour is the surface's tint times one factor:
                        // what is left after the best such fit is the
                        // rounding alone, under half a level in each channel
                        // and so under sqrt(3) / 2 in all three.
                        let tint = surface.tint().map(f32::from);
                        let got = got.map(f32::from);
                        let k = (0..3).map(|c| got[c] * tint[c]).sum::<f32>()
                            / (0..3).map(|c| tint[c] * tint[c]).sum::<f32>();
                        for c in 0..3 {
                            assert!(
                                (got[c] - k * tint[c]).abs() <= 0.87,
                                "image {image} ({col}, {row}): {got:?} is not the \
                                 {surface:?} tint {tint:?} times a factor"
                            );
                        }
                    }
                }
                kinds[(row * 640 + col) as usize] = Some(is_cube(surface));
            }
        }
        // The check reaches the silhouette: pixels wholly on the cube that
        // touch pixels wholly off it.
        let mut edge = 0;
        for row in 1..479u32 {
            for col in 1..639u32 {
                if kinds[(row * 640 + col) as usize] != Some(true) {
                    continue;
                }
                let neighbours = [
                    (col - 1, row),
                    (col + 1, row),
                    (col, row - 1),
                    (col, row + 1),
                ];
                if neighbours
                    .iter()
                    .any(|&(c, r)| kinds[(r * 640 + c) as usize] == Some(false))
                {
                    edge += 1;
                }
            }
        }
        assert!(
            edge > 100,
            "image {image}: {edge} silhouette pixels checked"
        );
    }
}

#[test]
fn each_camera_looks_at_the_cube_centre_upright_from_the_stated_arc() {
    let camera = demo_camera(&DemoSceneOptions::default());
    let azimuths = [-55.0f64, -27.5, 0.0, 27.5, 55.0];
    for (i, centre) in CAMERA_CENTRES.iter().enumerate() {
        let r = camera_rotation(*centre);
        for a in 0..3 {
            for b in 0..3 {
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!((dot(r[a], r[b]) - expected).abs() < 1e-12);
            }
        }
        // Right is horizontal and up points up.
        assert!(r[0][2].abs() < 1e-12 && r[1][2] > 0.0);
        // The cube's centre is on the optical axis, in front.
        let rel = [0.0 - centre[0], 0.0 - centre[1], 0.5 - centre[2]];
        let p = [dot(r[0], rel), dot(r[1], rel), dot(r[2], rel)];
        assert!(p[0].abs() < 1e-12 && p[1].abs() < 1e-12 && p[2] < 0.0);
        let (u, v) = camera.ray_to_pixel(p).unwrap();
        assert!((u - 320.0).abs() < 1e-9 && (v - 240.0).abs() < 1e-9);
        // 4 m from the z axis, at the stated azimuth from -y.
        let radius = centre[0].hypot(centre[1]);
        assert!((radius - 4.0).abs() < 1e-3, "camera {i}: radius {radius}");
        let azimuth = centre[0].atan2(-centre[1]).to_degrees();
        assert!(
            (azimuth - azimuths[i]).abs() < 0.02,
            "camera {i}: {azimuth}"
        );
        assert_eq!(centre[2], if i % 2 == 0 { 1.2 } else { 3.0 });
    }
}

#[test]
fn the_slab_test_names_the_face_a_ray_enters() {
    // From well outside each face, straight at its centre.
    let centre = [0.0, 0.0, 0.5];
    for (surface, normal) in [
        (Surface::MinusA, CUBE_A.map(|c| -c)),
        (Surface::PlusA, CUBE_A),
        (Surface::MinusB, CUBE_B.map(|c| -c)),
        (Surface::PlusB, CUBE_B),
        (Surface::Top, [0.0, 0.0, 1.0]),
    ] {
        let face = [0, 1, 2].map(|k| centre[k] + 0.5 * normal[k]);
        let origin = [0, 1, 2].map(|k| face[k] + 3.0 * normal[k]);
        let direction = normal.map(|c| -c);
        let (hit, lambda) = render::cube_hit(origin, direction).unwrap();
        assert_eq!(hit, surface);
        assert!((lambda - 3.0).abs() < 1e-12);
    }
    // A ray that passes beside the cube, and one starting inside it.
    assert!(render::cube_hit([2.0, 0.0, 0.5], [0.0, 1.0, 0.0]).is_none());
    assert!(render::cube_hit([0.0, 0.0, 0.5], [1.0, 0.0, 0.0]).is_none());
}

#[test]
fn value_noise_is_continuous_bounded_and_picked_by_its_seed() {
    let spacing = 0.02;
    let mut differs = 0;
    for k in 0..200 {
        let x = -1.0 + 0.0137 * f64::from(k);
        let y = 0.5 - 0.0071 * f64::from(k);
        let n = value_noise(x, y, spacing, 3);
        assert!((-1.0..=1.0).contains(&n));
        // Across a lattice line the value does not jump.
        let edge = (x / spacing).floor() * spacing;
        let left = value_noise(edge - 1e-9, y, spacing, 3);
        let right = value_noise(edge + 1e-9, y, spacing, 3);
        assert!((left - right).abs() < 1e-6, "{left} {right} at x = {edge}");
        if (value_noise(x, y, spacing, 4) - n).abs() > 0.05 {
            differs += 1;
        }
    }
    assert!(
        differs > 150,
        "another seed changed {differs} of 200 values"
    );
}

#[test]
fn for_point_count_scales_both_spacings() {
    let default = DemoSceneOptions::default();
    assert_eq!(
        DemoSceneOptions::for_point_count(DemoSceneOptions::DEFAULT_POINT_COUNT).unwrap(),
        default
    );
    let quarter =
        DemoSceneOptions::for_point_count(DemoSceneOptions::DEFAULT_POINT_COUNT / 4).unwrap();
    assert_eq!(
        quarter.ground_point_spacing,
        2.0 * default.ground_point_spacing
    );
    assert_eq!(quarter.face_point_spacing, 2.0 * default.face_point_spacing);
    assert_eq!(quarter.image_width, default.image_width);
    assert!(matches!(
        DemoSceneOptions::for_point_count(0),
        Err(DemoSceneError::Options(m)) if m.contains("point_count")
    ));
}

#[test]
fn render_refuses_each_option_out_of_range() {
    let d = DemoSceneOptions::default;
    let cases: Vec<(&str, DemoSceneOptions)> = vec![
        (
            "image_width",
            DemoSceneOptions {
                image_width: 0,
                ..d()
            },
        ),
        (
            "image_height",
            DemoSceneOptions {
                image_height: 65536,
                ..d()
            },
        ),
        (
            "focal_length_px",
            DemoSceneOptions {
                focal_length_px: f64::NAN,
                ..d()
            },
        ),
        (
            "focal_length_px",
            DemoSceneOptions {
                focal_length_px: 0.0,
                ..d()
            },
        ),
        (
            "supersampling",
            DemoSceneOptions {
                supersampling: 0,
                ..d()
            },
        ),
        (
            "ground_point_spacing",
            DemoSceneOptions {
                ground_point_spacing: -0.25,
                ..d()
            },
        ),
        (
            "face_point_spacing",
            DemoSceneOptions {
                face_point_spacing: f64::INFINITY,
                ..d()
            },
        ),
        (
            "patch_half_extent",
            DemoSceneOptions {
                patch_half_extent: 0.0,
                ..d()
            },
        ),
        (
            "patch_bitmap_resolution",
            DemoSceneOptions {
                patch_bitmap_resolution: 0,
                ..d()
            },
        ),
        (
            "max_obliquity_deg",
            DemoSceneOptions {
                max_obliquity_deg: 90.5,
                ..d()
            },
        ),
        (
            "max_obliquity_deg",
            DemoSceneOptions {
                max_obliquity_deg: f64::NAN,
                ..d()
            },
        ),
    ];
    for (name, options) in cases {
        match DemoScene::render(&options, &Progress::none()) {
            Err(DemoSceneError::Options(message)) => {
                assert!(message.starts_with(name), "{name}: {message}")
            }
            Err(e) => panic!("{name}: {e}"),
            Ok(_) => panic!("{name}: rendered"),
        }
    }
}

#[test]
fn render_counts_rows_then_images_and_stops_when_cancelled() {
    let options = DemoSceneOptions {
        image_width: 32,
        image_height: 24,
        focal_length_px: 28.0,
        ..DemoSceneOptions::default()
    };
    let events = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| match event {
        Event::Enter { phase, .. } => events.lock().unwrap().push(format!("enter {phase}")),
        Event::Count {
            done,
            total: Some(total),
            unit,
        } if done == total => events.lock().unwrap().push(format!("{done} {unit}")),
        _ => {}
    };
    DemoScene::render(&options, &Progress::to(&sink)).unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["enter render", "120 rows", "enter encode", "5 images"]
    );

    let cancelled = std::sync::atomic::AtomicBool::new(true);
    assert!(matches!(
        DemoScene::render(&options, &Progress::none().cancelled_by(&cancelled)),
        Err(DemoSceneError::Cancelled)
    ));
}

#[test]
fn perlin_noise_is_zero_on_the_lattice_continuous_and_bounded() {
    // 0 at every lattice node, negative indices among them; a spacing of a
    // power of 2 puts the nodes exactly where the coordinates say.
    for (i, j) in [(0, 0), (3, -7), (-12, 5), (40, 40)] {
        let (x, y) = (f64::from(i) * 0.125, f64::from(j) * 0.125);
        assert_eq!(perlin_noise(x, y, 0.125, 9), 0.0, "node ({i}, {j})");
    }
    let spacing = 0.1;
    // Over 200 × 200 cells, the noise at a cell's centre reaches the bound
    // where all four gradients point at the centre, and never passes it.
    let mut largest = 0f64;
    for i in -100..100 {
        for j in -100..100 {
            let x = (f64::from(i) + 0.5) * spacing;
            let y = (f64::from(j) + 0.5) * spacing;
            largest = largest.max(perlin_noise(x, y, spacing, 9).abs());
        }
    }
    assert!(
        (largest - PERLIN_BOUND).abs() < 1e-12,
        "largest {largest} at a cell centre"
    );
    let mut differs = 0;
    for k in 0..400 {
        let x = -1.0 + 0.0137 * f64::from(k);
        let y = 0.5 - 0.0071 * f64::from(k);
        let n = perlin_noise(x, y, spacing, 3);
        assert!(n.abs() <= PERLIN_BOUND, "{n} at ({x}, {y})");
        // Across a lattice line neither the value nor its slope jumps.
        let edge = (x / spacing).floor() * spacing;
        let h = 1e-6;
        let at = |x: f64| perlin_noise(x, y, spacing, 3);
        assert!((at(edge - h) - at(edge + h)).abs() < 1e-4);
        let slope_left = (at(edge - h) - at(edge - 2.0 * h)) / h;
        let slope_right = (at(edge + 2.0 * h) - at(edge + h)) / h;
        assert!(
            (slope_left - slope_right).abs() < 1e-3,
            "slope {slope_left} then {slope_right} at x = {edge}"
        );
        if (perlin_noise(x, y, spacing, 4) - n).abs() > 0.05 {
            differs += 1;
        }
    }
    assert!(
        differs > 300,
        "another seed changed {differs} of 400 values"
    );
}

#[test]
fn perlin_noise_and_its_fbm_give_the_stored_bits() {
    // Computed on x86-64 Windows. The functions use only `+`, `−`, `×`, `÷`
    // and `floor`, so every platform gives these bits.
    let fbm = FbmOptions {
        spacing: 0.3,
        octaves: 4,
        lacunarity: 2.0,
        gain: 0.6,
    };
    let values = || {
        [
            perlin_noise(0.37, -1.23, 0.1, 5),
            perlin_noise(-4.019, 2.5, 0.02, 0),
            perlin_fbm(0.37, -1.23, &fbm, 5),
            perlin_fbm(-4.019, 2.5, &TEXTURE_FBM, 0),
        ]
        .map(f64::to_bits)
    };
    let stored = [
        0xbfc9_78f3_8ada_0152,
        0xbfa2_7ab2_d12a_398e,
        0x3fcf_25d6_cd8d_ae52,
        0x3fa0_e9d8_bd56_163e,
    ];
    assert_eq!(values(), stored);
}

#[test]
fn perlin_fbm_adds_octaves_at_falling_amplitudes_and_is_bounded() {
    let options = |octaves| FbmOptions {
        spacing: 0.3,
        octaves,
        lacunarity: 2.0,
        gain: 0.6,
    };
    assert!((options(3).amplitude_sum() - (1.0 + 0.6 + 0.36)).abs() < 1e-15);
    for k in 0..100 {
        let x = -2.0 + 0.0413 * f64::from(k);
        let y = 1.0 - 0.0297 * f64::from(k);
        // One octave is the Perlin noise of the first octave's seed.
        let one = perlin_fbm(x, y, &options(1), 7);
        assert_eq!(one, perlin_noise(x, y, 0.3, octave_seed(7, 0)));
        // Each further octave adds the noise at half the last spacing, on its
        // shifted lattice with its own seed, at 0.6 of the last amplitude.
        for n in 2..=4u32 {
            let (with, without) = (options(n), options(n - 1));
            let added = perlin_fbm(x, y, &with, 7) * with.amplitude_sum()
                - perlin_fbm(x, y, &without, 7) * without.amplitude_sum();
            let spacing = 0.3 / 2f64.powi(n as i32 - 1);
            let shift = octave_shift(n - 1) * spacing;
            let octave = perlin_noise(x + shift, y + shift, spacing, octave_seed(7, n - 1));
            assert!(
                (added - 0.6f64.powi(n as i32 - 1) * octave).abs() < 1e-12,
                "octave {n} at ({x}, {y})"
            );
        }
        assert!(perlin_fbm(x, y, &options(4), 7).abs() <= PERLIN_BOUND);
    }
    // The octaves draw different gradients and sit on shifted lattices.
    assert_ne!(octave_seed(7, 0), octave_seed(7, 1));
    assert_ne!(octave_shift(1), octave_shift(2));
    assert!((0.0..1.0).contains(&octave_shift(3)));
}

#[test]
fn the_texture_fills_most_of_its_grey_range_and_never_leaves_it() {
    // The fBm's largest magnitude over every surface of four scenes stays
    // under the peak the texture maps to the range's ends, and comes within
    // 10% of it, so the range is neither passed nor left unused.
    let mut largest = 0f64;
    for seed in 0..4u64 {
        for surface in Surface::ALL {
            let (origin, extent) = if surface == Surface::Ground {
                (-5.0, 10.0)
            } else {
                (0.0, 1.0)
            };
            for a in 0..=150 {
                for b in 0..=150 {
                    let s = origin + extent * f64::from(a) / 150.0;
                    let t = origin + extent * f64::from(b) / 150.0;
                    let seed = seed * 7 + surface as u64;
                    largest = largest.max(perlin_fbm(s, t, &TEXTURE_FBM, seed).abs());
                }
            }
        }
    }
    assert!(
        (0.9..1.0).contains(&(largest / TEXTURE_PEAK)),
        "largest {largest}, peak {TEXTURE_PEAK}"
    );
}

/// The contrast of the grey in each 24 × 24 pixel window of the default
/// scene's renders, the side of a patch bitmap, that lies wholly on one
/// surface, by surface: the window's standard deviation, and its pixel-scale
/// contrast, the RMS difference between horizontally or vertically
/// neighbouring pixels over `√2`, which equals the standard deviation for
/// independent pixels.
fn window_contrasts() -> std::collections::BTreeMap<u8, Vec<(f64, f64)>> {
    const SIDE: u32 = 24;
    let mut contrasts = std::collections::BTreeMap::new();
    for (image, photograph) in scene().rendered_photographs().iter().enumerate() {
        for row0 in (0..=480 - SIDE).step_by(SIDE as usize) {
            'window: for col0 in (0..=640 - SIDE).step_by(SIDE as usize) {
                let mut surface = None;
                let mut grey = Vec::with_capacity((SIDE * SIDE) as usize);
                for row in row0..row0 + SIDE {
                    for col in col0..col0 + SIDE {
                        let Some((Some(s), _)) = uniform_pixel(image, col, row) else {
                            continue 'window;
                        };
                        if *surface.get_or_insert(s) != s {
                            continue 'window;
                        }
                        grey.push(f64::from(luma(pixel(photograph, col, row).map(f32::from))));
                    }
                }
                let n = grey.len() as f64;
                let mean = grey.iter().sum::<f64>() / n;
                let sd = (grey.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n).sqrt();
                let side = SIDE as usize;
                let (mut squares, mut pairs) = (0f64, 0f64);
                for r in 0..side {
                    for c in 0..side - 1 {
                        squares += (grey[r * side + c + 1] - grey[r * side + c]).powi(2);
                        squares += (grey[(c + 1) * side + r] - grey[c * side + r]).powi(2);
                        pairs += 2.0;
                    }
                }
                let fine = (squares / pairs / 2.0).sqrt();
                contrasts
                    .entry(surface.unwrap() as u8)
                    .or_insert_with(Vec::new)
                    .push((sd, fine));
            }
        }
    }
    contrasts
}

#[test]
fn every_patch_sized_window_has_texture_to_match() {
    // Measured on the default scene: the 10th percentile of the windows'
    // standard deviations is 9.0 to 11.6 levels across the four surfaces
    // seen, and of their pixel-scale contrasts 2.9 to 4.6. Without the
    // finest octave the pixel-scale contrast's falls to 2.2 on the ground and
    // both side faces.
    let contrasts = window_contrasts();
    for surface in [
        Surface::Ground,
        Surface::Top,
        Surface::MinusA,
        Surface::MinusB,
    ] {
        let windows = &contrasts[&(surface as u8)];
        assert!(
            windows.len() >= 10,
            "{surface:?}: {} windows",
            windows.len()
        );
        let tenth = |key: fn(&(f64, f64)) -> f64| {
            let values: Vec<f64> = windows.iter().map(key).collect();
            crate::numeric::quantile(&values, 0.1)
        };
        let (sd, fine) = (tenth(|w| w.0), tenth(|w| w.1));
        assert!(sd >= 8.0, "{surface:?}: standard deviation {sd}");
        assert!(fine >= 2.6, "{surface:?}: pixel-scale contrast {fine}");
    }
}
