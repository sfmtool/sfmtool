// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-image release: each image's focal scale and principal-point offset
//! solved under a prior centred on its camera's mean, the widths derived from
//! the solve, and the re-centring after every round.

use super::*;

/// A standard normal draw from two deterministic uniforms (Box–Muller).
fn gaussian(i: usize, salt: u64) -> f64 {
    let u1 = 0.5 * (jitter(i, salt) + 1.0) * 0.9998 + 0.0001;
    let u2 = 0.5 * (jitter(i, salt ^ 0x5555) + 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// A 60° `SIMPLE_PINHOLE` of 640×480 (short side 480) at principal point
/// `(cx, cy)`.
fn wide_pinhole(cx: f64, cy: f64) -> CameraIntrinsics {
    pinhole_with_field(60.0, cx, cy)
}

/// A scene whose points fill a 60° field, seen from `n_img` images on a
/// shallow arc, each image's observations made through `truth` with the
/// image's planted adjustment applied and `noise_px` of Gaussian noise per
/// axis added. Image `i` sees at most `per_image(i)` points, the first ones
/// that land inside it.
fn per_image_scene(
    truth: &CameraIntrinsics,
    n_img: usize,
    n_pt: usize,
    planted: &[IntrinsicAdjustment],
    noise_px: f64,
    noise_salt: u64,
    per_image: impl Fn(usize) -> usize,
) -> Scene {
    let f = truth.focal_lengths().0;
    let (half_w, half_h) = (
        8.0 * f64::from(truth.width) / 2.0 / f,
        8.0 * f64::from(truth.height) / 2.0 / f,
    );
    // The images look from a cap of directions, spread in azimuth and
    // elevation, each at its own point near the origin and rolled about its
    // axis: a general motion. Images whose optical axes all meet in one point,
    // or that turn about parallel axes along one arc, leave the per-image
    // intrinsics a family of exact solutions that only the prior chooses
    // among, which a test of recovery cannot read.
    let mut quats = Vec::new();
    let mut trans = Vec::new();
    for i in 0..n_img {
        let az = 0.5 * jitter(i, 12);
        let el = 0.4 * jitter(i, 13);
        let roll = 0.3 * jitter(i, 14);
        let center = 8.0 * Vector3::new(az.sin() * el.cos(), el.sin(), az.cos() * el.cos());
        let aim = Vector3::new(
            1.2 * jitter(i, 15),
            1.0 * jitter(i, 16),
            1.2 * jitter(i, 17),
        );
        let r = UnitQuaternion::face_towards(&(center - aim), &Vector3::y()).inverse();
        let r = UnitQuaternion::from_scaled_axis(Vector3::z() * roll) * r;
        quats.push(r);
        trans.push(-(r * center));
    }
    let points: Vec<[f64; 3]> = (0..n_pt)
        .map(|p| {
            [
                0.85 * half_w * jitter(p, 1),
                0.85 * half_h * jitter(p, 2),
                0.5 * half_w * jitter(p, 3),
            ]
        })
        .collect();
    let mut uv = Vec::new();
    let mut obs_img = Vec::new();
    let mut obs_pt = Vec::new();
    let mut seen = vec![0usize; n_img];
    let mut k = 0usize;
    for (p, x) in points.iter().enumerate() {
        for i in 0..n_img {
            if seen[i] >= per_image(i) {
                continue;
            }
            let cam = truth.with_adjustment(&planted[i], None);
            let c = quats[i] * Vector3::new(x[0], x[1], x[2]) + trans[i];
            let Some((u, v)) = cam.ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            if !(0.0..f64::from(cam.width)).contains(&u)
                || !(0.0..f64::from(cam.height)).contains(&v)
            {
                continue;
            }
            uv.push([
                u + noise_px * gaussian(k, noise_salt),
                v + noise_px * gaussian(k, noise_salt + 1),
            ]);
            obs_img.push(i as u32);
            obs_pt.push(p as u32);
            seen[i] += 1;
            k += 1;
        }
    }
    Scene {
        cam: truth.clone(),
        quats,
        trans,
        points,
        uv,
        obs_img,
        obs_pt,
    }
}

/// Perturb every pose a little, so the solve has something to do.
fn perturb_poses(s: &mut Scene) {
    for i in 0..s.quats.len() {
        let d = Vector3::new(jitter(i, 31), jitter(i, 32), jitter(i, 33)) * 2e-3;
        s.quats[i] = UnitQuaternion::from_scaled_axis(d) * s.quats[i];
        s.trans[i] += Vector3::new(jitter(i, 34), jitter(i, 35), jitter(i, 36)) * 0.01;
    }
}

/// Run the per-image release over `s` from the camera `start`, every image
/// starting at `start_adj` (zero where empty), with `releases` for the one
/// camera.
fn run_image(
    s: &mut Scene,
    start: &CameraIntrinsics,
    start_adj: &[IntrinsicAdjustment],
    release: CameraRelease,
    prior: Option<&ImagePriorOptions>,
    schedule: &[BaSchedule],
) -> BundleAdjustment {
    let n_img = s.quats.len();
    let adjustments: Vec<IntrinsicAdjustment> = if start_adj.is_empty() {
        vec![IntrinsicAdjustment::default(); n_img]
    } else {
        start_adj.to_vec()
    };
    let releases = [release];
    bundle_adjust(
        &BaCameras {
            cameras: std::slice::from_ref(start),
            image_camera: Cow::Owned(vec![0; n_img]),
            releases: Some(&releases),
            image_adjustments: Some(Cow::Owned(adjustments)),
            rolling_shutter_scanlines: None,
        },
        &mut s.quats,
        &mut s.trans,
        &mut s.points,
        &s.uv,
        &s.obs_img,
        &s.obs_pt,
        None,
        None,
        FreePointPolicy::default(),
        None,
        DEFAULT_PROTECTED_LOSS_SCALE,
        true,
        true,
        true,
        true,
        true,
        true,
        prior,
        schedule,
        60,
        2,
        12,
        &Progress::none(),
    )
}

/// The release of each image's principal point alone.
const IMAGE_PP: CameraRelease = CameraRelease {
    image_principal_point: true,
    ..CameraRelease::HELD
};

/// The release of each image's focal scale alone.
const IMAGE_F: CameraRelease = CameraRelease {
    image_focal: true,
    ..CameraRelease::HELD
};

/// The release of each image's focal scale and principal point.
const IMAGE_BOTH: CameraRelease = CameraRelease {
    image_focal: true,
    image_principal_point: true,
    ..CameraRelease::HELD
};

#[test]
fn image_adjustment_columns_match_a_central_difference() {
    // One image at the identity pose, read through its camera with an
    // adjustment applied, and one point off the axis. The log focal scale's
    // column is checked on the models whose focal the kernel releases, the
    // offset's on every model.
    let point = Vector3::new(0.3, -0.2, -5.0);
    let adjustment = IntrinsicAdjustment {
        focal: 0.012,
        principal_point: [0.008, -0.006],
        radial_bspline: [0.0; 2],
        rolling_shutter: [0.0; 3],
    };
    let h = 1e-6;
    // Every model with the adjustment above, and an `SFMTOOL_PINHOLE` image
    // that also holds a radial correction and rolling-shutter rates, on rows
    // and on columns, under which the log focal scale's column gains
    // `f_i·t·ṗ`.
    let corrected = IntrinsicAdjustment {
        radial_bspline: [0.02, -0.03],
        rolling_shutter: [2e-2, -1.5e-2, 1e-2],
        ..adjustment
    };
    let spline_pinhole = sfmtool_pinhole(250.0, &[0.002, 0.007, 0.014, 0.024, 0.036, 0.05]);
    let cases = crate::camera::intrinsics::tests::all_cameras()
        .into_iter()
        .map(|cam| (cam, adjustment, None))
        .chain(
            [
                RollingShutterScanlines::Rows,
                RollingShutterScanlines::Columns,
            ]
            .map(|s| (spline_pinhole.clone(), corrected, Some(s))),
        );
    for (cam, adjustment, scanlines) in cases {
        let name = cam.model_name();
        let image = cam.with_adjustment(&adjustment, scanlines);
        let Some((u0, v0)) = image.ray_to_pixel([point.x, point.y, point.z]) else {
            continue;
        };
        let (cx, cy) = image.principal_point();
        let focal = focal_is_releasable(&cam.model);
        let lens = LinLens {
            cam: &image,
            analytic: cam.model.supports_pixel_jacobian(),
            cx,
            cy,
            f: cam.focal_lengths().0,
            f_image: image.focal_lengths().0,
            m: f64::from(cam.width.min(cam.height)),
            opt_f: false,
            opt_k1: false,
            opt_bspline: false,
            opt_pp: false,
            opt_image_f: focal,
            opt_image_pp: true,
            n_coeffs: 0,
            d_max: 0.0,
            radial: SplineRadial::IncidenceAngle,
            slot0: 0,
        };
        let st = LinState {
            lenses: std::slice::from_ref(&lens),
            ci_cam: &[0],
            q: &[UnitQuaternion::identity()],
            t: &[Vector3::zeros()],
            xp: &[point],
            bases: &[(Vector3::zeros(), Vector3::zeros())],
            cp_dir: &[false],
            cp_held: &[false],
            cp_origin: &[None],
            cp_inverse: &[],
            inv_depth: &[],
            uv: &[[u0 + 0.5, v0 - 0.5]],
            n_shared: lens_block_width(0),
            img_cols: 9,
        };
        let b = observation_blocks::<BASE_IMG_CAM_COLS>(&st, 0, 0, 0, 1e18);
        let adj_c = BASE_IMG_CAM_COLS - 3;
        for k in 0..3 {
            assert_eq!(
                b.idx[adj_c + k],
                lens_block_width(0) + IMG_S_SLOT + k,
                "{name}: adjustment column {k}'s slot"
            );
        }
        for k in 0..3 {
            if k == 0 && !focal {
                assert_eq!(
                    b.cam_j[(0, adj_c)],
                    0.0,
                    "{name}: an unreleased focal scale"
                );
                continue;
            }
            let moved = |d: f64| {
                let mut a = adjustment;
                match k {
                    // The focal's parameter is the log focal scale
                    // `t = ln(1 + s)`, so the difference steps `t`.
                    0 => a.focal = (a.focal.ln_1p() + d).exp_m1(),
                    1 => a.principal_point[0] += d,
                    _ => a.principal_point[1] += d,
                }
                cam.with_adjustment(&a, scanlines)
                    .ray_to_pixel([point.x, point.y, point.z])
                    .expect("in domain")
            };
            let (up, vp) = moved(h);
            let (um, vm) = moved(-h);
            let fd = [(up - um) / (2.0 * h), (vp - vm) / (2.0 * h)];
            for (row, want) in fd.iter().enumerate() {
                let got = b.cam_j[(row, adj_c + k)];
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{name}: adjustment {k}, row {row} is {got}, the difference {want}"
                );
            }
        }
    }
}

/// `count` adjustments drawn about zero with spread `tau` on each axis of the
/// principal point and none in the focal.
fn principal_point_draws(count: usize, tau: f64) -> Vec<IntrinsicAdjustment> {
    (0..count)
        .map(|i| IntrinsicAdjustment {
            focal: 0.0,
            principal_point: [tau * gaussian(i, 41), tau * gaussian(i, 42)],
            radial_bspline: [0.0; 2],
            rolling_shutter: [0.0; 3],
        })
        .collect()
}

/// The standard deviation of the principal-point offsets of `planted` about
/// their mean, both axes pooled.
fn principal_point_spread(planted: &[IntrinsicAdjustment]) -> f64 {
    let n = planted.len() as f64;
    let mean = |k: usize| planted.iter().map(|a| a.principal_point[k]).sum::<f64>() / n;
    let (mx, my) = (mean(0), mean(1));
    let ss: f64 = planted
        .iter()
        .map(|a| (a.principal_point[0] - mx).powi(2) + (a.principal_point[1] - my).powi(2))
        .sum();
    (ss / (2.0 * n)).sqrt()
}

/// The principal-point width the final round of `out`'s one camera ran under.
fn pp_width(out: &BundleAdjustment) -> f64 {
    out.image_priors[0]
        .expect("a per-image release")
        .widths
        .principal_point
}

#[test]
fn planted_per_image_adjustments_are_recovered_about_their_mean() {
    // Noise-free observations made through a camera whose images each carry a
    // focal scale and a principal-point offset about a common part that is
    // not zero: the focal 1% long and the principal point (6, −4) px off
    // centre on average. The solve starts every image at the shared camera
    // with no adjustment, and recovers each image's intrinsics: the camera
    // takes the mean, and the adjustments end with a mean of zero. The focal's
    // mean is the geometric mean of `1 + s`, the mean of the log focal scale;
    // the arithmetic mean differs from it by about `var(s)/2`, 2e-5 here,
    // far beyond the tolerance.
    let truth = wide_pinhole(326.0, 236.0);
    let n_img = 20;
    let planted: Vec<IntrinsicAdjustment> = (0..n_img)
        .map(|i| IntrinsicAdjustment {
            focal: 0.01 + 0.006 * gaussian(i, 43),
            principal_point: [0.005 * gaussian(i, 41), 0.005 * gaussian(i, 42)],
            radial_bspline: [0.0; 2],
            rolling_shutter: [0.0; 3],
        })
        .collect();
    let mut s = per_image_scene(&truth, n_img, 400, &planted, 0.0, 21, |_| 100);
    perturb_poses(&mut s);
    let start = wide_pinhole(320.0, 240.0);
    let out = run_image(&mut s, &start, &[], IMAGE_BOTH, None, &DEFAULT_SCHEDULE);
    let n = n_img as f64;
    let scale = (planted.iter().map(|a| a.focal.ln_1p()).sum::<f64>() / n).exp();
    let mean = |k: usize| planted.iter().map(|a| a.principal_point[k]).sum::<f64>() / n;
    let (mx, my) = (mean(0), mean(1));
    // The camera holds the mean.
    let (cx, cy) = out.cameras[0].principal_point();
    let (want_cx, want_cy) = (326.0 + mx * 480.0, 236.0 + my * 480.0);
    assert!(
        (cx - want_cx).abs() < 1e-4 && (cy - want_cy).abs() < 1e-4,
        "principal point ({cx}, {cy}), the images' mean ({want_cx}, {want_cy})"
    );
    let f = out.cameras[0].focal_lengths().0;
    let want_f = truth.focal_lengths().0 * scale;
    assert!(
        (f / want_f - 1.0).abs() < 1e-7,
        "focal {f}, the images' mean {want_f}"
    );
    // Each image keeps its difference from the mean.
    for (i, (got, planted)) in out.image_adjustments.iter().zip(&planted).enumerate() {
        let want_s = (1.0 + planted.focal) / scale - 1.0;
        assert!(
            (got.focal - want_s).abs() < 1e-7,
            "image {i}: focal scale {}, planted {want_s}",
            got.focal
        );
        let want = [
            planted.principal_point[0] - mx,
            planted.principal_point[1] - my,
        ];
        for k in 0..2 {
            assert!(
                (got.principal_point[k] - want[k]).abs() < 1e-7,
                "image {i}: offset {:?}, planted {want:?}",
                got.principal_point
            );
        }
    }
    // The mean adjustment is zero, to rounding: the mean of the log focal
    // scale, so the geometric mean of `1 + s` is one.
    let mean_of = |value: &dyn Fn(&IntrinsicAdjustment) -> f64| {
        out.image_adjustments.iter().map(value).sum::<f64>() / n
    };
    assert!(
        mean_of(&|a| a.focal.ln_1p()).abs() < 1e-15,
        "mean log focal scale"
    );
    assert!(
        mean_of(&|a| a.principal_point[0]).abs() < 1e-15,
        "mean x offset"
    );
    assert!(
        mean_of(&|a| a.principal_point[1]).abs() < 1e-15,
        "mean y offset"
    );
    let worst = out
        .residual_norms
        .iter()
        .copied()
        .filter(|r| r.is_finite())
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-4, "largest residual {worst} px");
    // The prior says it was derived.
    let prior = out.image_priors[0].expect("a per-image release");
    assert!(prior.derived);
    assert!(prior.widths.focal.is_finite() && prior.widths.principal_point.is_finite());
}

#[test]
fn the_derived_width_recovers_a_planted_spread() {
    // Forty images each seeing 60 points through its own principal point,
    // drawn with a spread of 0.4% of the short side (1.9 px), and half a
    // pixel of noise. The derived width is that spread, within what forty
    // images can say about it.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(40, 0.004);
    let tau = principal_point_spread(&planted);
    let mut s = per_image_scene(&truth, 40, 400, &planted, 0.5, 101, |_| 60);
    perturb_poses(&mut s);
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_PP,
        None,
        &DEFAULT_SCHEDULE,
    );
    let width = pp_width(&out);
    assert!(
        (0.75 * tau..1.35 * tau).contains(&width),
        "derived width {width}, planted spread {tau}"
    );
    let prior = out.image_priors[0].unwrap();
    assert_eq!(
        prior.spread.principal_point, width,
        "a width the clamp did not set"
    );
    assert!(
        prior.widths.focal.is_nan(),
        "the focal is not released per image"
    );
}

#[test]
fn the_derived_width_holds_when_each_image_is_barely_determined() {
    // Each image sees 25 points with 0.7 px of noise, so the variance of its
    // offset is comparable to the planted spread's square. Undoing the prior's
    // pull and subtracting the noise recovers the spread, and deriving again
    // on the same observations round after round does not run it down.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(40, 0.004);
    let tau = principal_point_spread(&planted);
    let mut s = per_image_scene(&truth, 40, 400, &planted, 0.7, 21, |_| 25);
    perturb_poses(&mut s);
    let start = wide_pinhole(320.0, 240.0);
    let round = BaSchedule {
        trim_px: f64::INFINITY,
        loss_scale: 1.0,
    };
    let widths: Vec<f64> = [2usize, 4, 6]
        .iter()
        .map(|&rounds| {
            let out = run_image(
                &mut s.clone(),
                &start,
                &[],
                IMAGE_PP,
                None,
                &vec![round; rounds],
            );
            pp_width(&out)
        })
        .collect();
    assert!(
        (0.7 * tau..1.4 * tau).contains(&widths[0]),
        "derived width {} after one derivation, planted spread {tau}",
        widths[0]
    );
    for w in &widths[1..] {
        assert!(
            (w / widths[0] - 1.0).abs() < 0.05,
            "the width moved from {} to {w} over later rounds",
            widths[0]
        );
    }
}

#[test]
fn an_image_with_few_observations_does_not_inflate_the_width() {
    // The scene of the planted spread, with one more image that sees eight
    // points through an offset of 5% of the short side, twelve planted
    // spreads out. Its variance is large, so it weighs almost nothing in the
    // spread, and the width stays near the one without it.
    let truth = wide_pinhole(326.0, 236.0);
    let mut planted = principal_point_draws(40, 0.004);
    let start = wide_pinhole(320.0, 240.0);
    let base_width = {
        let mut s = per_image_scene(&truth, 40, 400, &planted, 0.5, 101, |_| 60);
        perturb_poses(&mut s);
        pp_width(&run_image(
            &mut s,
            &start,
            &[],
            IMAGE_PP,
            None,
            &DEFAULT_SCHEDULE,
        ))
    };
    planted.push(IntrinsicAdjustment {
        focal: 0.0,
        principal_point: [0.05, -0.05],
        radial_bspline: [0.0; 2],
        rolling_shutter: [0.0; 3],
    });
    let mut s = per_image_scene(&truth, 41, 400, &planted, 0.5, 101, |i| {
        if i == 40 {
            8
        } else {
            60
        }
    });
    perturb_poses(&mut s);
    let out = run_image(&mut s, &start, &[], IMAGE_PP, None, &DEFAULT_SCHEDULE);
    let width = pp_width(&out);
    assert!(
        (width / base_width - 1.0).abs() < 0.15,
        "width {width} with the sparse image, {base_width} without"
    );
}

#[test]
fn a_camera_whose_images_do_not_vary_derives_the_floor() {
    // Every image through the same principal point: the spread rounds to
    // nothing, and the width is the floor.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = vec![IntrinsicAdjustment::default(); 30];
    let mut s = per_image_scene(&truth, 30, 400, &planted, 0.0, 21, |_| 60);
    perturb_poses(&mut s);
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_PP,
        None,
        &DEFAULT_SCHEDULE,
    );
    let floor = ImagePriorOptions::default().floor.principal_point;
    assert_eq!(pp_width(&out), floor);
    assert!(out.image_priors[0].unwrap().spread.principal_point < floor);
}

#[test]
fn fixed_widths_are_used_as_given() {
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(12, 0.004);
    let mut s = per_image_scene(&truth, 12, 300, &planted, 0.3, 21, |_| 60);
    let fixed = ImagePriorWidths {
        focal: 0.0123,
        principal_point: 0.0456,
    };
    let options = ImagePriorOptions {
        fixed: Some(fixed),
        ..ImagePriorOptions::default()
    };
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_BOTH,
        Some(&options),
        &DEFAULT_SCHEDULE,
    );
    let prior = out.image_priors[0].unwrap();
    assert_eq!(prior.widths, fixed);
    assert!(!prior.derived);
    assert!(prior.spread.focal.is_nan() && prior.spread.principal_point.is_nan());
}

#[test]
fn a_narrow_prior_leaves_the_images_mean_free() {
    // Every image through the same principal point, 6 px right of and 4 px
    // above the starting camera's, under a fixed width far narrower than that
    // offset, in one round. The prior is centred on the images' mean rather
    // than on zero, so it holds the images to each other and not to the
    // starting camera: the images move together, and the re-centring hands
    // their common offset to the camera.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = vec![IntrinsicAdjustment::default(); 20];
    let mut s = per_image_scene(&truth, 20, 400, &planted, 0.3, 21, |_| 60);
    perturb_poses(&mut s);
    let narrow = ImagePriorWidths {
        focal: 1e-5,
        principal_point: 1e-5,
    };
    let options = ImagePriorOptions {
        fixed: Some(narrow),
        ..ImagePriorOptions::default()
    };
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_PP,
        Some(&options),
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 1.0,
        }],
    );
    let (cx, cy) = out.cameras[0].principal_point();
    assert!(
        (cx - 326.0).abs() < 1.5 && (cy - 236.0).abs() < 1.5,
        "principal point ({cx}, {cy}), the images' (326, 236)"
    );
}

#[test]
fn a_narrow_focal_prior_leaves_the_images_mean_free() {
    // Every image through the same focal, 3% longer than the starting
    // camera's, under a fixed focal width far narrower than that, in one
    // round. The prior is centred on the mean of the images' log focal scales
    // rather than on zero, so the images move together and the re-centring
    // hands their common scale to the camera.
    let truth = wide_pinhole(320.0, 240.0);
    let planted = vec![
        IntrinsicAdjustment {
            focal: 0.03,
            principal_point: [0.0, 0.0],
            radial_bspline: [0.0; 2],
            rolling_shutter: [0.0; 3],
        };
        20
    ];
    let mut s = per_image_scene(&truth, 20, 400, &planted, 0.3, 21, |_| 60);
    perturb_poses(&mut s);
    let options = ImagePriorOptions {
        fixed: Some(ImagePriorWidths {
            focal: 1e-5,
            principal_point: 1e-5,
        }),
        ..ImagePriorOptions::default()
    };
    let out = run_image(
        &mut s,
        &truth,
        &[],
        IMAGE_F,
        Some(&options),
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 1.0,
        }],
    );
    let f = out.cameras[0].focal_lengths().0;
    let want = truth.focal_lengths().0 * 1.03;
    assert!(
        (f / want - 1.0).abs() < 2e-3,
        "focal {f}, the images' {want}"
    );
}

#[test]
fn the_derived_focal_width_recovers_a_planted_spread() {
    // Forty images each seeing 60 points through its own focal, whose log
    // focal scales are drawn with a spread of 1%, and half a pixel of noise.
    // The derived focal width is that spread of the log focal scale, within
    // what forty images can say about it.
    let truth = wide_pinhole(320.0, 240.0);
    let planted: Vec<IntrinsicAdjustment> = (0..40)
        .map(|i| IntrinsicAdjustment {
            focal: (0.01 * gaussian(i, 43)).exp_m1(),
            principal_point: [0.0, 0.0],
            radial_bspline: [0.0; 2],
            rolling_shutter: [0.0; 3],
        })
        .collect();
    let n = planted.len() as f64;
    let t: Vec<f64> = planted.iter().map(|a| a.focal.ln_1p()).collect();
    let t_mean = t.iter().sum::<f64>() / n;
    let tau = (t.iter().map(|x| (x - t_mean).powi(2)).sum::<f64>() / n).sqrt();
    let mut s = per_image_scene(&truth, 40, 400, &planted, 0.5, 101, |_| 60);
    perturb_poses(&mut s);
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_F,
        None,
        &DEFAULT_SCHEDULE,
    );
    let prior = out.image_priors[0].expect("a per-image release");
    let width = prior.widths.focal;
    assert!(
        (0.75 * tau..1.35 * tau).contains(&width),
        "derived focal width {width}, planted spread {tau}"
    );
    assert_eq!(prior.spread.focal, width, "a width the clamp did not set");
    assert!(prior.widths.principal_point.is_nan());
}

#[test]
fn a_one_round_schedule_derives_once() {
    // A single round runs a second time under widths derived from the first,
    // so its prior is derived and the spread it came from is reported.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(12, 0.004);
    let mut s = per_image_scene(&truth, 12, 300, &planted, 0.3, 21, |_| 60);
    let out = run_image(
        &mut s,
        &wide_pinhole(320.0, 240.0),
        &[],
        IMAGE_PP,
        None,
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 1.0,
        }],
    );
    let prior = out.image_priors[0].unwrap();
    assert!(prior.derived);
    assert!(prior.spread.principal_point.is_finite());
    assert_ne!(
        prior.widths.principal_point,
        ImagePriorOptions::default().loose.principal_point
    );
}

#[test]
fn an_image_outside_the_solve_keeps_its_intrinsics_through_the_recentring() {
    // Image 12 has no observation and carries an adjustment of its own. The
    // camera's shared values move to the mean of the images the solve saw,
    // and image 12's adjustment moves the other way, so its intrinsics are the
    // ones it had.
    let truth = wide_pinhole(326.0, 236.0);
    let mut planted: Vec<IntrinsicAdjustment> = (0..12)
        .map(|i| IntrinsicAdjustment {
            focal: 0.01 + 0.004 * gaussian(i, 43),
            principal_point: [0.01 + 0.004 * gaussian(i, 41), 0.004 * gaussian(i, 42)],
            radial_bspline: [0.0; 2],
            rolling_shutter: [0.0; 3],
        })
        .collect();
    planted.push(IntrinsicAdjustment::default());
    let mut s = per_image_scene(&truth, 13, 300, &planted, 0.0, 21, |i| {
        if i == 12 {
            0
        } else {
            60
        }
    });
    assert!(!s.obs_img.contains(&12));
    let start = wide_pinhole(320.0, 240.0);
    let outside = IntrinsicAdjustment {
        focal: -0.02,
        principal_point: [0.03, -0.01],
        radial_bspline: [0.0; 2],
        rolling_shutter: [0.0; 3],
    };
    let mut start_adj = vec![IntrinsicAdjustment::default(); 13];
    start_adj[12] = outside;
    let before = start.with_adjustment(&outside, None);
    let out = run_image(
        &mut s,
        &start,
        &start_adj,
        IMAGE_BOTH,
        None,
        &DEFAULT_SCHEDULE,
    );
    assert_ne!(out.cameras[0].principal_point(), start.principal_point());
    assert_ne!(out.cameras[0].focal_lengths(), start.focal_lengths());
    let after = out.cameras[0].with_adjustment(&out.image_adjustments[12], None);
    let (fb, fa) = (before.focal_lengths().0, after.focal_lengths().0);
    assert!((fa / fb - 1.0).abs() < 1e-12, "focal {fb} → {fa}");
    let (pb, pa) = (before.principal_point(), after.principal_point());
    assert!(
        (pa.0 - pb.0).abs() < 1e-9 && (pa.1 - pb.1).abs() < 1e-9,
        "principal point {pb:?} → {pa:?}"
    );
}

#[test]
fn the_shared_and_per_image_release_of_one_kind_is_the_per_image_release() {
    // Asking for the shared principal point and focal beside their per-image
    // releases solves exactly what the per-image releases alone solve.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(10, 0.004);
    let scene = per_image_scene(&truth, 10, 200, &planted, 0.3, 21, |_| 50);
    let start = wide_pinhole(320.0, 240.0);
    let both = CameraRelease {
        focal: true,
        principal_point: true,
        ..IMAGE_BOTH
    };
    let mut a = scene.clone();
    let mut b = scene.clone();
    let out_a = run_image(&mut a, &start, &[], IMAGE_BOTH, None, &DEFAULT_SCHEDULE);
    let out_b = run_image(&mut b, &start, &[], both, None, &DEFAULT_SCHEDULE);
    assert_bitwise_equal(&a, &out_a, &b, &out_b);
    assert_eq!(out_a.cameras, out_b.cameras);
    assert_eq!(out_a.image_adjustments, out_b.image_adjustments);
    assert_eq!(
        format!("{:?}", out_a.image_priors),
        format!("{:?}", out_b.image_priors)
    );
}

#[test]
fn per_image_flags_with_nothing_released_per_image_change_nothing() {
    // The flags on with the camera's release holding its images, and the
    // flags off: the same solve, bit for bit, with no prior reported.
    let truth = wide_pinhole(326.0, 236.0);
    let planted = principal_point_draws(8, 0.004);
    let scene = per_image_scene(&truth, 8, 200, &planted, 0.3, 21, |_| 50);
    let start = wide_pinhole(320.0, 240.0);
    let shared = CameraRelease {
        focal: true,
        principal_point: true,
        ..CameraRelease::HELD
    };
    let mut a = scene.clone();
    let mut b = scene.clone();
    let out_a = run_image(&mut a, &start, &[], shared, None, &DEFAULT_SCHEDULE);
    let out_b = run_pp(&mut b, &start, true, true, &DEFAULT_SCHEDULE);
    assert_bitwise_equal(&a, &out_a, &b, &out_b);
    assert_eq!(out_a.cameras, out_b.cameras);
    assert_eq!(out_a.image_priors, vec![None]);
    assert!(out_a
        .image_adjustments
        .iter()
        .all(IntrinsicAdjustment::is_zero));
}

#[test]
fn the_image_focal_guard_keeps_the_focal_positive() {
    // The guard reads the log focal scale `t`: any finite `t` whose focal
    // `f·exp(t)` is finite and above `1e-6`, so `1 + s = exp(t) > 0` always.
    assert!(image_focal_step_admissible(500.0, 0.0));
    assert!(image_focal_step_admissible(500.0, -0.9));
    assert!(image_focal_step_admissible(500.0, -1.5));
    assert!(image_focal_step_admissible(500.0, 2.0));
    assert!(!image_focal_step_admissible(500.0, -25.0));
    assert!(!image_focal_step_admissible(500.0, 800.0));
    assert!(!image_focal_step_admissible(500.0, f64::NAN));
    assert!(!image_focal_step_admissible(500.0, f64::INFINITY));
    assert!(!image_focal_step_admissible(500.0, f64::NEG_INFINITY));
    assert!(!image_focal_step_admissible(1e-7, 0.0));
}

#[test]
fn an_image_principal_point_pulled_off_the_image_stays_on_it() {
    // Image 0's observations are made through a principal point 40 px left of
    // the image. Under a prior wide enough not to hold it, the solve pulls its
    // offset toward that point, rejects every step that would take it off the
    // image, and ends with it on the image.
    let truth = wide_pinhole(320.0, 240.0);
    let mut planted = vec![IntrinsicAdjustment::default(); 8];
    planted[0].principal_point = [-360.0 / 480.0, 0.0];
    let mut s = per_image_scene(&truth, 8, 300, &planted, 0.0, 21, |_| 80);
    // Image 0 starts 20 px in from the edge, and nothing is trimmed, so its
    // observations, tens of pixels off, stay in the solve.
    let mut start_adj = vec![IntrinsicAdjustment::default(); 8];
    start_adj[0].principal_point = [-300.0 / 480.0, 0.0];
    let options = ImagePriorOptions {
        fixed: Some(ImagePriorWidths {
            focal: 1.0,
            principal_point: 10.0,
        }),
        ..ImagePriorOptions::default()
    };
    let out = run_image(
        &mut s,
        &truth,
        &start_adj,
        IMAGE_PP,
        Some(&options),
        &[BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 10.0,
        }],
    );
    for (i, a) in out.image_adjustments.iter().enumerate() {
        let (cx, cy) = out.cameras[0].with_adjustment(a, None).principal_point();
        assert!(
            (0.0..=640.0).contains(&cx) && (0.0..=480.0).contains(&cy),
            "image {i}'s principal point ({cx}, {cy}) is off the image"
        );
    }
    let (cx0, _) = out.cameras[0]
        .with_adjustment(&out.image_adjustments[0], None)
        .principal_point();
    assert!(
        cx0 < 20.0,
        "image 0's principal point x {cx0} did not move toward the planted one"
    );
}

#[test]
fn the_derivation_undoes_the_prior_and_subtracts_the_noise() {
    // The derivation alone, on samples made the way a round makes them: each
    // image's data-only estimate is its planted deviation plus noise of
    // variance `v`, and the solved value is that estimate pulled toward the
    // mean by the round's prior. With `v` about the planted spread's square
    // and the round run under a width half the spread, the derived spread is
    // the planted one.
    let tau = 0.004;
    let sigma = 0.002;
    let n = 400;
    let sigma_noise = 0.5;
    let samples: Vec<(IntrinsicAdjustment, Option<Matrix3<f64>>)> = (0..n)
        .map(|i| {
            // The images' variances differ by a factor of four.
            let v = tau * tau * (0.5 + 0.75 * (jitter(i, 7) + 1.0));
            let estimate = |k: u64| tau * gaussian(i, 50 + k) + v.sqrt() * gaussian(i, 60 + k);
            // The pull of a prior stated at the level of the noise:
            // `d = d̂·σ²/(σ² + v)`.
            let pull = sigma * sigma / (sigma * sigma + v);
            let mut inv = Matrix3::zeros();
            inv[(1, 1)] = v / (sigma_noise * sigma_noise);
            inv[(2, 2)] = v / (sigma_noise * sigma_noise);
            (
                IntrinsicAdjustment {
                    focal: 0.0,
                    principal_point: [estimate(0) * pull, estimate(1) * pull],
                    radial_bspline: [0.0; 2],
                    rolling_shutter: [0.0; 3],
                },
                Some(inv),
            )
        })
        .collect();
    let round = ImagePriorWidths {
        focal: 0.1,
        principal_point: sigma,
    };
    let options = ImagePriorOptions::default();
    let (next, spread) = derive_image_prior_widths(
        &samples,
        false,
        true,
        round,
        sigma_noise,
        sigma_noise,
        &options,
    );
    assert!(
        (spread.principal_point / tau - 1.0).abs() < 0.15,
        "derived spread {}, planted {tau}",
        spread.principal_point
    );
    assert_eq!(next.principal_point, spread.principal_point);
    assert!(spread.focal.is_nan());
    assert_eq!(next.focal, options.loose.focal);
    // The pulled samples alone spread far less than the planted spread.
    let pulled: Vec<IntrinsicAdjustment> = samples.iter().map(|(a, _)| *a).collect();
    let shrunk = principal_point_spread(&pulled);
    assert!(shrunk < 0.6 * tau, "the pulled samples spread {shrunk}");
    // One image of positive weight leaves the loose width.
    let (one, one_spread) = derive_image_prior_widths(
        &samples[..1],
        false,
        true,
        round,
        sigma_noise,
        sigma_noise,
        &options,
    );
    assert_eq!(one.principal_point, options.loose.principal_point);
    assert!(one_spread.principal_point.is_nan());
}

#[test]
fn the_focal_width_is_derived_from_the_log_focal_scale() {
    // The same deviations given once as each image's log focal scale
    // `t = ln(1 + s)` and once as both axes of its principal-point offset, with
    // the same variances and the same round's width: the derivation reads the
    // focal through `t`, so the two spreads agree to rounding. The spread of
    // 20% is wide enough that the stored scales `s` spread differently.
    let tau = 0.2;
    let sigma_noise = 0.5;
    let samples = |focal: bool| -> Vec<(IntrinsicAdjustment, Option<Matrix3<f64>>)> {
        (0..200)
            .map(|i| {
                let v = 0.01 * (0.5 + 0.75 * (jitter(i, 7) + 1.0));
                let x = tau * gaussian(i, 50) + v.sqrt() * gaussian(i, 60);
                let mut inv = Matrix3::zeros();
                for k in 0..3 {
                    inv[(k, k)] = v / (sigma_noise * sigma_noise);
                }
                let a = if focal {
                    IntrinsicAdjustment {
                        focal: x.exp_m1(),
                        principal_point: [0.0, 0.0],
                        radial_bspline: [0.0; 2],
                        rolling_shutter: [0.0; 3],
                    }
                } else {
                    IntrinsicAdjustment {
                        focal: 0.0,
                        principal_point: [x, x],
                        radial_bspline: [0.0; 2],
                        rolling_shutter: [0.0; 3],
                    }
                };
                (a, Some(inv))
            })
            .collect()
    };
    let round = ImagePriorWidths {
        focal: 0.1,
        principal_point: 0.1,
    };
    let options = ImagePriorOptions::default();
    let (_, of_focal) = derive_image_prior_widths(
        &samples(true),
        true,
        false,
        round,
        sigma_noise,
        sigma_noise,
        &options,
    );
    let (_, of_pp) = derive_image_prior_widths(
        &samples(false),
        false,
        true,
        round,
        sigma_noise,
        sigma_noise,
        &options,
    );
    assert!(
        (of_focal.focal / of_pp.principal_point - 1.0).abs() < 1e-9,
        "focal spread {}, the same deviations' {}",
        of_focal.focal,
        of_pp.principal_point
    );
}
