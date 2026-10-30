// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The release of each image's radial correction and rolling-shutter rates
//! over a whole reconstruction: the scanlines choice and its race, the
//! write-back of the corrections, the report, and the refusals.
//!
//! The fixture is a scene of `SFMTOOL_PINHOLE` images seen from a general
//! motion, each image's keypoints made through its own planted correction and
//! rates, read on rows or on columns, with a little noise.

use super::*;

/// A 60° `SFMTOOL_PINHOLE` of 640×480 (short side 480) with a mild barrel
/// spline on a domain that reaches past the image's corners.
fn spline_pinhole() -> CameraIntrinsics {
    CameraIntrinsics {
        model: CameraModel::SfmtoolPinhole {
            focal_length: 320.0 / 30f64.to_radians().tan(),
            principal_point_x: 326.0,
            principal_point_y: 236.0,
            bspline_rho_max: 0.8,
            bspline: vec![-0.001, -0.004, -0.009, -0.016, -0.025, -0.036],
        },
        width: 640,
        height: 480,
        rolling_shutter: None,
    }
}

/// A standard normal draw from an index, by Box–Muller over [`jitter`].
fn gaussian(i: usize, salt: u64) -> f64 {
    let u1 = 0.5 * (jitter(i, salt) + 1.0) * 0.9998 + 0.0001;
    let u2 = 0.5 * (jitter(i, salt ^ 0x5555) + 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// `count` corrections drawn about zero, each radial coefficient with spread
/// `tau_e` and each rate with spread `tau_w`, with their mean removed.
fn correction_draws(count: usize, tau_e: f64, tau_w: f64, salt: u64) -> Vec<IntrinsicAdjustment> {
    let mut out: Vec<IntrinsicAdjustment> = (0..count)
        .map(|i| IntrinsicAdjustment {
            radial_bspline: [tau_e * gaussian(i, salt + 1), tau_e * gaussian(i, salt + 2)],
            rolling_shutter: [
                tau_w * gaussian(i, salt + 3),
                tau_w * gaussian(i, salt + 4),
                tau_w * gaussian(i, salt + 5),
            ],
            ..IntrinsicAdjustment::default()
        })
        .collect();
    let n = count as f64;
    let mean_e: [f64; 2] =
        std::array::from_fn(|k| out.iter().map(|a| a.radial_bspline[k]).sum::<f64>() / n);
    let mean_w: [f64; 3] =
        std::array::from_fn(|k| out.iter().map(|a| a.rolling_shutter[k]).sum::<f64>() / n);
    for a in &mut out {
        for (e, m) in a.radial_bspline.iter_mut().zip(mean_e) {
            *e -= m;
        }
        for (w, m) in a.rolling_shutter.iter_mut().zip(mean_w) {
            *w -= m;
        }
    }
    out
}

const CORRECTION_IMAGES: usize = 30;
const CORRECTION_POINTS: usize = 400;
const PER_IMAGE_POINTS: usize = 80;

/// A scene of [`CORRECTION_IMAGES`] images, image `i` taken through camera
/// `camera_of(i)` of `cameras` with the planted adjustment `planted[i]`, its
/// rates read on `planted_scanlines[camera]`, looking from a cap of directions
/// at [`CORRECTION_POINTS`] points, each image keeping the first
/// [`PER_IMAGE_POINTS`] that land inside it, with 0.1 px of noise per axis.
/// The value handed back carries no correction and no scanlines setting, and
/// its poses are perturbed.
fn corrections_source(
    cameras: Vec<CameraIntrinsics>,
    camera_of: impl Fn(usize) -> u32,
    planted: &[IntrinsicAdjustment],
    planted_scanlines: &[RollingShutterScanlines],
) -> SfmrReconstruction {
    let n_img = CORRECTION_IMAGES;
    let mut recon = SfmrReconstruction::demo(1);
    let f = cameras[0].focal_lengths().0;
    let (half_w, half_h) = (
        8.0 * f64::from(cameras[0].width) / 2.0 / f,
        8.0 * f64::from(cameras[0].height) / 2.0 / f,
    );
    recon.image_set.per_image_adjustments = planted_scanlines
        .iter()
        .map(|&s| PerImageAdjustments {
            rolling_shutter_scanlines: Some(s),
            ..PerImageAdjustments::default()
        })
        .collect();
    recon.image_set.cameras = cameras;
    recon.image_set.images = (0..n_img)
        .map(|i| {
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
            SfmrImage {
                name: format!("image_{i:03}.jpg"),
                camera_index: camera_of(i),
                quaternion_wxyz: r,
                translation_xyz: -(r * center),
                intrinsic_adjustment: planted[i],
            }
        })
        .collect();
    let stats = recon.image_set.depth_statistics.images[0].clone();
    recon.image_set.depth_statistics.images = vec![stats; n_img];
    recon.image_set.depth_histogram_counts =
        vec![recon.image_set.depth_histogram_counts[0].clone(); n_img];
    recon.point_set.points = (0..CORRECTION_POINTS)
        .map(|p| Point3D {
            position: Point3::new(
                0.85 * half_w * jitter(p, 1),
                0.85 * half_h * jitter(p, 2),
                0.5 * half_w * jitter(p, 3),
            ),
            w: 1.0,
            color: [10, 20, 30],
            rms_reprojection_error: 0.25,
            normal: Vector3::new(0.0, 0.0, -1.0),
        })
        .collect();

    let intrinsics: Vec<CameraIntrinsics> = (0..n_img)
        .map(|i| recon.image_set.image_intrinsics(i).into_owned())
        .collect();
    let mut tracks = Vec::new();
    let mut counts = vec![0u32; CORRECTION_POINTS];
    let mut keypoints: Vec<[f32; 2]> = Vec::new();
    let mut seen = vec![0usize; n_img];
    for (p, count) in counts.iter_mut().enumerate() {
        let x = recon.point_set.points[p].position;
        for i in 0..n_img {
            if seen[i] >= PER_IMAGE_POINTS {
                continue;
            }
            let image = &recon.image_set.images[i];
            let c = image.quaternion_wxyz * x.coords + image.translation_xyz;
            let Some((u, v)) = intrinsics[i].ray_to_pixel([c.x, c.y, c.z]) else {
                continue;
            };
            let cam = &intrinsics[i];
            if !(0.0..f64::from(cam.width)).contains(&u)
                || !(0.0..f64::from(cam.height)).contains(&v)
            {
                continue;
            }
            let k = keypoints.len();
            keypoints.push([
                (u + 0.1 * gaussian(k, 21)) as f32,
                (v + 0.1 * gaussian(k, 22)) as f32,
            ]);
            tracks.push(TrackObservation {
                image_index: i as u32,
                point_index: p as u32,
            });
            *count += 1;
            seen[i] += 1;
        }
    }
    // Keep the points at least two images see, and close up the indexes.
    let live: Vec<bool> = counts.iter().map(|&c| c >= 2).collect();
    let mut new_index = vec![u32::MAX; CORRECTION_POINTS];
    let mut next = 0u32;
    for p in 0..CORRECTION_POINTS {
        if live[p] {
            new_index[p] = next;
            next += 1;
        }
    }
    let rows: Vec<usize> = (0..tracks.len())
        .filter(|&r| live[tracks[r].point_index as usize])
        .collect();
    let mut keypoints_xy = Array2::<f32>::zeros((rows.len(), 2));
    let mut kept_tracks = Vec::with_capacity(rows.len());
    for (n, &r) in rows.iter().enumerate() {
        keypoints_xy[[n, 0]] = keypoints[r][0];
        keypoints_xy[[n, 1]] = keypoints[r][1];
        kept_tracks.push(TrackObservation {
            image_index: tracks[r].image_index,
            point_index: new_index[tracks[r].point_index as usize],
        });
    }
    recon.point_set.points = (0..CORRECTION_POINTS)
        .filter(|&p| live[p])
        .map(|p| recon.point_set.points[p].clone())
        .collect();
    let set = &mut recon.point_set;
    set.observation_counts = counts.iter().copied().filter(|&c| c >= 2).collect();
    set.tracks = kept_tracks;
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy,
        image_file_hashes: vec![[0u8; 16]; n_img],
    };
    set.patch_u_halfvec_xyz = None;
    set.patch_v_halfvec_xyz = None;
    recon.metadata.feature_source =
        sfmtool_sfmr_format::FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    recon.rebuild_derived_fields();

    // What the solve is handed: no correction, no setting, the poses nudged.
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.intrinsic_adjustment = IntrinsicAdjustment::default();
        let d = Vector3::new(jitter(i, 31), jitter(i, 32), jitter(i, 33)) * 2e-3;
        image.quaternion_wxyz = UnitQuaternion::from_scaled_axis(d) * image.quaternion_wxyz;
        image.translation_xyz += Vector3::new(jitter(i, 34), jitter(i, 35), jitter(i, 36)) * 0.01;
    }
    recon.image_set.per_image_adjustments.clear();
    recon
}

/// One camera's scene with corrections planted on `scanlines`.
fn one_camera_source(scanlines: RollingShutterScanlines, salt: u64) -> SfmrReconstruction {
    let planted = correction_draws(CORRECTION_IMAGES, 3e-3, 4e-3, salt);
    corrections_source(vec![spline_pinhole()], |_| 0, &planted, &[scanlines])
}

/// Both corrections released per image.
const CORRECTIONS: CameraRelease = CameraRelease {
    image_radial_bspline: true,
    image_rolling_shutter: true,
    ..CameraRelease::HELD
};

const BOTH: [RollingShutterScanlines; 2] = [
    RollingShutterScanlines::Rows,
    RollingShutterScanlines::Columns,
];

/// The phase names opened at any depth, in order, that name a race's solve.
fn race_phases(collector: &Collector) -> Vec<&'static str> {
    collector
        .entered()
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| name.starts_with("solve with"))
        .collect()
}

#[test]
fn unknown_scanlines_pick_the_planted_ones() {
    // A camera with no stored scanlines and no choice falls back to UNKNOWN:
    // two solves, and the one on the scanlines the rates were planted on ends
    // at the lower cost, is kept, and is written into the camera.
    for (n, planted) in BOTH.into_iter().enumerate() {
        let source = one_camera_source(planted, 70 + 10 * n as u64);
        let collector = Collector::recording();
        let sink = |event: Event<'_>| collector.push(event);
        let (out, report) = bundle_adjust(
            &source,
            &BundleAdjustOptions::uniform(1, CORRECTIONS),
            &Progress::to(&sink),
        )
        .expect("well posed");
        assert_eq!(
            race_phases(&collector),
            ["solve with rows", "solve with columns"]
        );
        let camera = &report.cameras[0];
        let race = camera.scanlines_race.expect("a race");
        // The planted rates are clear enough that the other solve is stopped,
        // and the costs are the two at the stop, their difference the gap.
        let stop = race.stop.expect("the trailing solve was stopped");
        assert!(race.stopped_early);
        assert_ne!(stop.stopped, planted);
        assert!(race.observations > 1000, "{race:?}");
        assert!(race.sigma_px > 0.05 && race.sigma_px < 0.2, "{race:?}");
        assert!(
            ((race.rows_cost - race.columns_cost).abs() - stop.gap).abs() < 1e-9 * stop.gap,
            "{race:?}"
        );
        assert!(stop.gap > 100.0 && stop.winner_cost.is_finite(), "{race:?}");
        // The stopped solve's phase says where it stopped.
        let stopped_phase = match stop.stopped {
            RollingShutterScanlines::Rows => "solve with rows",
            RollingShutterScanlines::Columns => "solve with columns",
        };
        let note = format!(
            "stopped at round {}, iteration {}",
            stop.round, stop.iteration
        );
        assert!(
            collector
                .left()
                .iter()
                .any(|(name, _, said)| *name == stopped_phase && said.as_deref() == Some(&note)),
            "{:?}",
            collector.left()
        );
        let (won, lost) = match planted {
            RollingShutterScanlines::Rows => (race.rows_cost, race.columns_cost),
            RollingShutterScanlines::Columns => (race.columns_cost, race.rows_cost),
        };
        assert!(won < lost, "{planted:?}: {race:?}");
        assert_eq!(camera.rolling_shutter_scanlines, Some(planted));
        assert!(camera.image_radial_bspline_released && camera.image_rolling_shutter_released);
        assert_eq!(
            out.image_set.camera_per_image_adjustments(0),
            PerImageAdjustments {
                focal: false,
                principal_point: false,
                radial_bspline: true,
                rolling_shutter: true,
                rolling_shutter_scanlines: Some(planted),
            }
        );
        // The progress names the comparison and the setting kept.
        let said: Vec<String> = collector
            .events()
            .into_iter()
            .filter_map(|e| match e {
                Owned::Message(_, _, text) => Some(text),
                _ => None,
            })
            .collect();
        assert!(
            said.iter()
                .any(|t| t.starts_with("camera 0: ") && t.ends_with(planted.as_str())),
            "{said:?}"
        );
        // The solved rates are near the planted ones.
        let truth = correction_draws(CORRECTION_IMAGES, 3e-3, 4e-3, 70 + 10 * n as u64);
        let (mut err, mut size) = (0.0, 0.0);
        for (image, want) in out.image_set.images.iter().zip(&truth) {
            for (a, b) in image
                .intrinsic_adjustment
                .rolling_shutter
                .iter()
                .zip(want.rolling_shutter)
            {
                err += (a - b).powi(2);
                size += b * b;
            }
        }
        assert!(
            err < 0.1 * size,
            "{planted:?}: rates off by {err} of {size}"
        );
        assert!(report.median_residual_after < 0.2, "{report:?}");
    }
}

#[test]
fn a_given_or_stored_setting_runs_one_solve_on_it() {
    // Rates planted on columns. A choice of ROWS runs one solve on rows,
    // whatever the camera stores, and writes rows; with no choice, a stored
    // COLUMNS runs one solve on columns.
    let source = one_camera_source(RollingShutterScanlines::Columns, 90);
    let mut stored = source.clone();
    stored.image_set.per_image_adjustments = vec![PerImageAdjustments {
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Columns),
        ..PerImageAdjustments::default()
    }];
    let rows = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![Some(RollingShutterScanlinesChoice::Rows)],
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    };
    for value in [&source, &stored] {
        let (out, report) = bundle_adjust(value, &rows, &Progress::none()).expect("well posed");
        assert_eq!(report.cameras[0].scanlines_race, None);
        assert_eq!(
            report.cameras[0].rolling_shutter_scanlines,
            Some(RollingShutterScanlines::Rows)
        );
        assert_eq!(
            out.image_set
                .camera_per_image_adjustments(0)
                .rolling_shutter_scanlines,
            Some(RollingShutterScanlines::Rows)
        );
    }
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);
    let (out, report) = bundle_adjust(
        &stored,
        &BundleAdjustOptions::uniform(1, CORRECTIONS),
        &Progress::to(&sink),
    )
    .expect("well posed");
    assert!(race_phases(&collector).is_empty());
    assert_eq!(report.cameras[0].scanlines_race, None);
    assert_eq!(
        report.cameras[0].rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Columns)
    );
    assert_eq!(
        out.image_set
            .camera_per_image_adjustments(0)
            .rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Columns)
    );
    // An explicit UNKNOWN races even where a setting is stored.
    let unknown = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![Some(RollingShutterScanlinesChoice::Unknown)],
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    };
    let (_, report) = bundle_adjust(&stored, &unknown, &Progress::none()).expect("well posed");
    assert!(report.cameras[0].scanlines_race.is_some());
    assert_eq!(
        report.cameras[0].rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Columns)
    );
}

/// Two cameras alternating over the images, rows planted on camera 0 and
/// columns on camera 1.
fn two_camera_source() -> SfmrReconstruction {
    let planted = correction_draws(CORRECTION_IMAGES, 3e-3, 4e-3, 120);
    let mut second = spline_pinhole();
    second.model = CameraModel::SfmtoolPinhole {
        focal_length: 560.0,
        principal_point_x: 318.0,
        principal_point_y: 243.0,
        bspline_rho_max: 0.8,
        bspline: vec![-0.001, -0.003, -0.007, -0.012, -0.019, -0.027],
    };
    corrections_source(
        vec![spline_pinhole(), second],
        |i| (i % 2) as u32,
        &planted,
        &BOTH,
    )
}

/// Poses, points, cameras and adjustments as bits.
fn state_bits(recon: &SfmrReconstruction) -> Vec<u64> {
    let mut bits = value_bits(recon);
    for camera in &recon.image_set.cameras {
        bits.extend(
            format!("{:?}", camera.model)
                .bytes()
                .map(u64::from)
                .collect::<Vec<_>>(),
        );
    }
    for image in &recon.image_set.images {
        let a = image.intrinsic_adjustment;
        bits.push(a.focal.to_bits());
        bits.extend(
            a.principal_point
                .iter()
                .chain(&a.radial_bspline)
                .chain(&a.rolling_shutter)
                .map(|v| v.to_bits()),
        );
    }
    bits
}

#[test]
fn several_unknown_cameras_are_raced_in_order_each_from_the_last_winner() {
    // Both cameras UNKNOWN: camera 0 is raced first with camera 1's rates held
    // at zero, then camera 1 from camera 0's winner. Each picks its planted
    // scanlines. The answer is the one two adjustments in sequence give: the
    // first racing camera 0 with camera 1 releasing no rates, the second
    // started from its answer, where camera 0 now stores its winner, racing
    // camera 1. That is the race's documented starting state, to the bit.
    let source = two_camera_source();
    let both = CameraRelease {
        image_radial_bspline: true,
        image_rolling_shutter: true,
        ..CameraRelease::HELD
    };
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);
    let (out, report) = bundle_adjust(
        &source,
        &BundleAdjustOptions::uniform(2, both),
        &Progress::to(&sink),
    )
    .expect("well posed");
    assert_eq!(
        race_phases(&collector),
        [
            "solve with rows",
            "solve with columns",
            "solve with rows",
            "solve with columns"
        ]
    );
    assert_eq!(
        report.cameras[0].rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Rows)
    );
    assert_eq!(
        report.cameras[1].rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Columns)
    );
    assert!(report.cameras.iter().all(|c| c.scanlines_race.is_some()));
    // Each race ran the early stop, and where it stopped a solve it stopped
    // the one not chosen; the answer is the one two full solves per race give.
    for camera in &report.cameras {
        if let Some(stop) = camera.scanlines_race.unwrap().stop {
            assert_ne!(Some(stop.stopped), camera.rolling_shutter_scanlines);
        }
    }
    let full = BundleAdjustOptions {
        scanlines_early_stop: ScanlinesEarlyStopOptions {
            enabled: false,
            ..ScanlinesEarlyStopOptions::default()
        },
        ..BundleAdjustOptions::uniform(2, both)
    };
    let (full_out, _) = bundle_adjust(&source, &full, &Progress::none()).expect("full solves");
    same_bits(
        &state_bits(&out),
        &state_bits(&full_out),
        "against two full solves each",
    );

    let first = releasing(&[
        both,
        CameraRelease {
            image_rolling_shutter: false,
            ..both
        },
    ]);
    let (step, step_report) = bundle_adjust(&source, &first, &Progress::none()).expect("first");
    assert_eq!(step_report.points_deleted, 0);
    assert_eq!(
        step.image_set
            .camera_per_image_adjustments(0)
            .rolling_shutter_scanlines,
        Some(RollingShutterScanlines::Rows)
    );
    assert!(step
        .image_set
        .images
        .iter()
        .filter(|i| i.camera_index == 1)
        .all(|i| i.intrinsic_adjustment.rolling_shutter == [0.0; 3]));
    let (twice, _) = bundle_adjust(
        &step,
        &BundleAdjustOptions::uniform(2, both),
        &Progress::none(),
    )
    .expect("second");
    same_bits(&state_bits(&out), &state_bits(&twice), "two races");
}

#[test]
fn an_unposed_image_keeps_its_spline_and_the_report_states_the_spread() {
    // Image 7 is unposed and carries a radial correction of its own. The mean
    // of the posed images' corrections moves into the spline, and the unposed
    // image's correction takes it off, so its own spline is the one it had.
    let mut source = one_camera_source(RollingShutterScanlines::Rows, 140);
    source.image_set.images[7].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    source.image_set.images[7]
        .intrinsic_adjustment
        .radial_bspline = [2e-3, -1e-3];
    // Every posed image starts from a common correction, so there is a mean
    // to move.
    for (i, image) in source.image_set.images.iter_mut().enumerate() {
        if i != 7 {
            image.intrinsic_adjustment.radial_bspline = [1e-3, 2e-3];
        }
    }
    let options = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![Some(RollingShutterScanlinesChoice::Rows)],
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    };
    let (out, report) = bundle_adjust(&source, &options, &Progress::none()).expect("well posed");
    let spline = |c: &CameraIntrinsics| c.model.radial_spline().unwrap().0.to_vec();
    assert_ne!(
        spline(&out.image_set.cameras[0]),
        spline(&source.image_set.cameras[0])
    );
    let before = spline(&source.image_set.image_intrinsics(7));
    let after = spline(&out.image_set.image_intrinsics(7));
    for (a, b) in after.iter().zip(&before) {
        assert!((a - b).abs() < 1e-12, "{after:?} against {before:?}");
    }
    assert_eq!(
        out.image_set.images[7].intrinsic_adjustment.rolling_shutter,
        [0.0; 3]
    );

    let spread = report.cameras[0]
        .image_adjustments
        .as_ref()
        .expect("a per-image release");
    assert_eq!(spread.images, CORRECTION_IMAGES - 1);
    assert!(spread.radial_bspline_mean.iter().all(|m| m.abs() < 1e-12));
    assert!(spread.rolling_shutter_mean.iter().all(|m| m.abs() < 1e-6));
    assert!(spread.radial_bspline_std.iter().all(|&s| s > 0.0));
    assert!(spread.rolling_shutter_std.iter().all(|&s| s > 1e-3));
    assert!(spread.prior.radial_bspline.is_finite() && spread.prior.rolling_shutter.is_finite());
    assert!(spread.prior.focal.is_nan() && spread.prior.principal_point.is_nan());
    // The standard deviation is the written-back corrections' own.
    let posed: Vec<_> = (0..CORRECTION_IMAGES).filter(|&i| i != 7).collect();
    let ws: Vec<f64> = posed
        .iter()
        .map(|&i| out.image_set.images[i].intrinsic_adjustment.rolling_shutter[0])
        .collect();
    let mean = ws.iter().sum::<f64>() / ws.len() as f64;
    let sd = (ws.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / ws.len() as f64).sqrt();
    assert!((spread.rolling_shutter_std[0] - sd).abs() < 1e-15);
}

#[test]
fn an_unposed_image_drops_its_rates_when_the_setting_changes() {
    // The camera stores COLUMNS and its unposed image 7 carries rates read on
    // them. Solved on COLUMNS, the image keeps its rates; solved on ROWS, the
    // setting written is rows and the image's rates, which meant something
    // else on columns, are zero.
    let mut source = one_camera_source(RollingShutterScanlines::Columns, 145);
    source.image_set.per_image_adjustments = vec![PerImageAdjustments {
        rolling_shutter: true,
        rolling_shutter_scanlines: Some(RollingShutterScanlines::Columns),
        ..PerImageAdjustments::default()
    }];
    source.image_set.images[7].translation_xyz = Vector3::new(f64::NAN, 0.0, 0.0);
    let rates = [1e-3, -2e-3, 5e-4];
    source.image_set.images[7]
        .intrinsic_adjustment
        .rolling_shutter = rates;
    for (choice, kept) in [
        (RollingShutterScanlinesChoice::Columns, rates),
        (RollingShutterScanlinesChoice::Rows, [0.0; 3]),
    ] {
        let options = BundleAdjustOptions {
            rolling_shutter_scanlines: vec![Some(choice)],
            ..BundleAdjustOptions::uniform(1, CORRECTIONS)
        };
        let (out, _) = bundle_adjust(&source, &options, &Progress::none()).expect("well posed");
        assert_eq!(
            out.image_set
                .camera_per_image_adjustments(0)
                .rolling_shutter_scanlines,
            choice.setting()
        );
        assert_eq!(
            out.image_set.images[7].intrinsic_adjustment.rolling_shutter, kept,
            "{choice:?}"
        );
    }
}

#[test]
fn the_report_names_the_corrections_beyond_two_prior_widths() {
    let source = one_camera_source(RollingShutterScanlines::Rows, 150);
    let options = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![Some(RollingShutterScanlinesChoice::Rows)],
        image_prior: ImagePriorOptions {
            fixed: Some(ImagePriorWidths {
                focal: 0.01,
                principal_point: 0.01,
                radial_bspline: 5e-4,
                rolling_shutter: 5e-4,
            }),
            ..ImagePriorOptions::default()
        },
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    };
    let (out, report) = bundle_adjust(&source, &options, &Progress::none()).expect("well posed");
    let spread = report.cameras[0].image_adjustments.clone().unwrap();
    let names: Vec<&str> = spread
        .near_prior
        .iter()
        .map(|n| n.parameter.name())
        .collect();
    assert!(names.contains(&"radial_bspline"), "{names:?}");
    assert!(names.contains(&"rolling_shutter"), "{names:?}");
    for near in &spread.near_prior {
        let a = out.image_set.images[near.image].intrinsic_adjustment;
        let want = match near.parameter {
            ImageAdjustmentParameter::RadialBspline => {
                (a.radial_bspline[0] - spread.radial_bspline_mean[0])
                    .hypot(a.radial_bspline[1] - spread.radial_bspline_mean[1])
                    / 5e-4
            }
            ImageAdjustmentParameter::RollingShutter => {
                (0..3)
                    .map(|k| (a.rolling_shutter[k] - spread.rolling_shutter_mean[k]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / 5e-4
            }
            other => panic!("{other:?} was not released"),
        };
        assert!((near.widths / want - 1.0).abs() < 1e-9);
    }
}

#[test]
fn a_correction_release_the_camera_cannot_take_is_refused() {
    // The radial correction and the rates on a SIMPLE_PINHOLE, the rates on an
    // SFMTOOL_FISHEYE, and the radial correction on an SFMTOOL_PINHOLE whose
    // domain is not positive and finite, each refused naming the camera.
    let rig = truth_through(vec![pinhole(), spline_fisheye(vec![0.0; 4])], |i| {
        (i % 2) as u32
    });
    let radial = CameraRelease {
        image_radial_bspline: true,
        ..CameraRelease::HELD
    };
    let rates = CameraRelease {
        image_rolling_shutter: true,
        ..CameraRelease::HELD
    };
    let error = |value: &SfmrReconstruction, releases: &[CameraRelease]| {
        bundle_adjust(value, &releasing(releases), &Progress::none())
            .err()
            .expect("refused")
    };
    assert_eq!(
        error(&rig, &[radial, CameraRelease::HELD]),
        BundleAdjustError::ImageRadialBsplineNotReleasable {
            camera: 0,
            camera_model: "SIMPLE_PINHOLE",
        }
    );
    assert_eq!(
        error(&rig, &[CameraRelease::HELD, rates]),
        BundleAdjustError::ImageRollingShutterNotReleasable {
            camera: 1,
            camera_model: "SFMTOOL_FISHEYE",
        }
    );
    let message = error(&rig, &[CameraRelease::HELD, rates]).to_string();
    assert!(
        message.contains("camera 1") && message.contains("SFMTOOL_PINHOLE"),
        "{message}"
    );
    for rho_max in [0.0, f64::INFINITY] {
        let mut value = truth_through(vec![spline_pinhole()], |_| 0);
        if let CameraModel::SfmtoolPinhole {
            bspline_rho_max, ..
        } = &mut value.image_set.cameras[0].model
        {
            *bspline_rho_max = rho_max;
        }
        assert_eq!(
            error(&value, &[radial]),
            BundleAdjustError::ImageRadialBsplineNotReleasable {
                camera: 0,
                camera_model: "SFMTOOL_PINHOLE",
            }
        );
    }
    // A scanlines list one camera short.
    let options = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![None],
        ..BundleAdjustOptions::uniform(2, CameraRelease::HELD)
    };
    assert_eq!(
        bundle_adjust(&rig, &options, &Progress::none()).err(),
        Some(BundleAdjustError::ScanlinesCount {
            choices: 1,
            cameras: 2,
        })
    );
}

#[test]
fn a_scanlines_choice_without_rates_released_changes_nothing() {
    // A solve that releases neither correction is the solve it was before
    // the release existed, whatever scanlines are asked for: no race, no
    // setting written.
    let source = one_camera_source(RollingShutterScanlines::Rows, 160);
    let release = CameraRelease {
        image_focal: true,
        image_principal_point: true,
        ..CameraRelease::HELD
    };
    let plain = BundleAdjustOptions::uniform(1, release);
    let asked = BundleAdjustOptions {
        rolling_shutter_scanlines: vec![Some(RollingShutterScanlinesChoice::Unknown)],
        ..BundleAdjustOptions::uniform(1, release)
    };
    let (a, ra) = bundle_adjust(&source, &plain, &Progress::none()).expect("well posed");
    let (b, rb) = bundle_adjust(&source, &asked, &Progress::none()).expect("well posed");
    same_bits(&state_bits(&a), &state_bits(&b), "value");
    same_bits(&report_bits(&ra), &report_bits(&rb), "report");
    assert_eq!(rb.cameras[0].scanlines_race, None);
    assert_eq!(rb.cameras[0].rolling_shutter_scanlines, None);
    assert_eq!(
        b.image_set
            .camera_per_image_adjustments(0)
            .rolling_shutter_scanlines,
        None
    );
}

#[test]
fn a_cancelled_race_stops_both_solves() {
    let source = one_camera_source(RollingShutterScanlines::Rows, 170);
    let collector = Collector::cancelling();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink).cancelled_by(&collector.cancel);
    let error = bundle_adjust(
        &source,
        &BundleAdjustOptions::uniform(1, CORRECTIONS),
        &progress,
    )
    .err()
    .expect("the sink stopped it");
    assert_eq!(error, BundleAdjustError::Cancelled);
    // Both solves started at once, and each left its phase: the
    // cancellation stopped both, and nothing was written back.
    assert_eq!(
        race_phases(&collector),
        ["solve with rows", "solve with columns"]
    );
    let left: Vec<_> = collector
        .left()
        .into_iter()
        .map(|(name, _, _)| name)
        .filter(|name| name.starts_with("solve with"))
        .collect();
    assert_eq!(left, ["solve with rows", "solve with columns"]);
    assert!(collector
        .entered()
        .iter()
        .all(|(name, _)| *name != "write back"));

    // With the early stop off the solves run one after the other, and the
    // second never starts.
    let collector = Collector::cancelling();
    let sink = |event: Event<'_>| collector.push(event);
    let progress = Progress::to(&sink).cancelled_by(&collector.cancel);
    let error = bundle_adjust(&source, &full_solves(), &progress)
        .err()
        .expect("the sink stopped it");
    assert_eq!(error, BundleAdjustError::Cancelled);
    assert_eq!(race_phases(&collector), ["solve with rows"]);
}

/// Both corrections released per image with the early stop off: two full
/// solves, one after the other.
fn full_solves() -> BundleAdjustOptions {
    BundleAdjustOptions {
        scanlines_early_stop: ScanlinesEarlyStopOptions {
            enabled: false,
            ..ScanlinesEarlyStopOptions::default()
        },
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    }
}

/// The report as text, every float in it spelled exactly.
fn report_text(report: &BundleAdjustReport) -> String {
    format!("{report:?}")
}

#[test]
fn the_stop_falls_where_the_measurement_replayed_it() {
    // The measurement's synthetic scenes at the default rate spread are
    // these: the rule stops columns at round 2, iteration 4 with rows
    // planted, and rows at round 2, iteration 2 with columns planted
    // (bundle-adjust-measurements.md, "What the rule would do").
    for (n, planted, stopped, iteration) in [
        (
            0,
            RollingShutterScanlines::Rows,
            RollingShutterScanlines::Columns,
            4,
        ),
        (
            1,
            RollingShutterScanlines::Columns,
            RollingShutterScanlines::Rows,
            2,
        ),
    ] {
        let source = one_camera_source(planted, 70 + 10 * n as u64);
        let (_, report) = bundle_adjust(
            &source,
            &BundleAdjustOptions::uniform(1, CORRECTIONS),
            &Progress::none(),
        )
        .expect("well posed");
        let stop = report.cameras[0].scanlines_race.unwrap().stop.unwrap();
        assert_eq!(
            (stop.stopped, stop.round, stop.iteration),
            (stopped, 2, iteration),
            "{planted:?}: {stop:?}"
        );
    }
}

#[test]
fn the_winner_of_a_stopped_race_is_the_solve_it_would_be_alone() {
    // The rule stops the trailing solve, and the answer is, to the bit, the
    // one two full solves give, and the one a solve given the winner's
    // setting gives: holding the winner in its progress sink changed nothing
    // it computed.
    for (n, planted) in BOTH.into_iter().enumerate() {
        let source = one_camera_source(planted, 70 + 10 * n as u64);
        let (raced, raced_report) = bundle_adjust(
            &source,
            &BundleAdjustOptions::uniform(1, CORRECTIONS),
            &Progress::none(),
        )
        .expect("well posed");
        assert!(
            raced_report.cameras[0]
                .scanlines_race
                .unwrap()
                .stopped_early
        );
        let (full, full_report) =
            bundle_adjust(&source, &full_solves(), &Progress::none()).expect("well posed");
        let given = BundleAdjustOptions {
            rolling_shutter_scanlines: vec![Some(planted.into())],
            ..BundleAdjustOptions::uniform(1, CORRECTIONS)
        };
        let (alone, alone_report) =
            bundle_adjust(&source, &given, &Progress::none()).expect("well posed");
        same_bits(
            &state_bits(&raced),
            &state_bits(&full),
            "against two full solves",
        );
        same_bits(
            &state_bits(&raced),
            &state_bits(&alone),
            "against one solve",
        );
        // The reports differ in the race alone.
        let mut without_race = raced_report.clone();
        without_race.cameras[0].scanlines_race = None;
        let mut full_without = full_report.clone();
        full_without.cameras[0].scanlines_race = None;
        assert_eq!(report_text(&without_race), report_text(&full_without));
        assert_eq!(report_text(&without_race), report_text(&alone_report));
        // The two full solves chose the same setting, by its final costs.
        let full_race = full_report.cameras[0].scanlines_race.unwrap();
        assert!(full_race.stop.is_none() && !full_race.stopped_early);
    }
}

#[test]
fn a_race_the_rule_never_stops_is_two_full_solves_to_the_bit() {
    let source = one_camera_source(RollingShutterScanlines::Rows, 70);
    let never = BundleAdjustOptions {
        scanlines_early_stop: ScanlinesEarlyStopOptions {
            gap_threshold: f64::INFINITY,
            ..ScanlinesEarlyStopOptions::default()
        },
        ..BundleAdjustOptions::uniform(1, CORRECTIONS)
    };
    let collector = Collector::recording();
    let sink = |event: Event<'_>| collector.push(event);
    let (raced, raced_report) =
        bundle_adjust(&source, &never, &Progress::to(&sink)).expect("well posed");
    let (full, full_report) =
        bundle_adjust(&source, &full_solves(), &Progress::none()).expect("well posed");
    same_bits(&state_bits(&raced), &state_bits(&full), "value");
    assert_eq!(report_text(&raced_report), report_text(&full_report));
    assert!(raced_report.cameras[0]
        .scanlines_race
        .unwrap()
        .stop
        .is_none());
    // The columns solve's events are held until the rows solve has closed
    // its phase, so each solve's phase reads whole, rows first.
    let solves: Vec<String> = collector
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Owned::Enter(name, _) if name.starts_with("solve with") => Some(format!(">{name}")),
            Owned::Leave(name, _, _) if name.starts_with("solve with") => Some(format!("<{name}")),
            _ => None,
        })
        .collect();
    assert_eq!(
        solves,
        [
            ">solve with rows",
            "<solve with rows",
            ">solve with columns",
            "<solve with columns"
        ]
    );
}

#[test]
fn a_stopped_race_stops_at_the_same_place_every_time() {
    // The rule reads the solves' values, never the threads' timing.
    let source = one_camera_source(RollingShutterScanlines::Columns, 80);
    let runs: Vec<(Vec<u64>, String)> = (0..4)
        .map(|_| {
            let (out, report) = bundle_adjust(
                &source,
                &BundleAdjustOptions::uniform(1, CORRECTIONS),
                &Progress::none(),
            )
            .expect("well posed");
            (state_bits(&out), report_text(&report))
        })
        .collect();
    for run in &runs[1..] {
        same_bits(&run.0, &runs[0].0, "value");
        assert_eq!(run.1, runs[0].1);
    }
}

#[test]
fn a_scanlines_choice_reads_and_writes_its_names() {
    for choice in [
        RollingShutterScanlinesChoice::Rows,
        RollingShutterScanlinesChoice::Columns,
        RollingShutterScanlinesChoice::Unknown,
    ] {
        assert_eq!(
            RollingShutterScanlinesChoice::parse(choice.name()),
            Ok(choice)
        );
    }
    assert!(RollingShutterScanlinesChoice::parse("rows").is_err());
    assert_eq!(RollingShutterScanlinesChoice::Unknown.setting(), None);
    assert_eq!(
        RollingShutterScanlinesChoice::from(RollingShutterScanlines::Columns),
        RollingShutterScanlinesChoice::Columns
    );
}

/// Writes one-camera scenes with rates planted on rows and on columns at a few
/// rate spreads, for the `scanlines_race` example to race. Run by hand, with
/// `SCANLINES_RACE_SCENES` naming the directory to write them into:
/// `cargo test -p sfmtool-core --lib write_scanlines_race_scenes -- --ignored`.
/// The directory needs a `.sfm-workspace.json` for the example to load them.
#[test]
#[ignore = "writes the scanlines race measurement's scenes; run by hand"]
fn write_scanlines_race_scenes() {
    let dir = std::path::PathBuf::from(
        std::env::var("SCANLINES_RACE_SCENES").expect("SCANLINES_RACE_SCENES names a directory"),
    );
    for tau_w in [4e-3, 1e-3, 3e-4, 1e-4] {
        for (n, planted) in BOTH.into_iter().enumerate() {
            let salt = 70 + 10 * n as u64;
            let draws = correction_draws(CORRECTION_IMAGES, 3e-3, tau_w, salt);
            let mut source = corrections_source(vec![spline_pinhole()], |_| 0, &draws, &[planted]);
            // The format needs each point's patch frame beside embedded
            // observations; the solve does not read it.
            let n_pt = source.point_set.points.len();
            let frame = |axis: usize| {
                Array2::from_shape_fn((n_pt, 3), |(_, k)| if k == axis { 0.01f32 } else { 0.0 })
            };
            source.point_set.patch_u_halfvec_xyz = Some(frame(0));
            source.point_set.patch_v_halfvec_xyz = Some(frame(1));
            // And every keypoint inside its image, which the noise can push a
            // keypoint at the edge out of by a fraction of a pixel.
            if let ObservationSource::EmbeddedPatches { keypoints_xy, .. } =
                &mut source.point_set.observations
            {
                for mut row in keypoints_xy.rows_mut() {
                    row[0] = row[0].clamp(0.0, 639.99);
                    row[1] = row[1].clamp(0.0, 479.99);
                }
            }
            let name = format!("planted_{}_rates_{tau_w:e}.sfmr", planted.as_str());
            source.save(&dir.join(name)).expect("write a scene");
        }
    }
}
