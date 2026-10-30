// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Records the cost trajectories of the two solves of a scanlines race, for
//! measuring a rule that would stop the trailing solve early. It runs the
//! reconstruction-level bundle adjustment with every camera's scanlines
//! `UNKNOWN`, so each camera that releases its rates is raced rows against
//! columns with both solves run to the end, and keeps what the kernel reports
//! through its `Progress` for each schedule round of each solve: the
//! observations the round kept, its noise level, and the cost its damping
//! ladder compares before its first iteration and after each. The race's own
//! comparison is kept beside them. The race's early stop is off, so both
//! solves run to the end, one after the other, unless `--early-stop` turns
//! it on: the race then runs as the adjustment runs it by default, the
//! stopped solve's trajectory ends where it was stopped, the JSON's `stop`
//! says where, and only the whole run's `seconds` is a wall time (the
//! columns solve's events reach the sink after the rows solve's). `scripts/scanlines_race_early_stop.py`
//! evaluates the rule on the output, and
//! `specs/core/reconstruction/bundle-adjust-measurements.md` holds the results.
//!
//! Usage:
//!   scanlines_race <in.sfmr> <out.json> [--pinhole] [--release lens|corrections]
//!     [--schedule default|inf] [--no-cross] [--max-iters N] [--workspace DIR]
//!     [--early-stop]
//!
//! `--pinhole` first switches every camera to `SFMTOOL_PINHOLE`, which the
//! corrections need. `--release lens` (the default) releases each camera's
//! focal and distortion, each image's focal and principal point, and each
//! image's radial correction and rolling-shutter rates; `--release
//! corrections` releases only the last two. `--schedule inf` is one round of
//! trim `inf` and loss scale 1000, which a derived prior runs twice; `default`
//! is the kernel's three rounds. `--no-cross` keeps the free points in their
//! representation. `--workspace` names the workspace whose `.sift` files hold
//! the keypoints of a value without the inline keypoint column, which is read,
//! never written.
//!
//! The output JSON holds, per race (one per camera whose scanlines are
//! raced), the race's `rows_cost`, `columns_cost`, `sigma_px` and
//! `observations`, and under `solves` the rows solve and the columns solve,
//! each a list of rounds `{kept, noise, costs}` with `costs` in px² and the
//! solve's wall time in seconds.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Value};
use sfmtool_core::camera::refit_intrinsics::RefitTarget;
use sfmtool_core::geometry::bundle_adjust::{
    BaSchedule, FreePointPolicy, COST_VALUE, DEFAULT_SCHEDULE, KEPT_VALUE, NOISE_VALUE,
};
use sfmtool_core::progress::{Event, Progress};
use sfmtool_core::reconstruction::bundle_adjust::{
    bundle_adjust, BundleAdjustOptions, CameraRelease, RollingShutterScanlinesChoice,
};
use sfmtool_core::reconstruction::switch_camera_model::{
    switch_camera_model, SwitchCameraModelOptions,
};
use sfmtool_core::SfmrReconstruction;

/// One schedule round of one solve, as its progress reported it.
#[derive(Default)]
struct Round {
    kept: Option<f64>,
    noise: Option<f64>,
    costs: Vec<f64>,
}

/// One solve of a race: its rounds and its wall time.
#[derive(Default)]
struct Solve {
    rounds: Vec<Round>,
    seconds: f64,
}

/// What the sink has seen: the races so far, two solves each, and which solve
/// is running (`0` rows, `1` columns).
#[derive(Default)]
struct Seen {
    races: Vec<[Solve; 2]>,
    running: Option<usize>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let out_path = PathBuf::from(&args[2]);
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let flag = |name: &str| args.iter().any(|a| a == name);

    let mut recon = SfmrReconstruction::load(&path, &Progress::none()).expect("load");
    if let Some(ws) = opt("--workspace") {
        recon.workspace_dir = PathBuf::from(ws);
    }
    let cameras: Vec<usize> = (0..recon.image_set.cameras.len()).collect();
    if flag("--pinhole") {
        let (switched, _) = switch_camera_model(
            &recon,
            &cameras,
            &RefitTarget::SfmtoolPinhole { coeff_count: None },
            &SwitchCameraModelOptions {
                outermost_keypoint: false,
                ..SwitchCameraModelOptions::default()
            },
        )
        .expect("switch to SFMTOOL_PINHOLE");
        recon = switched;
    }
    let release = match opt("--release").as_deref().unwrap_or("lens") {
        "lens" => CameraRelease {
            focal: true,
            distortion: true,
            image_focal: true,
            image_principal_point: true,
            image_radial_bspline: true,
            image_rolling_shutter: true,
            ..CameraRelease::HELD
        },
        "corrections" => CameraRelease {
            image_radial_bspline: true,
            image_rolling_shutter: true,
            ..CameraRelease::HELD
        },
        other => panic!("--release is lens or corrections, not {other}"),
    };
    let schedule_name = opt("--schedule").unwrap_or_else(|| "default".into());
    let schedule = match schedule_name.as_str() {
        "default" => DEFAULT_SCHEDULE.to_vec(),
        "inf" => vec![BaSchedule {
            trim_px: f64::INFINITY,
            loss_scale: 1000.0,
        }],
        other => panic!("--schedule is default or inf, not {other}"),
    };
    let mut options = BundleAdjustOptions::uniform(cameras.len(), release);
    options.rolling_shutter_scanlines =
        vec![Some(RollingShutterScanlinesChoice::Unknown); cameras.len()];
    options.schedule = schedule;
    options.scanlines_early_stop.enabled = flag("--early-stop");
    if flag("--no-cross") {
        options.free_points = FreePointPolicy::NO_CROSS;
    }
    if let Some(n) = opt("--max-iters") {
        options.max_iters = n.parse().expect("--max-iters is a count");
    }

    let seen = Mutex::new(Seen::default());
    let started = Mutex::new(Instant::now());
    let sink = |event: Event<'_>| {
        let mut seen = seen.lock().unwrap();
        match event {
            Event::Enter { phase, .. } if phase.starts_with("solve with ") => {
                let s = usize::from(phase == "solve with columns");
                if s == 0 {
                    seen.races.push(Default::default());
                }
                seen.running = Some(s);
                *started.lock().unwrap() = Instant::now();
            }
            Event::Leave { phase, .. } if phase.starts_with("solve with ") => {
                let s = seen.running.take().expect("a running solve");
                let seconds = started.lock().unwrap().elapsed().as_secs_f64();
                seen.races.last_mut().unwrap()[s].seconds = seconds;
            }
            Event::Enter { phase: "round", .. } => {
                if let Some(s) = seen.running {
                    seen.races.last_mut().unwrap()[s]
                        .rounds
                        .push(Round::default());
                }
            }
            Event::Value { name, value } => {
                let Some(s) = seen.running else { return };
                let round = seen.races.last_mut().unwrap()[s]
                    .rounds
                    .last_mut()
                    .expect("a value inside a round");
                match name {
                    KEPT_VALUE => round.kept = Some(value),
                    NOISE_VALUE => round.noise = Some(value),
                    COST_VALUE => round.costs.push(value),
                    _ => {}
                }
            }
            _ => {}
        }
    };
    let t0 = Instant::now();
    let (_, report) = bundle_adjust(&recon, &options, &Progress::to(&sink)).expect("adjust");
    let seconds = t0.elapsed().as_secs_f64();

    let seen = seen.into_inner().unwrap();
    let raced: Vec<_> = report
        .cameras
        .iter()
        .filter_map(|c| c.scanlines_race.map(|race| (c.camera, race)))
        .collect();
    assert_eq!(raced.len(), seen.races.len(), "one race per raced camera");
    let solve_json = |solve: &Solve| -> Value {
        json!({
            "seconds": solve.seconds,
            "rounds": solve
                .rounds
                .iter()
                .map(|r| json!({"kept": r.kept, "noise": r.noise, "costs": r.costs}))
                .collect::<Vec<_>>(),
        })
    };
    let races: Vec<Value> = raced
        .iter()
        .zip(&seen.races)
        .map(|((camera, race), solves)| {
            json!({
                "camera": camera,
                "rows_cost": race.rows_cost,
                "columns_cost": race.columns_cost,
                "sigma_px": race.sigma_px,
                "observations": race.observations,
                "stop": race.stop.map(|stop| json!({
                    "stopped": stop.stopped.as_str(),
                    "round": stop.round,
                    "iteration": stop.iteration,
                    "gap": stop.gap,
                    "winner_cost": stop.winner_cost,
                })),
                "solves": {"rows": solve_json(&solves[0]), "columns": solve_json(&solves[1])},
            })
        })
        .collect();
    let out = json!({
        "input": path.file_name().map(|n| n.to_string_lossy().to_string()),
        "pinhole": flag("--pinhole"),
        "release": opt("--release").unwrap_or_else(|| "lens".into()),
        "schedule": schedule_name,
        "max_iters": options.max_iters,
        "cross": options.free_points.cross,
        "early_stop": options.scanlines_early_stop.enabled,
        "observations": report.observations,
        "median_residual_before": report.median_residual_before,
        "median_residual_after": report.median_residual_after,
        "seconds": seconds,
        "races": races,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write");
    eprintln!(
        "{} race(s) in {seconds:.1} s, written to {}",
        races.len(),
        out_path.display()
    );
}
