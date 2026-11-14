// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Time and score the pairwise-flow normal on real points.
//!
//! ```text
//! cargo run --release -p sfmtool-core --example pairwise_flow_measure -- \
//!     <file.sfmr> <count> [--points P,Q,...] [--chi2 X] [--threads N]
//! ```
//!
//! Takes `<count>` finite points with three or more observations, evenly
//! spaced through the file's point order, or the points `--points` lists;
//! turns each one's patch to the mean viewing direction of its
//! observations; runs `pairwise_flow_normal` from there at the default
//! options, or with `max_mean_chi2` set by `--chi2`; and prints, per point
//! and in total, the outcome, the angle between the step's normal and the
//! file's stored normal, and the time. `--threads N` runs the step in a rayon pool of `N` threads
//! (the pairs of a round are read in parallel); by default the global pool.
//! The points run one after another, so the time per point is the step's own.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nalgebra::Vector3;
use sfmtool_core::bench::{
    create_track, pairwise_flow_normal, tilt_patch, Bench, CreateTrackOptions, NormalEstimate,
    PairwiseFlowNormalOptions, PairwiseFlowOutcome,
};
use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::camera::{CameraIntrinsics, PhotographCache};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::numeric::median;
use sfmtool_core::patch::normal_refine::ProjectedImage;
use sfmtool_core::progress::Progress;
use sfmtool_core::{EditedReconstruction, SfmrReconstruction};

fn pose_of(image: &sfmtool_core::SfmrImage) -> RigidTransform {
    let q = image.quaternion_wxyz;
    RigidTransform::from_wxyz_translation(
        [q.w, q.i, q.j, q.k],
        [
            image.translation_xyz.x,
            image.translation_xyz.y,
            image.translation_xyz.z,
        ],
    )
}

fn degrees(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = PathBuf::from(&args[1]);
    let wanted: usize = args[2].parse().expect("a point count");
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|k| args[k + 1].clone())
    };
    let threads: Option<usize> = flag("--threads").map(|n| n.parse().expect("a thread count"));
    let listed: Option<Vec<u32>> = flag("--points").map(|s| {
        s.split(',')
            .map(|p| p.parse().expect("a point index"))
            .collect()
    });
    let chi2: Option<f64> = flag("--chi2").map(|x| x.parse().expect("a χ² bar"));

    let recon = SfmrReconstruction::load(&path, &Progress::none()).expect("load");
    let images = &recon.image_set.images;
    let poses: Vec<RigidTransform> = images.iter().map(pose_of).collect();
    let intrinsics = recon.image_set.all_image_intrinsics();
    let cams: Vec<&CameraIntrinsics> = intrinsics.iter().map(|c| c.as_ref()).collect();
    let paths: Vec<PathBuf> = images
        .iter()
        .map(|i| recon.workspace_dir.join(&i.name))
        .collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let decode = Instant::now();
    let cache = PhotographCache::new(0, 12);
    let (pyramids, _) = cache.get_many(&refs, &Progress::none()).expect("decode");
    let pyramids: Vec<Arc<ImageU8Pyramid>> = pyramids
        .into_iter()
        .map(|p| p.expect("every image decodes"))
        .collect();
    eprintln!(
        "decoded {} images in {:.1}s",
        images.len(),
        decode.elapsed().as_secs_f64()
    );
    let views: Vec<ProjectedImage<'_>> = (0..images.len())
        .map(|i| ProjectedImage {
            camera: cams[i],
            cam_from_world: &poses[i],
            pyramid: &pyramids[i],
        })
        .collect();

    let counts = &recon.point_set.observation_counts;
    let eligible: Vec<u32> = (0..recon.point_count() as u32)
        .filter(|&p| recon.point_set.points[p as usize].w != 0.0 && counts[p as usize] >= 3)
        .collect();
    let take = wanted.min(eligible.len());
    let chosen: Vec<u32> = listed.unwrap_or_else(|| {
        (0..take)
            .map(|k| eligible[k * eligible.len() / take])
            .collect()
    });
    let centres: Vec<Vector3<f64>> = poses
        .iter()
        .map(|p| p.inverse_translation_origin().coords)
        .collect();
    let edited = EditedReconstruction::new(Arc::new(recon.clone_for_edit()));
    let defaults = PairwiseFlowNormalOptions::default();
    let options = PairwiseFlowNormalOptions {
        max_mean_chi2: chi2.unwrap_or(defaults.max_mean_chi2),
        ..defaults
    };

    let pool = threads.map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("a pool")
    });
    let mut angles = Vec::new();
    let mut times = Vec::new();
    let mut outcomes = [0usize; 4];
    let started = Instant::now();
    for &p in &chosen {
        let (bench, report) =
            create_track(&Bench::new(), &edited, p, &CreateTrackOptions::default())
                .expect("a finite point");
        let track = (**bench.track(&report.label).expect("the track")).clone();
        let payload = track.track().expect("track stage");
        let patch = payload.patch_frame.as_ref().expect("a patch");
        let stored = patch.normal();
        let mean_view = track
            .in_observations()
            .iter()
            .map(|&k| {
                (centres[track.observations[k].image as usize] - patch.center.coords).normalize()
            })
            .fold(Vector3::zeros(), |s, d| s + d)
            .normalize();
        let (start, _) = tilt_patch(&track, &edited, mean_view).expect("a finite track");
        let t0 = Instant::now();
        let run = || pairwise_flow_normal(&start, &edited, &views, &options, &Progress::none());
        let result = match &pool {
            Some(pool) => pool.install(run),
            None => run(),
        };
        let seconds = t0.elapsed().as_secs_f64();
        times.push(seconds);
        let (_, report) = match result {
            Ok(r) => r,
            Err(e) => {
                println!("point {p}: refused: {e}");
                continue;
            }
        };
        let NormalEstimate::PairwiseFlow(rounds) = &report.estimate else {
            unreachable!("the pairwise flow step reports its rounds");
        };
        let slot = match rounds.outcome {
            PairwiseFlowOutcome::Turned => 0,
            PairwiseFlowOutcome::SinglePair => 1,
            PairwiseFlowOutcome::Undetermined => 2,
            PairwiseFlowOutcome::Unstable => 3,
        };
        outcomes[slot] += 1;
        let start_off = degrees(mean_view, stored);
        let off = rounds.normal.map(|n| degrees(Vector3::from(n), stored));
        if let Some(off) = off {
            angles.push(off);
        }
        let pairs = rounds.rounds.last().map_or(0, |r| r.pairs.len());
        println!(
            "point {p}: {} views, {pairs} pairs, start {start_off:.1} deg, {:?}, {}, {seconds:.2}s",
            counts[p as usize],
            rounds.outcome,
            off.map_or("-".to_string(), |o| format!("{o:.2} deg")),
        );
    }
    let total = started.elapsed().as_secs_f64();
    println!(
        "{} points: turned {}, single pair {}, undetermined {}, unstable {}",
        chosen.len(),
        outcomes[0],
        outcomes[1],
        outcomes[2],
        outcomes[3]
    );
    println!(
        "median angle from the stored normal over the turned: {:.2} deg",
        median(&angles)
    );
    println!(
        "time per point: median {:.2}s, mean {:.2}s; total {total:.1}s; threads {}",
        median(&times),
        total / chosen.len().max(1) as f64,
        threads.map_or("global".to_string(), |n| n.to_string()),
    );
}
