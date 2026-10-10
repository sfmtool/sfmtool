// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Compares keypoint localization by alignment to the reference render
//! against congealing on a reconstruction whose stored poses are taken as
//! ground truth. Built as it is, it runs this tree's localizer (methods `pd`
//! and `ex`), and CI compiles it so it keeps up with the localizer's
//! interface. Built with `RUSTFLAGS="--cfg congeal"` in a copy of commit
//! `83ffb08e`, before congealing was removed, it runs that commit's
//! congealing localizer (method `congeal`) instead. How to build both and
//! report on the output is in `scripts/keypoint_localization/README.md`.
//!
//! Usage:
//!   align_vs_congealing <in.sfmr> <out.json> [--ref-mode displaced|stored]
//!     [--disp 0,0.5,1,2,3] [--min-views 3] [--workspace DIR] [--threads N]
//!     [--references FILE] [--methods pd,ex] [--search S] [--top N]
//!     [--bins LO-HI:N,...] [--default-gates]
//!
//! `--default-gates` keeps the localizer's agreement gates at their defaults
//! (they are off otherwise), for timing what they cost.
//! `--methods` runs a subset of the methods, `--search` sets the localizer's
//! search radius, `--top N` keeps the `N` longest tracks, and `--bins` keeps
//! `N` tracks spread evenly over the eligible points in each track-length
//! range `LO-HI`. Only the images the kept tracks observe are decoded.
//!
//! For every point with at least `--min-views` track views, each view's ground
//! truth (GT) keypoint is the projection of the patch centre at the stored pose.
//! At displacement 0 every view starts at its stored keypoint; at displacement
//! `d > 0` a view starts at its GT keypoint plus `d` px in a direction fixed by
//! a hash of (point, displacement index, slot). With `--ref-mode stored` the
//! reference view keeps its stored keypoint at every displacement; with
//! `--ref-mode displaced` it is displaced like every other view.
//!
//! The output JSON is `{"references": {point: slot}, "points": [...]}`. Each
//! point lists one run per (displacement, method) with, per kept view, its
//! slot in the track, `err` (px from the GT keypoint), `zncc` (the score the
//! localizer reports: the plain score against the template for alignment, the
//! leave-one-out score for congealing) and `retri` (the residual px after
//! re-triangulating the kept views' keypoints at the stored poses). Built as
//! it is, each kept view also carries `pz` and `bz`: its plain and
//! blur-matched scores against the reference render, read as the bench reads a
//! row against the stored bitmap (the view's tile rendered at its final
//! keypoint, scored by `BitmapScorer` against the reference's tile at its
//! keypoint), `1` for the reference itself.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use nalgebra::{Matrix3, Vector3};
use serde_json::{json, Value};
use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::camera::{CameraIntrinsics, PhotographCache};
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
#[cfg(not(congeal))]
use sfmtool_core::patch::keypoint_localize::SearchStrategy;
use sfmtool_core::patch::keypoint_localize::{
    try_localize_patch_keypoints, KeypointLocalization, KeypointLocalizeParams,
};
use sfmtool_core::patch::normal_refine::ProjectedImage;
#[cfg(not(congeal))]
use sfmtool_core::patch::reference_view::render_view_tile;
#[cfg(not(congeal))]
use sfmtool_core::patch::stored_bitmap::{bitmap_from_tile, bitmap_planes, BitmapScorer};
use sfmtool_core::patch::PatchCloud;
use sfmtool_core::progress::Progress;
use sfmtool_core::SfmrReconstruction;

#[cfg(congeal)]
const METHODS: &[&str] = &["congeal"];
#[cfg(not(congeal))]
const METHODS: &[&str] = &["pd", "ex"];

/// Runs one method and returns the localization and the per-view score it
/// reports.
#[cfg(congeal)]
fn run_method(
    _method: &str,
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    _reference: Option<usize>,
    base: &KeypointLocalizeParams,
) -> (KeypointLocalization, Vec<f64>) {
    let loc =
        try_localize_patch_keypoints(patch, views, view_set, Some(seeds), base, &Progress::none())
            .unwrap();
    let z = loc.loo_zncc.clone();
    (loc, z)
}

#[cfg(not(congeal))]
fn run_method(
    method: &str,
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    reference: Option<usize>,
    base: &KeypointLocalizeParams,
) -> (KeypointLocalization, Vec<f64>) {
    let mut p = base.clone();
    p.search_strategy = match method {
        "ex" => SearchStrategy::Exhaustive,
        _ => SearchStrategy::PlusDescent,
    };
    let loc = try_localize_patch_keypoints(
        patch,
        views,
        view_set,
        Some(seeds),
        reference,
        &p,
        &Progress::none(),
    )
    .unwrap();
    let z = loc.zncc.clone();
    (loc, z)
}

/// Per kept view, its plain and blur-matched scores against the reference's
/// render, as the bench scores a row against the stored bitmap: each tile
/// rendered at the view's final keypoint and read by [`BitmapScorer`] against
/// the reference's tile. `None` for every view where there is no reference.
#[cfg(not(congeal))]
fn bitmap_scores(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    loc: &KeypointLocalization,
    base: &KeypointLocalizeParams,
) -> Vec<Option<(f64, f64)>> {
    let r = base.resolution as usize;
    let tile = |k: usize| {
        render_view_tile(
            patch,
            &views[loc.views[k] as usize],
            Some(loc.keypoints[k]),
            r,
            base.sampler,
            &Progress::none(),
        )
    };
    let Some(reference) = loc
        .reference
        .and_then(|im| loc.views.iter().position(|&v| v == im))
    else {
        return vec![None; loc.views.len()];
    };
    let planes = bitmap_planes(&bitmap_from_tile(&tile(reference)), r);
    let mut scorer = BitmapScorer::new(&planes, base.window);
    (0..loc.views.len())
        .map(|k| {
            if k == reference {
                return Some((1.0, 1.0));
            }
            let s = scorer.score(&tile(k).planes(), None);
            Some((s.plain_zncc, s.blur_matched_zncc))
        })
        .collect()
}

/// The branch's reference-view rule pick at the stored keypoints, as a slot.
#[cfg(not(congeal))]
fn rule_pick(
    patch: &OrientedPatch,
    views: &[ProjectedImage<'_>],
    view_set: &[u32],
    seeds: &[Option<[f64; 2]>],
    base: &KeypointLocalizeParams,
) -> Option<usize> {
    let loc = try_localize_patch_keypoints(
        patch,
        views,
        view_set,
        Some(seeds),
        None,
        base,
        &Progress::none(),
    )
    .unwrap();
    loc.reference
        .and_then(|im| view_set.iter().position(|&v| v == im))
}

/// Congealing has no reference rule; without `--references` and a stored
/// reference the point has no reference slot.
#[cfg(congeal)]
fn rule_pick(
    _patch: &OrientedPatch,
    _views: &[ProjectedImage<'_>],
    _view_set: &[u32],
    _seeds: &[Option<[f64; 2]>],
    _base: &KeypointLocalizeParams,
) -> Option<usize> {
    None
}

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

/// Deterministic value in [0, 1) from two integers.
fn hash(a: u64, b: u64) -> f64 {
    let mut x = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_mul(0xC2B2AE3D27D4EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 29;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// Re-triangulates a point from keypoints at the given poses (Gauss-Newton with
/// a numeric Jacobian) and returns the per-view reprojection residuals in px.
fn retriangulate(
    x0: Vector3<f64>,
    obs: &[(&CameraIntrinsics, &RigidTransform, [f64; 2])],
) -> Option<Vec<f64>> {
    if obs.len() < 2 {
        return None;
    }
    let res = |x: &Vector3<f64>| -> Option<Vec<f64>> {
        let mut r = Vec::with_capacity(obs.len() * 2);
        for (cam, pose, kp) in obs {
            let p = cam.project_homogeneous(pose, *x, 1.0)?;
            r.push(p[0] - kp[0]);
            r.push(p[1] - kp[1]);
        }
        Some(r)
    };
    let mut x = x0;
    let scale = 1e-6 * x0.norm().max(1.0);
    for _ in 0..15 {
        let r0 = res(&x)?;
        let mut jt = vec![[0.0f64; 3]; r0.len()];
        for a in 0..3 {
            let mut xp = x;
            xp[a] += scale;
            let r1 = res(&xp)?;
            for k in 0..r0.len() {
                jt[k][a] = (r1[k] - r0[k]) / scale;
            }
        }
        let mut h = Matrix3::<f64>::zeros();
        let mut g = Vector3::<f64>::zeros();
        for k in 0..r0.len() {
            let j = Vector3::new(jt[k][0], jt[k][1], jt[k][2]);
            h += j * j.transpose();
            g += j * r0[k];
        }
        let step = h.try_inverse()? * g;
        x -= step;
        if step.norm() < 1e-9 * x.norm().max(1.0) {
            break;
        }
    }
    let r = res(&x)?;
    Some(
        (0..obs.len())
            .map(|k| r[2 * k].hypot(r[2 * k + 1]))
            .collect(),
    )
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
    let min_views: usize = opt("--min-views").and_then(|s| s.parse().ok()).unwrap_or(3);
    let disps: Vec<f64> = opt("--disp")
        .unwrap_or_else(|| "0,0.5,1,2,3".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let ref_mode = opt("--ref-mode").unwrap_or_else(|| "displaced".into());
    assert!(
        ref_mode == "displaced" || ref_mode == "stored",
        "--ref-mode is displaced or stored"
    );
    let displace_reference = ref_mode == "displaced";
    let top: Option<usize> = opt("--top").and_then(|s| s.parse().ok());
    let search_opt: Option<f64> = opt("--search").and_then(|s| s.parse().ok());
    let default_gates = args.iter().any(|a| a == "--default-gates");
    let method_filter: Option<Vec<String>> =
        opt("--methods").map(|s| s.split(',').map(String::from).collect());
    let methods: Vec<&str> = METHODS
        .iter()
        .copied()
        .filter(|m| {
            method_filter
                .as_ref()
                .is_none_or(|f| f.iter().any(|x| x == m))
        })
        .collect();
    let threads: usize = opt("--threads").and_then(|s| s.parse().ok()).unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    // Reference slots from an earlier run's output, keyed by point index.
    let given_refs: Option<HashMap<usize, Option<usize>>> = opt("--references").map(|f| {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        v["references"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, s)| (k.parse().unwrap(), s.as_u64().map(|s| s as usize)))
            .collect()
    });

    let t0 = Instant::now();
    let mut recon = SfmrReconstruction::load(&path, &Progress::none()).expect("load");
    if let Some(ws) = opt("--workspace") {
        recon.workspace_dir = PathBuf::from(ws);
    }
    let images = &recon.image_table.images;
    let poses: Vec<RigidTransform> = images.iter().map(pose_of).collect();
    let cams: Vec<&CameraIntrinsics> = images
        .iter()
        .map(|i| &recon.image_table.cameras[i.camera_index as usize])
        .collect();
    let cloud = PatchCloud::from_stored_frames(&recon).expect("patch frames");
    let r = recon
        .point_set
        .patch_bitmaps_y_x_rgba
        .as_ref()
        .map_or(24, |b| b.shape()[1]) as u32;
    let offsets = &recon.point_set.observation_offsets;
    let tracks = &recon.point_set.tracks;
    let kps = recon.keypoints_xy().expect("inline keypoints");
    let stored_refs = recon.point_set.reference_observations.clone();
    eprintln!(
        "loaded {} images, {} points in {:.1}s; stored references: {}",
        images.len(),
        recon.point_count(),
        t0.elapsed().as_secs_f64(),
        stored_refs.is_some()
    );

    let mut sel: Vec<usize> = cloud
        .point_indexes
        .iter()
        .enumerate()
        .filter(|(_, &pid)| {
            let p = pid as usize;
            (offsets[p + 1] - offsets[p]) as usize >= min_views
        })
        .map(|(ci, _)| ci)
        .collect();
    eprintln!("{} points with >= {min_views} views, R = {r}", sel.len());
    if let Some(n) = top {
        // Keep the `n` longest tracks.
        let len = |ci: usize| {
            let p = cloud.point_indexes[ci] as usize;
            offsets[p + 1] - offsets[p]
        };
        sel.sort_by_key(|&ci| std::cmp::Reverse(len(ci)));
        sel.truncate(n);
        sel.sort();
        eprintln!("kept the {} longest tracks", sel.len());
    }
    if let Some(spec) = opt("--bins") {
        // `lo-hi:N,...`: per track-length range, N tracks spread evenly over the
        // eligible points (in point order).
        let len = |ci: usize| {
            let p = cloud.point_indexes[ci] as usize;
            (offsets[p + 1] - offsets[p]) as usize
        };
        let mut keep = Vec::new();
        for part in spec.split(',') {
            let (range, cnt) = part.split_once(':').unwrap();
            let (lo, hi) = range.split_once('-').unwrap();
            let (lo, hi, cnt): (usize, usize, usize) = (
                lo.parse().unwrap(),
                hi.parse().unwrap(),
                cnt.parse().unwrap(),
            );
            let el: Vec<usize> = sel
                .iter()
                .copied()
                .filter(|&ci| (lo..=hi).contains(&len(ci)))
                .collect();
            let take = cnt.min(el.len());
            for t in 0..take {
                keep.push(el[t * el.len() / take]);
            }
            eprintln!("bin {lo}-{hi}: {} eligible, kept {take}", el.len());
        }
        keep.sort();
        keep.dedup();
        sel = keep;
    }
    // Decode only the images the selected tracks observe; `compact[i]` is image
    // `i`'s index in the views table, `needed[c]` the inverse.
    let mut needed: Vec<usize> = sel
        .iter()
        .flat_map(|&ci| {
            let p = cloud.point_indexes[ci] as usize;
            (offsets[p] as usize..offsets[p + 1] as usize).map(|j| tracks[j].image_index as usize)
        })
        .collect();
    needed.sort();
    needed.dedup();
    let mut compact = vec![u32::MAX; images.len()];
    for (c, &i) in needed.iter().enumerate() {
        compact[i] = c as u32;
    }
    let paths: Vec<PathBuf> = needed
        .iter()
        .map(|&i| recon.workspace_dir.join(&images[i].name))
        .collect();
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let td = Instant::now();
    let cache = PhotographCache::new(0, 12);
    let (pyramids, _) = cache.get_many(&refs, &Progress::none()).expect("decode");
    let pyramids: Vec<Option<Arc<ImageU8Pyramid>>> = pyramids;
    eprintln!(
        "decoded {} images in {:.1}s",
        needed.len(),
        td.elapsed().as_secs_f64()
    );
    // A dense views table indexed by compact image index.
    let views: Vec<ProjectedImage<'_>> = needed
        .iter()
        .enumerate()
        .map(|(c, &i)| ProjectedImage {
            camera: cams[i],
            cam_from_world: &poses[i],
            pyramid: pyramids[c].as_ref().expect("image decoded"),
        })
        .collect();

    // Gates off except the member gate, which stays at its default.
    let mut base = KeypointLocalizeParams {
        resolution: r,
        min_absolute_zncc: 0.0,
        min_relative_zncc: 0.0,
        max_shift_px: 1e9,
        ..Default::default()
    };
    if let Some(sv) = search_opt {
        base.search = sv;
    }
    if default_gates {
        let defaults = KeypointLocalizeParams::default();
        base.min_absolute_zncc = defaults.min_absolute_zncc;
        base.min_relative_zncc = defaults.min_relative_zncc;
    }
    eprintln!("search = {}", base.search);

    use rayon::prelude::*;
    let results: Vec<Value> = sel
        .par_iter()
        .map(|&ci| {
            let patch = &cloud.patches[ci];
            let pid = cloud.point_indexes[ci] as usize;
            let range = offsets[pid] as usize..offsets[pid + 1] as usize;
            let view_set: Vec<u32> = range.clone().map(|j| tracks[j].image_index).collect();
            let view_set_c: Vec<u32> = view_set.iter().map(|&i| compact[i as usize]).collect();
            let stored: Vec<[f64; 2]> = range
                .clone()
                .map(|j| [kps[[j, 0]] as f64, kps[[j, 1]] as f64])
                .collect();
            let gt: Vec<Option<[f64; 2]>> = view_set
                .iter()
                .map(|&i| {
                    cams[i as usize].project_homogeneous(
                        &poses[i as usize],
                        patch.center.coords,
                        patch.w,
                    )
                })
                .collect();
            let stored_seeds: Vec<Option<[f64; 2]>> = stored.iter().map(|&k| Some(k)).collect();
            // The reference: given, else stored, else the rule's pick at the
            // stored keypoints.
            let reference: Option<usize> = match &given_refs {
                Some(m) => m.get(&pid).copied().flatten(),
                None => stored_refs
                    .as_ref()
                    .and_then(|v| usize::try_from(v[pid]).ok())
                    .or_else(|| rule_pick(patch, &views, &view_set_c, &stored_seeds, &base)),
            };
            let mut runs = Vec::new();
            for (di, &d) in disps.iter().enumerate() {
                let seeds: Vec<Option<[f64; 2]>> = (0..view_set.len())
                    .map(|k| {
                        if d == 0.0 || (!displace_reference && Some(k) == reference) {
                            return Some(stored[k]);
                        }
                        let g = gt[k]?;
                        let th = 2.0
                            * std::f64::consts::PI
                            * hash(pid as u64 * 7919 + di as u64, k as u64);
                        Some([g[0] + d * th.cos(), g[1] + d * th.sin()])
                    })
                    .collect();
                for &m in &methods {
                    let t = Instant::now();
                    let (loc, z) =
                        run_method(m, patch, &views, &view_set_c, &seeds, reference, &base);
                    let secs = t.elapsed().as_secs_f64();
                    #[cfg(not(congeal))]
                    let scores = bitmap_scores(patch, &views, &loc, &base);
                    let mut per = Vec::new();
                    let mut obs = Vec::new();
                    for (k, &imc) in loc.views.iter().enumerate() {
                        let slot = view_set_c.iter().position(|&v| v == imc).unwrap();
                        let im = view_set[slot];
                        let kp = loc.keypoints[k];
                        obs.push((cams[im as usize], &poses[im as usize], kp));
                        let err = gt[slot].map(|g| (kp[0] - g[0]).hypot(kp[1] - g[1]));
                        let seed_err = seeds[slot]
                            .zip(gt[slot])
                            .map(|(s, g)| (s[0] - g[0]).hypot(s[1] - g[1]));
                        #[allow(unused_mut)]
                        let mut v =
                            json!({"slot": slot, "err": err, "zncc": z[k], "seed_err": seed_err});
                        #[cfg(not(congeal))]
                        if let Some((pz, bz)) = scores[k] {
                            v["pz"] = json!(pz);
                            v["bz"] = json!(bz);
                        }
                        per.push(v);
                    }
                    if let Some(rs) = retriangulate(patch.center.coords, &obs) {
                        for (k, v) in per.iter_mut().enumerate() {
                            v["retri"] = json!(rs[k]);
                        }
                    }
                    runs.push(json!({"disp": d, "method": m, "secs": secs, "views": per}));
                }
            }
            json!({"point": pid, "n": view_set.len(), "reference": reference, "runs": runs})
        })
        .collect();
    let references: serde_json::Map<String, Value> = results
        .iter()
        .map(|p| (p["point"].to_string(), p["reference"].clone()))
        .collect();
    let out = json!({
        "sfmr": path.display().to_string(),
        "ref_mode": ref_mode,
        "methods": methods,
        "search": base.search,
        "seconds": t0.elapsed().as_secs_f64(),
        "references": references,
        "points": results,
    });
    std::fs::write(&out_path, serde_json::to_string(&out).unwrap()).unwrap();
    eprintln!("done in {:.1}s", t0.elapsed().as_secs_f64());
}
