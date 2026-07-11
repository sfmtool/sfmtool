// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for cluster-patch refinement: synthetic warp recovery, the vetting
//! gates, determinism, and AVX2-vs-scalar equivalence.

use super::kernels::{eval_zncc_scalar, SupportTables, TileCache};
use super::*;
use crate::camera::remap::{ImageU8, ImageU8Pyramid};
use ndarray::{Array2, Array3 as NdArray3};

/// Smooth band-limited analytic texture in `[12, 242]` (no u8 clipping).
fn texture(x: f64, y: f64) -> f64 {
    127.0
        + 50.0 * (0.11 * x + 0.06 * y + 1.3).sin()
        + 35.0 * (0.05 * x - 0.12 * y + 0.7).sin()
        + 20.0 * (0.17 * x + 0.13 * y + 2.9).sin()
        + 10.0 * (0.29 * x - 0.23 * y + 0.4).cos()
}

/// Render a 1-channel image whose pixel `(col, row)` holds `f` at the pixel
/// center `(col + 0.5, row + 0.5)` (the shared pixel-center convention).
fn make_image(w: u32, h: u32, f: impl Fn(f64, f64) -> f64) -> ImageU8 {
    let mut data = vec![0u8; (w * h) as usize];
    for row in 0..h {
        for col in 0..w {
            let v = f(col as f64 + 0.5, row as f64 + 0.5);
            data[(row * w + col) as usize] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    ImageU8::new(w, h, 1, data)
}

fn pyramid(img: &ImageU8) -> ImageU8Pyramid {
    ImageU8Pyramid::build(img, 6)
}

/// Owned per-image feature arrays (so tests can build borrowed
/// [`FeatureGeometry`] views).
struct ImageFeatures {
    pos: Array2<f32>,
    aff: NdArray3<f32>,
}

impl ImageFeatures {
    fn new(features: &[([f64; 2], Mat2)]) -> ImageFeatures {
        let n = features.len();
        let mut pos = Array2::zeros((n, 2));
        let mut aff = NdArray3::zeros((n, 2, 2));
        for (i, (p, a)) in features.iter().enumerate() {
            pos[[i, 0]] = p[0] as f32;
            pos[[i, 1]] = p[1] as f32;
            for r in 0..2 {
                for c in 0..2 {
                    aff[[i, r, c]] = a[r][c] as f32;
                }
            }
        }
        ImageFeatures { pos, aff }
    }
}

fn geometry(feats: &[ImageFeatures]) -> Vec<FeatureGeometry<'_>> {
    feats
        .iter()
        .map(|f| FeatureGeometry {
            positions_xy: f.pos.view(),
            affine_shapes: f.aff.view(),
        })
        .collect()
}

fn rot2(deg: f64) -> Mat2 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s], [s, c]]
}

fn matvec(a: &Mat2, x: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}

fn apply23(m: &ndarray::ArrayView2<'_, f64>, x: [f64; 2]) -> [f64; 2] {
    [
        m[[0, 0]] * x[0] + m[[0, 1]] * x[1] + m[[0, 2]],
        m[[1, 0]] * x[0] + m[[1, 1]] * x[1] + m[[1, 2]],
    ]
}

/// Run one synthetic recovery case: image 1 shows [`texture`]; image 2 shows
/// it warped by the known affine `x₂ = A_true·x₁ + t_true`; the member's SIFT
/// seed is perturbed by `(dlog_s, drot_deg, shift_px)`. Asserts the
/// non-reference member is `Kept` and the recovered absolute affine maps the
/// reference's support grid within `max_rmse` px of the truth.
fn run_recovery_case(
    scale: f64,
    rot_deg: f64,
    shear: f64,
    dlog_s: f64,
    drot_deg: f64,
    shift_px: [f64; 2],
    max_rmse: f64,
) -> ClusterRefineResult {
    let params = ClusterRefineParams::default();
    let a_true = mul2(
        &mul2(&[[scale, 0.0], [0.0, scale]], &rot2(rot_deg)),
        &[[1.0, shear], [0.0, 1.0]],
    );
    let pos_ref = [64.0, 64.0];
    let a_ref = [[2.5, 0.0], [0.0, 2.5]];
    let pos_mem_true = [64.0, 64.0];
    let t_true = [
        pos_mem_true[0] - (a_true[0][0] * pos_ref[0] + a_true[0][1] * pos_ref[1]),
        pos_mem_true[1] - (a_true[1][0] * pos_ref[0] + a_true[1][1] * pos_ref[1]),
    ];
    let img1 = make_image(128, 128, texture);
    let a_true_inv = inv2(&a_true);
    let img2 = make_image(128, 128, |x, y| {
        let p = matvec(&a_true_inv, [x - t_true[0], y - t_true[1]]);
        texture(p[0], p[1])
    });

    // Perturbed member seed: `A_mem = A_true · N · A_ref`,
    // `N = e^{Δlog s} R(Δrot)`, position offset by `shift_px`.
    let n_pert = {
        let s = dlog_s.exp();
        let r = rot2(drot_deg);
        [[s * r[0][0], s * r[0][1]], [s * r[1][0], s * r[1][1]]]
    };
    let a_mem = mul2(&a_true, &mul2(&n_pert, &a_ref));
    let pos_mem = [pos_mem_true[0] + shift_px[0], pos_mem_true[1] + shift_px[1]];

    let feats = [
        ImageFeatures::new(&[(pos_ref, a_ref)]),
        ImageFeatures::new(&[(pos_mem, a_mem)]),
    ];
    let pyramids = [pyramid(&img1), pyramid(&img2)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );

    // Reference selection is by scale, so either member can be the
    // reference; evaluate the recovered warp in the matching direction.
    let ref_k = result.reference_members[0];
    assert!(
        ref_k == 0 || ref_k == 1,
        "reference must be a cluster member"
    );
    let other = 1 - ref_k as usize;
    assert_eq!(
        result.member_status[other],
        MemberStatus::Kept,
        "recovered member must vet as Kept (zncc {}, shift {})",
        result.member_zncc[other],
        result.member_shift_px[other],
    );

    // Ground truth in the refined direction (reference → other image).
    let (w_true, t_w) = if ref_k == 0 {
        (a_true, t_true)
    } else {
        let inv = inv2(&a_true);
        let ti = matvec(&inv, t_true);
        (inv, [-ti[0], -ti[1]])
    };
    let (pos_r, a_r) = if ref_k == 0 {
        (pos_ref, a_ref)
    } else {
        (pos_mem, a_mem)
    };

    let rec = result.member_affines.index_axis(ndarray::Axis(0), other);
    let res = params.resolution as usize;
    let step = 2.0 * params.radius / res as f64;
    let off = 0.5 * step - params.radius;
    let mut sq_sum = 0.0;
    for row in 0..res {
        for col in 0..res {
            let u = [col as f64 * step + off, row as f64 * step + off];
            let x = [
                pos_r[0] + a_r[0][0] * u[0] + a_r[0][1] * u[1],
                pos_r[1] + a_r[1][0] * u[0] + a_r[1][1] * u[1],
            ];
            let p = apply23(&rec, x);
            let q = [
                w_true[0][0] * x[0] + w_true[0][1] * x[1] + t_w[0],
                w_true[1][0] * x[0] + w_true[1][1] * x[1] + t_w[1],
            ];
            sq_sum += (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
        }
    }
    let rmse = (sq_sum / (res * res) as f64).sqrt();
    assert!(
        rmse <= max_rmse,
        "support-grid RMSE {rmse:.3} px exceeds {max_rmse} \
         (scale {scale}, rot {rot_deg}, shear {shear}, ref {ref_k})"
    );
    result
}

#[test]
fn synthetic_recovery_across_warp_range() {
    // Warp range per the spec: scale 0.8–1.5×, rotation ≤ 20°, shear ≤ 0.15;
    // seed noise at the experiment-observed levels (|Δlog s| 0.07, |Δrot| 4°,
    // 1 px shift); acceptance 0.3 px support-grid RMSE.
    run_recovery_case(1.0, 0.0, 0.0, 0.07, 4.0, [1.0, 0.0], 0.3);
    run_recovery_case(0.8, -12.0, 0.1, -0.07, -4.0, [0.0, 1.0], 0.3);
    run_recovery_case(1.25, 20.0, 0.0, 0.07, 4.0, [0.7, -0.7], 0.3);
    run_recovery_case(1.5, 8.0, 0.15, -0.05, 3.0, [-0.7, 0.7], 0.3);
    run_recovery_case(0.9, -20.0, -0.15, 0.05, -3.0, [-1.0, 0.0], 0.3);
}

#[test]
fn gate_low_zncc_rejects_flat_member() {
    // The member's image is flat texture: every member channel is windowed-
    // flat, contributes 0 to the score, and the low-ZNCC gate rejects it.
    // (An *unrelated smooth* texture is deliberately not used here: over a
    // ~50-effective-sample Gaussian window the affine optimizer can chase a
    // spurious ZNCC above the permissive 0.85 gate, tripping the shift gate
    // instead — the flat case pins the RejectedLowZncc path
    // deterministically.) The localizability gate is disabled: a flat patch
    // is exactly what it excludes (see gate_unlocalizable_member_excluded),
    // and this test pins the downstream ZNCC path.
    let params = ClusterRefineParams {
        max_keypoint_uncertainty: 0.0,
        ..Default::default()
    };
    let img1 = make_image(128, 128, texture);
    let img2 = make_image(128, 128, |_, _| 127.0);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([64.0, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img1), pyramid(&img2)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    // Equal scales tie-break to the lowest global member index.
    assert_eq!(result.reference_members[0], 0);
    assert_eq!(result.member_status[0], MemberStatus::Reference);
    assert_eq!(result.member_status[1], MemberStatus::RejectedLowZncc);
    assert_eq!(result.member_zncc[1], 0.0);
}

#[test]
fn gate_unlocalizable_member_excluded() {
    // Default params: the flat member's own patch has zero gradients, so its
    // weak-axis positional uncertainty is enormous and the localizability
    // gate excludes it before refinement. With one usable member left the
    // cluster is unrefinable.
    let params = ClusterRefineParams::default();
    let img1 = make_image(128, 128, texture);
    let img2 = make_image(128, 128, |_, _| 127.0);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([64.0, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img1), pyramid(&img2)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[1], MemberStatus::RejectedUnlocalizable);
    assert!(result.member_zncc[1].is_nan());
    assert!(result.member_shift_px[1].is_nan());
    assert_eq!(result.reference_members[0], REFERENCE_UNREFINABLE);
    assert_eq!(result.member_status[0], MemberStatus::NotEvaluated);
}

#[test]
fn gate_unlocalizable_member_cannot_be_reference() {
    // The flat-image member has the largest SIFT scale and would win
    // reference selection, but the gate runs first: the textured members
    // refine normally among themselves.
    let params = ClusterRefineParams::default();
    let img_flat = make_image(128, 128, |_, _| 127.0);
    let img = make_image(128, 128, texture);
    let a_big = [[3.0, 0.0], [0.0, 3.0]];
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a_big)]),
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([64.0, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img_flat), pyramid(&img), pyramid(&img)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 3],
        &[0, 1, 2],
        &[0, 0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[0], MemberStatus::RejectedUnlocalizable);
    // Equal scales among the survivors tie-break to the lowest member index.
    assert_eq!(result.reference_members[0], 1);
    assert_eq!(result.member_status[1], MemberStatus::Reference);
    assert_eq!(result.member_status[2], MemberStatus::Kept);
}

#[test]
fn gate_scores_border_member_with_clamped_sampling() {
    // A member whose patch straddles the image border is still scored — the
    // sampler clamps to the nearest valid pixel instead of skipping the
    // gate. On a flat image the clamped patch is flat, so the member is
    // RejectedUnlocalizable (before this behavior it fell through to the
    // seed frame gate as NotEvaluated).
    let params = ClusterRefineParams::default();
    let img1 = make_image(128, 128, texture);
    let img2 = make_image(128, 128, |_, _| 127.0);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    // The member's support (±10 px around x = 3) leaves the frame.
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([3.0, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img1), pyramid(&img2)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[1], MemberStatus::RejectedUnlocalizable);
    // A textured border member passes the gate and proceeds to the seed
    // frame gate (NotEvaluated), exactly as before.
    let pyramids = [pyramid(&img1), pyramid(&img1)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[1], MemberStatus::NotEvaluated);
}

#[test]
fn gate_shift_rejects_drifted_seed() {
    // Identical images, but the member's detection sits 1.5 px off the truth
    // while the gate allows 0.5 px: the optimizer recovers the offset and the
    // drift gate rejects it.
    let params = ClusterRefineParams {
        max_shift_px: 0.5,
        ..Default::default()
    };
    let img = make_image(128, 128, texture);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([65.5, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img), pyramid(&img)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[1], MemberStatus::RejectedShift);
    assert!(
        result.member_shift_px[1] > 0.5,
        "recovered drift {} should exceed the gate",
        result.member_shift_px[1]
    );
    assert!(
        result.member_zncc[1] > 0.9,
        "the warp itself should be good"
    );
}

#[test]
fn gate_out_of_frame_member_not_evaluated() {
    let params = ClusterRefineParams::default();
    let img = make_image(128, 128, texture);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    // The member's seed support (±10 px around x = 3) leaves the frame.
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([3.0, 64.0], a)]),
    ];
    let pyramids = [pyramid(&img), pyramid(&img)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.member_status[1], MemberStatus::NotEvaluated);
    assert!(result.member_zncc[1].is_nan());
    assert!(result.member_shift_px[1].is_nan());
}

#[test]
fn gate_all_references_out_of_frame_is_unrefinable() {
    let params = ClusterRefineParams::default();
    let img = make_image(128, 128, texture);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([3.0, 64.0], a)]),
        ImageFeatures::new(&[([64.0, 3.0], a)]),
    ];
    let pyramids = [pyramid(&img), pyramid(&img)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.reference_members[0], REFERENCE_UNREFINABLE);
    assert_eq!(result.member_status[0], MemberStatus::NotEvaluated);
    assert_eq!(result.member_status[1], MemberStatus::NotEvaluated);
}

#[test]
fn gate_degenerate_cluster_not_evaluated() {
    // One member has a degenerate affine shape -> fewer than 2 usable members.
    let params = ClusterRefineParams::default();
    let img = make_image(128, 128, texture);
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], [[2.5, 0.0], [0.0, 2.5]])]),
        ImageFeatures::new(&[([64.0, 64.0], [[0.0, 0.0], [0.0, 0.0]])]),
    ];
    let pyramids = [pyramid(&img), pyramid(&img)];
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        &params,
        None,
    );
    assert_eq!(result.reference_members[0], REFERENCE_UNREFINABLE);
    assert_eq!(result.member_status[0], MemberStatus::NotEvaluated);
    assert_eq!(result.member_status[1], MemberStatus::NotEvaluated);
}

#[test]
fn one_kept_member_per_image() {
    // Two members in the same image, both refinable: exactly one Kept, the
    // other DuplicateImage. A third member sharing the reference's image is
    // DuplicateImage without evaluation.
    let params = ClusterRefineParams::default();
    let img = make_image(128, 128, texture);
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let a_small = [[2.4, 0.0], [0.0, 2.4]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a), ([70.0, 70.0], a_small)]),
        ImageFeatures::new(&[([64.0, 64.0], a_small), ([64.5, 64.0], a_small)]),
    ];
    let pyramids = [pyramid(&img), pyramid(&img)];
    // Members: (img0, f0) = reference (largest scale), (img0, f1) shares the
    // reference's image, (img1, f0) and (img1, f1) compete for image 1.
    let result = refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 4],
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &params,
        None,
    );
    assert_eq!(result.reference_members[0], 0);
    assert_eq!(result.member_status[0], MemberStatus::Reference);
    assert_eq!(result.member_status[1], MemberStatus::DuplicateImage);
    let statuses = [result.member_status[2], result.member_status[3]];
    let kept = statuses
        .iter()
        .filter(|&&s| s == MemberStatus::Kept)
        .count();
    let dup = statuses
        .iter()
        .filter(|&&s| s == MemberStatus::DuplicateImage)
        .count();
    assert_eq!((kept, dup), (1, 1), "exactly one kept member per image");
}

#[test]
fn determinism_bit_identical_across_runs() {
    let run = || run_recovery_case(1.25, 20.0, 0.1, 0.07, 4.0, [0.7, -0.7], 0.3);
    let a = run();
    let b = run();
    assert_eq!(a.reference_members, b.reference_members);
    assert_eq!(a.member_status, b.member_status);
    assert_eq!(
        a.member_affines.as_slice().unwrap(),
        b.member_affines.as_slice().unwrap()
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.member_zncc), bits(&b.member_zncc));
    assert_eq!(bits(&a.member_shift_px), bits(&b.member_shift_px));
}

#[test]
fn avx2_matches_scalar_scores() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        eprintln!("skipping: not x86_64");
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            eprintln!("skipping: AVX2+FMA not available");
            return;
        }
        let params = ClusterRefineParams::default();
        let resolution = params.resolution;
        let support = build_support(params.window, resolution);
        let tables = SupportTables::new(&support, resolution);
        let img = make_image(128, 128, texture);
        let pyr = pyramid(&img);
        let geo = MemberGeo {
            k_global: 0,
            image: 0,
            pos: [64.0, 64.0],
            a: [[2.5, 0.0], [0.0, 2.5]],
            scale: 2.5,
        };
        let step = 2.0 * params.radius / resolution as f64;
        let off = 0.5 * step - params.radius;
        let tmpl = build_template(&pyr, &geo, &support, &tables, resolution, step, off).unwrap();

        // Score a spread of perturbed warps through both paths.
        let mut tiles = TileCache::default();
        let cases: [([f64; 2], Mat2); 4] = [
            ([0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]),
            ([1.5, -0.5], [[0.05, 0.02], [-0.03, 0.04]]),
            ([-2.0, 1.0], [[-0.08, 0.0], [0.0, -0.08]]),
            ([0.3, 0.3], [[0.0, 0.12], [-0.12, 0.0]]),
        ];
        for (t, d) in cases {
            let id = [[1.0 + d[0][0], d[0][1]], [d[1][0], 1.0 + d[1][1]]];
            let b = mul2(&id, &geo.a);
            let map = warp_map(geo.pos, t, &b, step, off);
            let level = level_for_map(&map, pyr.num_levels());
            let lmap = map_at_level(&map, level);
            let bbox = super::kernels::grid_bbox(&lmap, resolution);
            let tile = tiles.get_or_build(&pyr, level, bbox).unwrap();
            let a32 = {
                let a = &lmap.a;
                [
                    a[0] as f32,
                    a[1] as f32,
                    (a[2] - 0.5 - tile.x0 as f64) as f32,
                    a[3] as f32,
                    a[4] as f32,
                    (a[5] - 0.5 - tile.y0 as f64) as f32,
                ]
            };
            let scalar = eval_zncc_scalar(a32, tile, &tables, &tmpl).unwrap();
            // SAFETY: feature availability checked above.
            let avx2 =
                unsafe { super::kernels::eval_zncc_avx2(a32, tile, &tables, &tmpl).unwrap() };
            assert!(
                (scalar - avx2).abs() < 1e-4,
                "scalar {scalar} vs avx2 {avx2} diverge"
            );
        }
    }
}
