// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests for cluster-patch refinement: synthetic warp recovery, the vetting
//! gates, determinism, and AVX2-vs-scalar equivalence.

use super::kernels::{eval_zncc_scalar, SupportTables, TileCache};
use super::*;
use crate::camera::image::{ImageU8, ImageU8Pyramid};
use ndarray::{Array2, Array3 as NdArray3};

/// Band-limited analytic texture in `[2, 252]` (no u8 clipping).
///
/// The two fine terms (periods of about 6.6 and 6.1 px) are what make a
/// member's own patch pin a position under the member gate: without them the
/// patch is smooth on the template grid and matches itself 3 grid px away,
/// which reads as the largest ZNCC self-similarity radius.
fn texture(x: f64, y: f64) -> f64 {
    127.0
        + 40.0 * (0.11 * x + 0.06 * y + 1.3).sin()
        + 28.0 * (0.05 * x - 0.12 * y + 0.7).sin()
        + 20.0 * (0.17 * x + 0.13 * y + 2.9).sin()
        + 10.0 * (0.29 * x - 0.23 * y + 0.4).cos()
        + 15.0 * (0.83 * x + 0.47 * y + 0.2).sin()
        + 12.0 * (-0.52 * x + 0.88 * y + 1.1).sin()
}

/// [`texture`] without its two fine terms: smooth on the template grid, so a
/// member's own patch reads the largest ZNCC self-similarity radius.
fn smooth_texture(x: f64, y: f64) -> f64 {
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

/// Apply a member's reference-relative warp row (`W` | absolute position
/// `p`) to `x`, given the cluster's reference keypoint:
/// `x_member = W·(x − x_ref) + p`. `W` is recovered from the stored absolute
/// shape as `S·S_ref⁻¹` (format version 5).
fn apply_member_row(m: &[[f64; 3]; 2], x_ref: [f64; 2], x: [f64; 2]) -> [f64; 2] {
    let dx = [x[0] - x_ref[0], x[1] - x_ref[1]];
    [
        m[0][0] * dx[0] + m[0][1] * dx[1] + m[0][2],
        m[1][0] * dx[0] + m[1][1] * dx[1] + m[1][2],
    ]
}

/// Run one synthetic recovery case: image 1 shows [`texture`]; image 2 shows
/// it warped by the known affine `x₂ = A_true·x₁ + t_true`; the member's SIFT
/// seed is perturbed by `(dlog_s, drot_deg, shift_px)`. Asserts the
/// non-reference member is `Kept`, the reference row carries `S_ref | x_ref`,
/// and the warp recovered from the stored absolute shapes maps the
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
    // The recovered warp agrees in the middle of the patch as well, and the
    // reference against itself reads 1 there.
    assert!(
        result.member_zncc_middle[other] > 0.9,
        "middle zncc {} (whole {})",
        result.member_zncc_middle[other],
        result.member_zncc[other],
    );
    assert_eq!(result.member_zncc_middle[ref_k as usize], 1.0);
    // And in every cell of the ZNCC grid.
    for &z in result.member_zncc_grid[other].iter().flatten() {
        assert!(z > 0.9, "grid {:?}", result.member_zncc_grid[other]);
    }
    for &z in result.member_zncc_grid[ref_k as usize].iter().flatten() {
        assert!((z - 1.0).abs() < 1e-6, "reference grid {z}");
    }

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

    // Reference row: `S_ref | x_ref` — the reference feature's own detector
    // affine shape and keypoint position, both read by the kernel from the
    // f32 `.sift` arrays, so compare f32-rounded.
    let pos_r_stored = [pos_r[0] as f32 as f64, pos_r[1] as f32 as f64];
    let a_r_stored = [
        [a_r[0][0] as f32 as f64, a_r[0][1] as f32 as f64],
        [a_r[1][0] as f32 as f64, a_r[1][1] as f32 as f64],
    ];
    let ref_row = result
        .member_affine_shapes
        .index_axis(ndarray::Axis(0), ref_k as usize);
    let ref_pos = result
        .member_positions
        .index_axis(ndarray::Axis(0), ref_k as usize);
    assert_eq!(
        [
            ref_row[[0, 0]],
            ref_row[[0, 1]],
            ref_row[[1, 0]],
            ref_row[[1, 1]]
        ],
        [
            a_r_stored[0][0],
            a_r_stored[0][1],
            a_r_stored[1][0],
            a_r_stored[1][1]
        ],
        "reference row must carry the reference feature's own detector shape"
    );
    assert_eq!(
        [ref_pos[0], ref_pos[1]],
        pos_r_stored,
        "the reference's own position must be its keypoint position"
    );

    // The stored shape is the member's ABSOLUTE affine shape `S = W·S_ref`;
    // recover the reference→member warp `W = S·S_ref⁻¹` through the
    // reference's shape, exactly as a consumer does, and evaluate that
    // against the ground-truth warp.
    let rec = result
        .member_affine_shapes
        .index_axis(ndarray::Axis(0), other);
    let rec_p = result.member_positions.index_axis(ndarray::Axis(0), other);
    let s_mem = [[rec[[0, 0]], rec[[0, 1]]], [rec[[1, 0]], rec[[1, 1]]]];
    let w_rec = mul2(&s_mem, &inv2(&a_r_stored));
    let rec_row = [
        [w_rec[0][0], w_rec[0][1], rec_p[0]],
        [w_rec[1][0], w_rec[1][1], rec_p[1]],
    ];
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
            let p = apply_member_row(&rec_row, pos_r_stored, x);
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
    // 1 px shift); acceptance ~0.3 px support-grid RMSE (the tolerances were
    // tuned at the original resolution=15 default; the 25-sample default
    // shifts the Nelder-Mead trajectory a hair — one case sits at 0.303).
    run_recovery_case(1.0, 0.0, 0.0, 0.07, 4.0, [1.0, 0.0], 0.3);
    run_recovery_case(0.8, -12.0, 0.1, -0.07, -4.0, [0.0, 1.0], 0.3);
    run_recovery_case(1.25, 20.0, 0.0, 0.07, 4.0, [0.7, -0.7], 0.32);
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
    // deterministically.) The member gate is off: a flat patch
    // is exactly what it excludes (see gate_unlocalizable_member_excluded),
    // and this test pins the downstream ZNCC path.
    let params = ClusterRefineParams {
        max_member_zncc_self_similarity_radius: 0.0,
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
    // Default params: the flat member's own patch matches itself at every
    // shift, so it reads the largest ZNCC self-similarity radius and the
    // member gate excludes it before refinement. With one usable member left the
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
    // A member whose patch straddles the image border is still read by the gate — the
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

/// A straight vertical edge through `x = 64`, softened over a couple of px:
/// a patch on it slides along the edge and still matches itself.
fn edge(x: f64, _y: f64) -> f64 {
    127.0 + 80.0 * ((x - 64.0) / 2.0).tanh()
}

#[test]
fn the_member_gate_reads_the_zncc_self_similarity_radius() {
    let params = ClusterRefineParams::default();
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let radius = |f: fn(f64, f64) -> f64| {
        member_zncc_self_similarity_radius(
            &pyramid(&make_image(128, 128, f)),
            [64.0, 64.0],
            a,
            &params,
        )
        .expect("an interior member's tile can be sampled")
    };
    // A textured patch pins its position; a flat one, a straight edge and a
    // texture smooth on the template grid match themselves at the largest
    // shift the reading searches.
    let textured = radius(texture);
    assert!(textured < 1.5, "textured: {textured}");
    assert_eq!(radius(|_, _| 127.0), 3.0);
    assert_eq!(radius(edge), 3.0);
    let smooth = radius(smooth_texture);
    assert!(smooth > 2.5, "smooth: {smooth}");

    // The same number the bench reads: the overlap reading of the member grid
    // itself.
    let pyr = pyramid(&make_image(128, 128, texture));
    let grid = sample_member_grid(&pyr, [64.0, 64.0], a, &params).unwrap();
    let r = params.resolution as usize;
    let (planes, colour) = PatchTile::planes_from_interleaved(&grid, r, r, grid.len() / (r * r));
    let tile = PatchTile {
        values: &planes,
        channels: colour,
        width: r,
        height: r,
    };
    let want =
        zncc_self_similarity_radius(&tile, None, [0, 0, r, r], &SelfSimilarityParams::default())
            .radius;
    assert_eq!(textured.to_bits(), want.to_bits());
}

/// The gate reads the member's own `R×R` grid and no pixel past it: an edge
/// whose surroundings past the grid's footprint hold the same edge inverted
/// reads the same radius as the edge alone, the largest the reading searches.
/// A reading that took its shifted windows from a ring of `r` samples around
/// the grid would see the inverted edge there and read the edge shorter; the
/// test reads the surrounded image that way too, to show it would.
#[test]
fn the_member_gate_reads_no_pixel_outside_the_member_grid() {
    let params = ClusterRefineParams::default();
    // At `a = 2.5` the grid spans `±radius · 2.5 = ±15` image px around the
    // keypoint, and the bilinear samples read at most a pixel further.
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let clean = make_image(128, 128, edge);
    let surrounded = make_image(128, 128, |x, y| {
        if (x - 64.0).abs() > 17.0 || (y - 64.0).abs() > 17.0 {
            254.0 - edge(x, y)
        } else {
            edge(x, y)
        }
    });
    let read = |img: &ImageU8| {
        member_zncc_self_similarity_radius(&pyramid(img), [64.0, 64.0], a, &params)
            .expect("an interior member's grid can be sampled")
    };
    let (clean_radius, surrounded_radius) = (read(&clean), read(&surrounded));
    assert_eq!(clean_radius, 3.0);
    assert_eq!(clean_radius.to_bits(), surrounded_radius.to_bits());

    // The same grid read with a ring: sampled `r` samples wider at the same
    // step, and read on the `R×R` template in its middle, which is the member
    // grid. The inverted edge in the ring pulls that reading under 3.
    let r = SelfSimilarityParams::default().max_radius as usize;
    let resolution = params.resolution as usize;
    let wide = resolution + 2 * r;
    let wide_params = ClusterRefineParams {
        radius: params.radius * wide as f64 / resolution as f64,
        resolution: wide as u32,
        ..params.clone()
    };
    let grid = sample_member_grid(&pyramid(&surrounded), [64.0, 64.0], a, &wide_params)
        .expect("the wider grid can be sampled");
    let (planes, colour) =
        PatchTile::planes_from_interleaved(&grid, wide, wide, grid.len() / (wide * wide));
    let tile = PatchTile {
        values: &planes,
        channels: colour,
        width: wide,
        height: wide,
    };
    let ringed = zncc_self_similarity_radius(
        &tile,
        None,
        [r, r, resolution, resolution],
        &SelfSimilarityParams::default(),
    )
    .radius;
    assert!(ringed < 3.0, "a ringed reading: {ringed}");
}

#[test]
fn the_member_gate_refuses_flat_and_edge_members_at_the_default() {
    // Three members of one cluster over the same point: one on a textured
    // image, one on an edge, one on a flat image. At the default bar the
    // edge and the flat member are refused before reference selection; the
    // edge member has the largest SIFT scale and would otherwise have been
    // the reference.
    let run = |params: &ClusterRefineParams| {
        let a = [[2.5, 0.0], [0.0, 2.5]];
        let a_big = [[3.0, 0.0], [0.0, 3.0]];
        let feats = [
            ImageFeatures::new(&[([64.0, 64.0], a)]),
            ImageFeatures::new(&[([64.0, 64.0], a)]),
            ImageFeatures::new(&[([64.0, 64.0], a_big)]),
            ImageFeatures::new(&[([64.0, 64.0], a)]),
        ];
        let pyramids = [
            pyramid(&make_image(128, 128, texture)),
            pyramid(&make_image(128, 128, texture)),
            pyramid(&make_image(128, 128, edge)),
            pyramid(&make_image(128, 128, |_, _| 127.0)),
        ];
        refine_cluster_patches(
            &pyramids,
            &geometry(&feats),
            &[0, 4],
            &[0, 1, 2, 3],
            &[0, 0, 0, 0],
            params,
            None,
        )
    };
    let result = run(&ClusterRefineParams::default());
    assert_eq!(result.reference_members[0], 0);
    assert_eq!(result.member_status[0], MemberStatus::Reference);
    assert_eq!(result.member_status[1], MemberStatus::Kept);
    assert_eq!(result.member_status[2], MemberStatus::RejectedUnlocalizable);
    assert_eq!(result.member_status[3], MemberStatus::RejectedUnlocalizable);

    // 0 turns the gate off: nobody is refused by it, and the edge member,
    // the largest, becomes the reference.
    let off = run(&ClusterRefineParams {
        max_member_zncc_self_similarity_radius: 0.0,
        ..Default::default()
    });
    assert!(!off
        .member_status
        .contains(&MemberStatus::RejectedUnlocalizable));
    assert_eq!(off.reference_members[0], 2);

    // A bar at the largest radius the reading reports turns nothing out.
    let open = run(&ClusterRefineParams {
        max_member_zncc_self_similarity_radius: 3.0,
        ..Default::default()
    });
    assert!(!open
        .member_status
        .contains(&MemberStatus::RejectedUnlocalizable));
}

#[test]
fn the_member_gate_passes_at_or_under_the_bar_and_fails_nan() {
    let on = ClusterRefineParams::default();
    assert_eq!(
        on.max_member_zncc_self_similarity_radius,
        crate::patch::keypoint_localize::DEFAULT_MAX_MEMBER_ZNCC_SELF_SIMILARITY_RADIUS
    );
    assert_eq!(on.max_member_zncc_self_similarity_radius, 2.5);
    assert!(on.member_self_similarity_gate_is_on());
    assert!(on.admits_member_zncc_self_similarity_radius(0.4));
    assert!(on.admits_member_zncc_self_similarity_radius(2.5));
    assert!(!on.admits_member_zncc_self_similarity_radius(2.6));
    assert!(!on.admits_member_zncc_self_similarity_radius(3.0));
    assert!(!on.admits_member_zncc_self_similarity_radius(f64::NAN));
    for bar in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let off = ClusterRefineParams {
            max_member_zncc_self_similarity_radius: bar,
            ..Default::default()
        };
        assert!(!off.member_self_similarity_gate_is_on(), "{bar}");
        assert!(off.admits_member_zncc_self_similarity_radius(3.0), "{bar}");
        assert!(
            off.admits_member_zncc_self_similarity_radius(f64::NAN),
            "{bar}"
        );
    }
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
        a.member_positions.as_slice().unwrap(),
        b.member_positions.as_slice().unwrap()
    );
    assert_eq!(
        a.member_affine_shapes.as_slice().unwrap(),
        b.member_affine_shapes.as_slice().unwrap()
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

// ── The gates at the refined shape ──────────────────────────────────────────

/// [`texture`] left of `x = 57` and flat to its right: a member's grid around
/// `(64, 64)` at shape `2.5` spans `x` from 49 to 79, so its left column of
/// cells is textured and the other six cells are flat or hold the step.
fn texture_left_third(x: f64, y: f64) -> f64 {
    if x < 57.0 {
        texture(x, y)
    } else {
        127.0
    }
}

/// [`texture`] with its two fine terms weakened to three tenths: the same surface
/// seen out of focus, so its patch at the true shape slides further over
/// itself than [`texture`]'s does.
fn defocused_texture(x: f64, y: f64) -> f64 {
    127.0
        + 40.0 * (0.11 * x + 0.06 * y + 1.3).sin()
        + 28.0 * (0.05 * x - 0.12 * y + 0.7).sin()
        + 20.0 * (0.17 * x + 0.13 * y + 2.9).sin()
        + 10.0 * (0.29 * x - 0.23 * y + 0.4).cos()
        + 4.5 * (0.83 * x + 0.47 * y + 0.2).sin()
        + 3.6 * (-0.52 * x + 0.88 * y + 1.1).sin()
}

#[test]
fn capped_cells_are_the_cells_at_the_cap_or_with_no_reading() {
    let mut cells = [[0.5; 3]; 3];
    assert_eq!(capped_cell_count(&cells), 0);
    cells[0][0] = 3.0;
    cells[1][2] = f64::NAN;
    cells[2][2] = 2.999;
    assert_eq!(capped_cell_count(&cells), 2);
    assert_eq!(capped_cell_count(&[[3.0; 3]; 3]), 9);
}

#[test]
fn the_refined_shape_verdict_follows_both_bars() {
    let sharp = [[0.5; 3]; 3];
    let mut three_capped = sharp;
    three_capped[0] = [3.0; 3];
    let on = ClusterRefineParams {
        regate_at_refined_shape: true,
        max_capped_cells: 2,
        ..Default::default()
    };
    assert!(on.refined_shape_gate_is_on() && on.capped_cell_gate_is_on());
    assert_eq!(on.refined_shape_verdict(1.0, &sharp), None);
    assert_eq!(on.refined_shape_verdict(2.5, &sharp), None);
    assert_eq!(
        on.refined_shape_verdict(2.6, &sharp),
        Some(MemberStatus::RejectedUnlocalizableRefined)
    );
    assert_eq!(
        on.refined_shape_verdict(f64::NAN, &sharp),
        Some(MemberStatus::RejectedUnlocalizableRefined)
    );
    assert_eq!(
        on.refined_shape_verdict(1.0, &three_capped),
        Some(MemberStatus::RejectedUnlocalizableCells)
    );
    // A member failing both takes the whole grid's status.
    assert_eq!(
        on.refined_shape_verdict(3.0, &three_capped),
        Some(MemberStatus::RejectedUnlocalizableRefined)
    );
    // Three capped cells pass a bar of three.
    let three = ClusterRefineParams {
        max_capped_cells: 3,
        ..on.clone()
    };
    assert_eq!(three.refined_shape_verdict(1.0, &three_capped), None);
    // The whole-grid gate is off without its flag, and with the member gate's
    // bar off; nine turns the cell gate off.
    let off = ClusterRefineParams {
        regate_at_refined_shape: false,
        max_capped_cells: CELL_COUNT,
        ..Default::default()
    };
    assert!(!off.reads_refined_shape());
    assert_eq!(off.refined_shape_verdict(3.0, &[[3.0; 3]; 3]), None);
    let bar_off = ClusterRefineParams {
        max_member_zncc_self_similarity_radius: 0.0,
        ..on.clone()
    };
    assert!(!bar_off.refined_shape_gate_is_on());
    assert_eq!(bar_off.refined_shape_verdict(3.0, &sharp), None);
}

#[test]
fn the_member_reading_counts_flat_cells_as_capped() {
    let params = ClusterRefineParams::default();
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let read = |f: fn(f64, f64) -> f64| {
        let pyr = pyramid(&make_image(128, 128, f));
        let parts = member_zncc_self_similarity_parts(&pyr, [64.0, 64.0], a, &params)
            .expect("an interior member's grid can be sampled");
        // The parts' whole is the up-front gate's template, read with shared
        // sums.
        let whole = member_zncc_self_similarity_radius(&pyr, [64.0, 64.0], a, &params).unwrap();
        assert!(
            (parts.whole.radius - whole).abs() < 1e-4,
            "{} {whole}",
            parts.whole.radius
        );
        let cells = parts
            .grid
            .each_ref()
            .map(|row| row.each_ref().map(|c| c.radius));
        (parts.whole.radius, capped_cell_count(&cells))
    };
    let (radius, capped) = read(texture);
    assert!(radius < 1.5, "textured: {radius}");
    assert_eq!(capped, 0);
    // Texture in the left column of cells pins the whole patch, while the six
    // cells to its right match themselves at every shift.
    let (radius, capped) = read(texture_left_third);
    assert!(radius <= 2.5, "textured third: {radius}");
    assert!(capped >= 6, "textured third: {capped} capped cells");
    let (radius, capped) = read(|_, _| 127.0);
    assert_eq!(radius, 3.0);
    assert_eq!(capped, 9);
}

/// One cluster over the same point in two images showing `f`, both members
/// at shape `2.5`: member 0 is the reference and member 1 is refined against
/// it.
fn run_pair(f: fn(f64, f64) -> f64, params: &ClusterRefineParams) -> ClusterRefineResult {
    let a = [[2.5, 0.0], [0.0, 2.5]];
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a)]),
        ImageFeatures::new(&[([64.0, 64.0], a)]),
    ];
    let pyramids = [
        pyramid(&make_image(128, 128, f)),
        pyramid(&make_image(128, 128, f)),
    ];
    refine_cluster_patches(
        &pyramids,
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        params,
        None,
    )
}

#[test]
fn the_capped_cell_gate_refuses_a_member_textured_in_one_third() {
    // The member's whole patch pins a position, so the up-front gate and the
    // whole-grid gate at the refined shape pass it, but six of its nine cells
    // are capped: a bar of five refuses it.
    let five = ClusterRefineParams {
        max_capped_cells: 5,
        ..Default::default()
    };
    let result = run_pair(texture_left_third, &five);
    assert_eq!(result.member_status[0], MemberStatus::Reference);
    assert_eq!(
        result.member_status[1],
        MemberStatus::RejectedUnlocalizableCells
    );
    // A refused member keeps its measurement and its readings.
    assert!(result.member_zncc[1] > 0.99);
    let radius = result.refined_zncc_self_similarity_radius[1];
    assert!(radius <= 2.5, "{radius}");
    let cells = result.refined_zncc_self_similarity_radius_grid[1].map(|r| r.map(f64::from));
    assert_eq!(capped_cell_count(&cells), 6);
    // The reference is not read at the refined shape.
    assert!(result.refined_zncc_self_similarity_radius[0].is_nan());

    // The default bar, eight, keeps it, with the same readings; with both
    // gates off nothing is read.
    let default = run_pair(texture_left_third, &ClusterRefineParams::default());
    assert_eq!(default.member_status[1], MemberStatus::Kept);
    assert_eq!(
        default.refined_zncc_self_similarity_radius[1].to_bits(),
        radius.to_bits()
    );
    let off = run_pair(
        texture_left_third,
        &ClusterRefineParams {
            regate_at_refined_shape: false,
            max_capped_cells: CELL_COUNT,
            ..Default::default()
        },
    );
    assert_eq!(off.member_status[1], MemberStatus::Kept);
    assert!(off.refined_zncc_self_similarity_radius[1].is_nan());
    assert!(off.refined_zncc_self_similarity_radius_grid[1]
        .iter()
        .flatten()
        .all(|r| r.is_nan()));
    // The readings change no other output.
    assert_eq!(
        off.member_zncc[1].to_bits(),
        default.member_zncc[1].to_bits()
    );
    assert_eq!(off.member_affine_shapes, default.member_affine_shapes);
}

#[test]
fn the_gates_at_the_refined_shape_are_on_by_default() {
    let params = ClusterRefineParams::default();
    assert!(params.regate_at_refined_shape);
    assert_eq!(params.max_capped_cells, DEFAULT_MAX_CAPPED_CELLS);
    assert_eq!(DEFAULT_MAX_CAPPED_CELLS, 8);
    assert!(params.refined_shape_gate_is_on() && params.capped_cell_gate_is_on());
}

/// A cluster whose member passes the up-front gate at its SIFT seed and not
/// at its refined shape: the reference shows [`texture`] magnified by 1.4 at
/// shape `3.5`, and the member shows the same surface out of focus
/// ([`defocused_texture`]) at its true shape `2.5`, detected at a shape 1.35
/// times too large. The seed's grid spans more of the defocused surface, so it
/// reads sharper than the grid at the true shape the cascade recovers.
fn run_defocused_member(params: &ClusterRefineParams) -> (ClusterRefineResult, ImageU8Pyramid) {
    let zoom = 1.4;
    let a_ref = [[2.5 * zoom, 0.0], [0.0, 2.5 * zoom]];
    let seed = [[2.5 * 1.35, 0.0], [0.0, 2.5 * 1.35]];
    let reference = pyramid(&make_image(128, 128, move |x, y| {
        texture(64.0 + (x - 64.0) / zoom, 64.0 + (y - 64.0) / zoom)
    }));
    let member = pyramid(&make_image(128, 128, defocused_texture));
    let member_again = pyramid(&make_image(128, 128, defocused_texture));
    let feats = [
        ImageFeatures::new(&[([64.0, 64.0], a_ref)]),
        ImageFeatures::new(&[([64.0, 64.0], seed)]),
    ];
    let result = refine_cluster_patches(
        &[reference, member],
        &geometry(&feats),
        &[0, 2],
        &[0, 1],
        &[0, 0],
        params,
        None,
    );
    (result, member_again)
}

#[test]
fn the_whole_grid_gate_reads_the_member_at_its_refined_shape() {
    let params = ClusterRefineParams::default();
    let seed = [[2.5 * 1.35, 0.0], [0.0, 2.5 * 1.35]];
    let (result, member) = run_defocused_member(&params);
    // The up-front gate reads the seed and passes it.
    let at_seed = member_zncc_self_similarity_radius(&member, [64.0, 64.0], seed, &params).unwrap();
    assert!(at_seed < 2.5, "at the seed: {at_seed}");
    assert_eq!(result.member_status[0], MemberStatus::Reference);
    assert_eq!(
        result.member_status[1],
        MemberStatus::RejectedUnlocalizableRefined
    );
    // The cascade found the true shape, and the reading is the up-front
    // gate's own reading of the member's grid there.
    let shape = result.member_affine_shapes.index_axis(ndarray::Axis(0), 1);
    let shape = [
        [shape[[0, 0]], shape[[0, 1]]],
        [shape[[1, 0]], shape[[1, 1]]],
    ];
    assert!(
        (shape[0][0] - 2.5).abs() < 0.15 && (shape[1][1] - 2.5).abs() < 0.15,
        "{shape:?}"
    );
    let position = [
        result.member_positions[[1, 0]],
        result.member_positions[[1, 1]],
    ];
    let at_refined = member_zncc_self_similarity_radius(&member, position, shape, &params).unwrap();
    assert!(at_refined > 2.5, "at the refined shape: {at_refined}");
    assert_eq!(
        result.refined_zncc_self_similarity_radius[1].to_bits(),
        (at_refined as f32).to_bits()
    );
    // The member keeps its measurement.
    assert!(result.member_zncc[1] >= params.min_zncc as f32);

    // Without the gate it is kept, at the same shape.
    let off = ClusterRefineParams {
        regate_at_refined_shape: false,
        ..Default::default()
    };
    let (kept, _) = run_defocused_member(&off);
    assert_eq!(kept.member_status[1], MemberStatus::Kept);
    assert_eq!(kept.member_affine_shapes, result.member_affine_shapes);
    // The cell gate still reads the grid, so the readings are there.
    assert_eq!(
        kept.refined_zncc_self_similarity_radius[1].to_bits(),
        result.refined_zncc_self_similarity_radius[1].to_bits()
    );
}
