// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Fitting one loaded node's similarity transform onto another's frame.
//!
//! The Scene Graph panel's `Align to ▸ <node>` action (see
//! `specs/gui/gui-scene-graph.md`, "Node Transforms and Alignment"). The
//! estimation itself already lives in
//! [`sfmtool_core::analysis::alignment`] and
//! [`sfmtool_core::reconstruction::point_correspondence`]; what the viewer adds
//! is **correspondence gathering** — the two ways of deciding which of the
//! source's points map onto which of the target's — and the outcome numbers the
//! status line reports.
//!
//! ## The two modes
//!
//! - **Cameras** (default): images matched by `name` across the two
//!   reconstructions; the corresponded camera *centres* feed the fit. Works
//!   whenever the two were solved from overlapping image sets, which is the
//!   typical comparison case. Correspondence counts are in the tens or
//!   hundreds, far too few for RANSAC to have anything to consense over, so
//!   this mode relies on the trimmed refit alone.
//! - **Points**: [`find_point_correspondences`] matches 3D points through
//!   shared feature observations in shared images, yielding orders of magnitude
//!   more correspondences — enough for RANSAC. Needs feature-indexed
//!   observations ([`sfmtool_core::ObservationSource::SiftFiles`]) in *both*
//!   reconstructions, because the match identity is `(image, feature index)`.
//!
//! Points at infinity are excluded from both: a `w = 0` point stores a unit
//! bearing, not a location, and would drag a positional fit toward the origin.
//! This mirrors what `sfm align` does (`_finite_pair_mask` in
//! `src/sfmtool/_point_correspondence.py`).
//!
//! The fitted transform maps the **source's native coordinates into the
//! target's native coordinates**; composing it into the target's own displayed
//! frame is the caller's job (see [`crate::state::AppState::align_node`]).

use std::collections::HashMap;

use sfmtool_core::analysis::alignment::{estimate_alignment, ransac_alignment, AlignmentParams};
use sfmtool_core::reconstruction::point_correspondence::find_point_correspondences;
use sfmtool_core::{Se3Transform, SfmrReconstruction};

/// Refit iterations for the trimmed least-squares fit. One initial fit over
/// everything, then two refits on the best-fitting fraction.
const TRIM_ROUNDS: usize = 3;

/// Fraction of correspondences kept by each refit round.
const KEEP_FRACTION: f64 = 0.8;

/// Fewest camera correspondences a similarity fit is allowed to run on. Two
/// points leave the rotation about the line joining them unconstrained, so three
/// is the first count that pins an orientation down at all.
const MIN_CAMERA_CORRESPONDENCES: usize = 3;

/// Fewest point correspondences before and after RANSAC. Mirrors `sfm align`'s
/// `min_points` default.
const MIN_POINT_CORRESPONDENCES: usize = 10;

/// RANSAC rounds for the point mode.
///
/// `sfm align` uses 1000, but it hands the per-round inlier count to numpy;
/// here every round walks the correspondences in scalar Rust, and this fit runs
/// **synchronously on the UI thread**. 200 rounds keeps a 10⁵-correspondence
/// pair inside a frame or two while still finding the dominant transform — the
/// preliminary all-correspondence fit that sets the threshold has already done
/// most of the work.
const RANSAC_ITERATIONS: usize = 200;

/// Sample size per RANSAC round: the minimum that determines a similarity.
const RANSAC_SAMPLE: usize = 3;

/// Percentile of the preliminary-fit residuals used as the RANSAC inlier
/// threshold, as in `sfm align`.
const RANSAC_PERCENTILE: f64 = 0.95;

/// Fixed RANSAC seed, so the same pair of reconstructions always aligns the
/// same way.
const RANSAC_SEED: u64 = 42;

/// Lower bound on the RANSAC threshold, as a fraction of the target cloud's
/// extent. Comfortably above `f64` rounding noise (~1e-14 relative) and
/// comfortably below any residual a real fit leaves behind.
const ROUNDING_FLOOR: f64 = 1e-9;

/// Where the correspondences behind a fit come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignSource {
    /// Camera centres of same-named images.
    #[default]
    Cameras,
    /// 3D points matched through shared feature observations.
    Points,
}

impl AlignSource {
    /// What one correspondence is called in the status line.
    fn noun(self) -> &'static str {
        match self {
            AlignSource::Cameras => "cameras",
            AlignSource::Points => "points",
        }
    }
}

/// The two choices the `Align to ▸` popup offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignOptions {
    pub source: AlignSource,
    /// Fit a similarity (scale + rotation + translation) when `true`, a rigid
    /// transform when `false`. Similarity by default: two solves of the same
    /// shoot agree only up to scale unless something fixed it.
    pub estimate_scale: bool,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            source: AlignSource::Cameras,
            estimate_scale: true,
        }
    }
}

/// A successful fit and the numbers describing how well it went.
#[derive(Debug, Clone)]
pub struct AlignFit {
    /// Maps the source's native coordinates onto the target's native
    /// coordinates.
    pub transform: Se3Transform,
    /// How many correspondences were gathered.
    pub correspondences: usize,
    /// How many of them the final fit actually used.
    pub inliers: usize,
    /// RMS residual over the inliers, in the target's units.
    pub rms: f64,
    /// Which mode produced them.
    pub source: AlignSource,
}

/// Fit the similarity taking `source` onto `target`.
///
/// Returns the reason as an `Err` string on every failure the spec names — no
/// shared images, too few correspondences, a degenerate configuration, an SVD
/// that would not converge — so the caller can leave the transform untouched and
/// report it.
pub fn fit_alignment(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    options: AlignOptions,
) -> Result<AlignFit, String> {
    match options.source {
        AlignSource::Cameras => fit_by_cameras(source, target, options.estimate_scale),
        AlignSource::Points => fit_by_points(source, target, options.estimate_scale),
    }
}

// ── Correspondence gathering ────────────────────────────────────────────

/// A flat `[x0, y0, z0, x1, …]` source/target pair, the layout the core
/// estimators take.
struct Correspondences {
    source: Vec<f64>,
    target: Vec<f64>,
}

impl Correspondences {
    fn len(&self) -> usize {
        self.source.len() / 3
    }

    fn push(&mut self, source: &nalgebra::Point3<f64>, target: &nalgebra::Point3<f64>) {
        self.source
            .extend_from_slice(&[source.x, source.y, source.z]);
        self.target
            .extend_from_slice(&[target.x, target.y, target.z]);
    }

    /// The subset named by `keep`, in the same flat layout.
    fn select(&self, keep: &[usize]) -> Correspondences {
        let mut out = Correspondences {
            source: Vec::with_capacity(keep.len() * 3),
            target: Vec::with_capacity(keep.len() * 3),
        };
        for &i in keep {
            out.source.extend_from_slice(&self.source[i * 3..i * 3 + 3]);
            out.target.extend_from_slice(&self.target[i * 3..i * 3 + 3]);
        }
        out
    }
}

/// `image name → index`, first occurrence winning — the same rule
/// `sfm align`'s by-cameras mode uses when a name repeats.
fn images_by_name(recon: &SfmrReconstruction) -> HashMap<&str, usize> {
    let mut map = HashMap::with_capacity(recon.images.len());
    for (i, image) in recon.images.iter().enumerate() {
        map.entry(image.name.as_str()).or_insert(i);
    }
    map
}

/// Image index pairs for the images the two reconstructions share by name.
fn shared_images(source: &SfmrReconstruction, target: &SfmrReconstruction) -> Vec<(u32, u32)> {
    let target_by_name = images_by_name(target);
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut pairs = Vec::new();
    for (i, image) in source.images.iter().enumerate() {
        let name = image.name.as_str();
        if seen.insert(name, ()).is_some() {
            continue; // a repeated source name pairs once, like the target side
        }
        if let Some(&j) = target_by_name.get(name) {
            pairs.push((i as u32, j as u32));
        }
    }
    pairs
}

/// Camera centres of the same-named images, source against target.
fn camera_correspondences(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
) -> Correspondences {
    let mut out = Correspondences {
        source: Vec::new(),
        target: Vec::new(),
    };
    for (i, j) in shared_images(source, target) {
        out.push(
            &source.images[i as usize].camera_center(),
            &target.images[j as usize].camera_center(),
        );
    }
    out
}

/// The three parallel track columns [`find_point_correspondences`] joins on.
fn track_columns(recon: &SfmrReconstruction) -> Option<(Vec<u32>, &[u32], Vec<u32>)> {
    let feature_indexes = recon.feature_indexes()?;
    let images = recon.tracks.iter().map(|t| t.image_index).collect();
    let points = recon.tracks.iter().map(|t| t.point_index).collect();
    Some((images, feature_indexes, points))
}

/// 3D point pairs matched through shared feature observations in shared images.
///
/// Pairs where either side is a point at infinity are dropped: a `w = 0` point's
/// stored position is a unit bearing, and feeding it to a positional fit would
/// corrupt the solve.
fn point_correspondences(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
) -> Result<Correspondences, String> {
    let shared = shared_images(source, target);
    if shared.is_empty() {
        return Err("the two reconstructions share no image names".to_string());
    }
    let (source_images, source_features, source_points) = track_columns(source)
        .ok_or("the source reconstruction has no feature indexes (embedded_patches)")?;
    let (target_images, target_features, target_points) = track_columns(target)
        .ok_or("the target reconstruction has no feature indexes (embedded_patches)")?;

    let shared_source: Vec<u32> = shared.iter().map(|&(s, _)| s).collect();
    let shared_target: Vec<u32> = shared.iter().map(|&(_, t)| t).collect();
    let matched = find_point_correspondences(
        &source_images,
        source_features,
        &source_points,
        &target_images,
        target_features,
        &target_points,
        &shared_source,
        &shared_target,
    );

    let mut out = Correspondences {
        source: Vec::new(),
        target: Vec::new(),
    };
    for (&s, &t) in matched.source_ids.iter().zip(matched.target_ids.iter()) {
        let (Some(sp), Some(tp)) = (source.points.get(s as usize), target.points.get(t as usize))
        else {
            continue;
        };
        if sp.is_at_infinity() || tp.is_at_infinity() {
            continue;
        }
        out.push(&sp.position, &tp.position);
    }
    if out.len() == 0 {
        return Err(format!(
            "no finite 3D point correspondences across the {} shared image(s)",
            shared.len()
        ));
    }
    Ok(out)
}

// ── Fits ────────────────────────────────────────────────────────────────

fn params(estimate_scale: bool) -> AlignmentParams {
    AlignmentParams {
        rounds: TRIM_ROUNDS,
        keep_fraction: KEEP_FRACTION,
        estimate_scale,
    }
}

/// Per-correspondence residual `‖s·R·src + t − tgt‖` under `transform`.
fn residuals(c: &Correspondences, transform: &Se3Transform) -> Vec<f64> {
    let rot = transform.rotation.to_rotation_matrix();
    (0..c.len())
        .map(|i| {
            let s =
                nalgebra::Vector3::new(c.source[i * 3], c.source[i * 3 + 1], c.source[i * 3 + 2]);
            let t =
                nalgebra::Vector3::new(c.target[i * 3], c.target[i * 3 + 1], c.target[i * 3 + 2]);
            (transform.scale * (rot * s) + transform.translation - t).norm()
        })
        .collect()
}

/// Indices of the `keep` smallest residuals — the correspondences the trimmed
/// refit converged on, recovered under the *final* transform so the reported
/// inlier count and RMS describe the fit that was actually kept.
fn trimmed_indices(residuals: &[f64], keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..residuals.len()).collect();
    order.sort_by(|&a, &b| residuals[a].total_cmp(&residuals[b]));
    order.truncate(keep.max(1).min(residuals.len()));
    order
}

/// Largest per-axis span of a flat coordinate list — a cheap stand-in for "how
/// big is this cloud", used only to scale the RANSAC threshold floor.
fn extent(flat: &[f64]) -> f64 {
    (0..3)
        .map(|axis| {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for v in flat.iter().skip(axis).step_by(3) {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
            if hi >= lo {
                hi - lo
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max)
}

fn rms(residuals: &[f64], keep: &[usize]) -> f64 {
    if keep.is_empty() {
        return 0.0;
    }
    let sum: f64 = keep.iter().map(|&i| residuals[i] * residuals[i]).sum();
    (sum / keep.len() as f64).sqrt()
}

/// How many correspondences the trimmed refit ends up fitting on.
fn kept_count(n: usize) -> usize {
    let keep = (KEEP_FRACTION * n as f64).round() as usize;
    keep.clamp(1, n)
}

fn fit_by_cameras(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    estimate_scale: bool,
) -> Result<AlignFit, String> {
    let c = camera_correspondences(source, target);
    let n = c.len();
    if n < MIN_CAMERA_CORRESPONDENCES {
        return Err(match n {
            0 => "the two reconstructions share no image names".to_string(),
            _ => format!(
                "only {n} shared image(s); a similarity needs at least \
                 {MIN_CAMERA_CORRESPONDENCES} camera correspondences"
            ),
        });
    }
    // No RANSAC: a few hundred camera centres are too few for a consensus
    // sample to beat fitting them all and trimming the worst away.
    let transform = estimate_alignment(&c.source, &c.target, n, params(estimate_scale))?;
    let resid = residuals(&c, &transform);
    let keep = trimmed_indices(&resid, kept_count(n));
    Ok(AlignFit {
        rms: rms(&resid, &keep),
        inliers: keep.len(),
        correspondences: n,
        transform,
        source: AlignSource::Cameras,
    })
}

fn fit_by_points(
    source: &SfmrReconstruction,
    target: &SfmrReconstruction,
    estimate_scale: bool,
) -> Result<AlignFit, String> {
    let c = point_correspondences(source, target)?;
    let n = c.len();
    if n < MIN_POINT_CORRESPONDENCES {
        return Err(format!(
            "only {n} shared 3D point(s); need at least {MIN_POINT_CORRESPONDENCES}"
        ));
    }

    // Threshold the way `sfm align` does: a preliminary all-correspondence fit,
    // then the 95th percentile of its residuals. A fixed absolute threshold
    // would be meaningless — reconstructions carry no metric scale.
    let preliminary = estimate_alignment(&c.source, &c.target, n, AlignmentParams::default())?;
    let mut prelim_resid = residuals(&c, &preliminary);
    prelim_resid.sort_by(f64::total_cmp);
    let threshold = prelim_resid[((n as f64 * RANSAC_PERCENTILE) as usize).min(n - 1)];
    // An exactly-corresponding pair — a synthetic fixture, or a reconstruction
    // aligned to a copy of itself — puts every residual at f64 rounding noise,
    // and `ransac_alignment` compares strictly less-than. Floor the threshold
    // well above that noise but far below anything a real fit produces, so such
    // a pair does not come back with zero inliers.
    let threshold = threshold.max(ROUNDING_FLOOR * extent(&c.target));

    let mask = ransac_alignment(
        &c.source,
        &c.target,
        n,
        RANSAC_ITERATIONS,
        threshold,
        RANSAC_SAMPLE,
        RANSAC_SEED,
    );
    let inlier_indices: Vec<usize> = (0..n).filter(|&i| mask[i]).collect();
    if inlier_indices.len() < MIN_POINT_CORRESPONDENCES {
        return Err(format!(
            "RANSAC kept only {} of {n} point correspondences; the two \
             reconstructions do not agree on a single transform",
            inlier_indices.len()
        ));
    }
    let inliers = c.select(&inlier_indices);

    let transform = estimate_alignment(
        &inliers.source,
        &inliers.target,
        inlier_indices.len(),
        params(estimate_scale),
    )?;
    let resid = residuals(&inliers, &transform);
    let keep = trimmed_indices(&resid, kept_count(inlier_indices.len()));
    Ok(AlignFit {
        rms: rms(&resid, &keep),
        inliers: inlier_indices.len(),
        correspondences: n,
        transform,
        source: AlignSource::Points,
    })
}

// ── Status text ─────────────────────────────────────────────────────────

/// `Aligned run_b → run_a: 214/243 cameras, RMS 0.031`.
pub fn success_message(source_label: &str, target_label: &str, fit: &AlignFit) -> String {
    format!(
        "Aligned {source_label} → {target_label}: {}/{} {}, RMS {:.3}",
        fit.inliers,
        fit.correspondences,
        fit.source.noun(),
        fit.rms,
    )
}

/// `Align run_b → run_a failed: <reason>`. The node's transform is left exactly
/// as it was.
pub fn failure_message(source_label: &str, target_label: &str, reason: &str) -> String {
    format!("Align {source_label} → {target_label} failed: {reason}")
}

#[cfg(test)]
mod tests;
