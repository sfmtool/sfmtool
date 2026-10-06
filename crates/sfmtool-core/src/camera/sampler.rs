// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which sampler renders a view's tile, and the rule that picks it per view.
//!
//! A patch tile is a photograph resampled through a patch's frame
//! ([`WarpMap::from_patch`]). [`Sampler`] names the three ways to do the
//! resampling: plain bilinear, one bilinear sample from the mip level the
//! warp's compression picks ([`Sampler::BilinearMip`]), or the anisotropic
//! walk along the warp's major axis ([`Sampler::Anisotropic`]).
//!
//! A kernel that renders tiles takes a [`SamplerChoice`] rather than a
//! [`Sampler`]. The default, [`SamplerChoice::PerView`], applies the **sampler
//! rule** ([`rule_sampler`]) to each view: a view renders with `Anisotropic`
//! when `σ_major ≥ √2` and `L = 2^l / max(σ_minor, 1) ≥ a`, with
//! `l = round(log2 max(σ_major, 1))`, and with `BilinearMip` otherwise. `σ` are
//! the singular values of the Jacobian of the patch grid's map into the
//! photograph at the tile's centre ([`patch_grid_jacobian`]), in photograph px
//! per grid px, and `a` is [`SamplerChoice::PerView`]'s
//! `anisotropic_threshold`, [`DEFAULT_ANISOTROPIC_THRESHOLD`] unless the caller
//! sets another. [`SamplerChoice::Fixed`] renders every view with one sampler,
//! for comparison runs and for callers that need one.
//!
//! The rule reads only the Jacobian, which is computed before the render, so
//! every kernel that renders the same view at the same placement chooses the
//! same sampler, and a reader that knows a render's zoom and `a` can tell which
//! sampler it used. See `specs/core/camera/image-warping.md` § "Choosing the
//! sampler per view".
//!
//! ```
//! use sfmtool_core::camera::sampler::{rule_sampler, Sampler, SamplerChoice};
//!
//! // A view facing the patch at 3× shrink: both axes read level 2 already.
//! assert_eq!(rule_sampler([3.0, 3.0], 1.5), Sampler::BilinearMip);
//! // The same shrink across, but four times as compressed along: `BilinearMip`
//! // would read level 4 for both axes, 16 / 3 times too coarse across.
//! assert_eq!(rule_sampler([12.0, 3.0], 1.5), Sampler::Anisotropic);
//! assert_eq!(
//!     SamplerChoice::default().for_singular_values(Some([12.0, 3.0])),
//!     Sampler::Anisotropic,
//! );
//! assert_eq!(
//!     SamplerChoice::Fixed(Sampler::BilinearMip).for_singular_values(Some([12.0, 3.0])),
//!     Sampler::BilinearMip,
//! );
//! ```

use std::f64::consts::SQRT_2;

use crate::camera::image::{ImageF32WithGrad, ImageU8, ImageU8Pyramid};
use crate::camera::remap::{
    remap_aniso_with_grad_into, remap_aniso_with_pyramid, remap_bilinear, remap_bilinear_mip,
    remap_bilinear_mip_with_grad_into, remap_bilinear_with_grad_into,
};
use crate::camera::warp_map::{patch_grid_jacobian, singular_values_2x2};
use crate::camera::{CameraIntrinsics, WarpMap};
use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::progress::{Phase, Progress};
use crate::progress_note;

/// How to sample a photograph's pyramid when rendering a patch tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sampler {
    /// Plain bilinear from the full-resolution level: the cheapest tap, and
    /// within ~1° of anisotropic on fronto-parallel pinhole views — but it
    /// aliases on compressive warps (e.g. cross-scale views with one camera
    /// much closer), which corrupts the score surface the refiners descend.
    Bilinear,
    /// Single bilinear sample from the pyramid level nearest the warp's local
    /// compression (`round(log2(sigma_major))` per pixel, from the Jacobian
    /// SVD). The mip level bounds the aliasing `Bilinear` suffers on
    /// compressive warps at ≈ bilinear cost — at the price of blurring oblique
    /// views, whose anisotropic footprint only `Anisotropic`'s multi-tap walk
    /// resolves. What the sampler rule picks for a view it does not move.
    BilinearMip,
    /// Anisotropic sampling over the pyramid — the patch warp's Jacobian SVD picks
    /// the level from the minor axis and walks up to [`MAX_ANISOTROPY`] samples
    /// along the major axis, de-aliasing oblique / grazing views and keeping
    /// the detail along their less compressed axis. With the AVX2 kernel a
    /// patch tile costs 0.65 to 1.55 times what a `BilinearMip` one does, the
    /// most on views compressed 10 times or more along one axis; the scalar path,
    /// on a CPU without AVX2, 1.8 to 4 times as much, and the value+gradient
    /// render the sub-pixel refiner reads, which is scalar everywhere, 2.8 to
    /// 7 times. What the sampler rule picks for a view it moves.
    Anisotropic,
}

impl Sampler {
    /// The sampler's name as the bindings, the command line and the reports
    /// spell it: `bilinear`, `bilinear_mip` or `anisotropic`.
    pub fn name(self) -> &'static str {
        match self {
            Sampler::Bilinear => "bilinear",
            Sampler::BilinearMip => "bilinear_mip",
            Sampler::Anisotropic => "anisotropic",
        }
    }

    /// The [`Progress::detail_phase`] a render under this sampler is timed in.
    pub fn render_phase(self) -> &'static str {
        match self {
            Sampler::Bilinear => "render bilinear",
            Sampler::BilinearMip => "render bilinear_mip",
            Sampler::Anisotropic => "render anisotropic",
        }
    }

    /// Whether a render under this sampler needs the warp map's per-pixel SVD
    /// ([`WarpMap::compute_svd`]).
    pub fn needs_svd(self) -> bool {
        !matches!(self, Sampler::Bilinear)
    }
}

/// The cap on the anisotropic sampler's samples along the major axis, which
/// every patch kernel renders with.
pub const MAX_ANISOTROPY: u32 = 16;

/// The default threshold `a` of the sampler rule: a view moves to
/// [`Sampler::Anisotropic`] when its minor axis would be read at least this
/// many times coarser than its own compression needs.
///
/// Above the √2 that rounding to the nearest mip level alone produces on a
/// view facing the patch, so such a view stays on [`Sampler::BilinearMip`].
/// The value was set from the comparison runs in
/// `specs/core/camera/image-warping.md` § "Choosing the sampler per view".
pub const DEFAULT_ANISOTROPIC_THRESHOLD: f64 = 1.5;

/// The mip level `l = round(log2 max(σ_major, 1))` that
/// [`Sampler::BilinearMip`] reads for a pixel whose Jacobian's larger singular
/// value is `sigma_major`, before any clamp to the pyramid's depth. `0` for a
/// `sigma_major` that is not a number, and `u32::MAX` for an infinite one.
pub fn bilinear_mip_level(sigma_major: f64) -> u32 {
    if sigma_major.is_nan() {
        return 0;
    }
    sigma_major.max(1.0).log2().round() as u32
}

/// How many times coarser [`Sampler::BilinearMip`] reads a view's minor axis
/// than that axis's own compression needs: `L = 2^l / max(σ_minor, 1)`, with
/// `l` the level [`bilinear_mip_level`] picks from `σ_major`.
///
/// `singular_values` is `[σ_major, σ_minor]` of the patch grid's Jacobian, in
/// photograph px per grid px. A minor axis that magnifies the photograph
/// (`σ_minor < 1`) can only lose detail down to one photograph pixel, hence the
/// floor at 1. `NaN` where either value is not a number, and infinite where
/// `σ_major` is infinite and `σ_minor` is not.
///
/// `l` is not clamped to a pyramid's depth. Past `σ_major = 2^(K − 0.5)`, with
/// `K` the top level of the pyramid a kernel reads, `BilinearMip` reads level
/// `K`, so its loss along the minor axis is `2^K / max(σ_minor, 1)`, less than
/// `L`, and it aliases along the major axis instead, by `σ_major / 2^K`. The
/// rule reads `L` unclamped for two reasons. The anisotropic sampler is the
/// better of the two there as well, since its samples along the major axis
/// reduce that aliasing. And the depth differs between callers (the command
/// line builds every level, SfM Explorer six), while the rule must choose the
/// same sampler for the same observation in every kernel and caller.
pub fn minor_axis_loss(singular_values: [f64; 2]) -> f64 {
    let [major, minor] = singular_values;
    if major.is_nan() || minor.is_nan() {
        return f64::NAN;
    }
    // `exp2` of the level rather than `powi`: an infinite `σ_major` gives the
    // level `u32::MAX`, which `powi` would read as `-1` through its `i32`.
    let level = bilinear_mip_level(major);
    f64::from(level).exp2() / minor.max(1.0)
}

/// The sampler rule: [`Sampler::Anisotropic`] when `σ_major ≥ √2` and
/// [`minor_axis_loss`] is at least `threshold`, [`Sampler::BilinearMip`]
/// otherwise.
///
/// `singular_values` is `[σ_major, σ_minor]` of the patch grid's Jacobian at
/// the tile's centre ([`patch_grid_jacobian`], [`singular_values_2x2`]), in
/// photograph px per grid px.
///
/// - Below `σ_major = √2`, `BilinearMip` reads level 0 for both axes, so a
///   view whose lower zoom is above about 0.71× loses nothing to it.
/// - `L` is how much coarser the minor axis is read than it needs. It is above
///   1 on a view facing the patch whenever `σ` is just under a level boundary
///   (up to √2), so `threshold` should be above √2 to keep those views on
///   `BilinearMip`.
///
/// A value that is not a number (no Jacobian, a degenerate one) keeps
/// `BilinearMip`.
pub fn rule_sampler(singular_values: [f64; 2], threshold: f64) -> Sampler {
    let [major, _] = singular_values;
    let loss = minor_axis_loss(singular_values);
    if major >= SQRT_2 && loss >= threshold {
        Sampler::Anisotropic
    } else {
        Sampler::BilinearMip
    }
}

/// Which sampler a kernel renders each view's tile with: one for every view,
/// or the sampler rule applied per view.
///
/// What every patch kernel's parameters carry in place of a fixed
/// [`Sampler`]. `From<Sampler>` gives [`Self::Fixed`], so a caller that wants
/// one sampler for every view writes `Sampler::Bilinear.into()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplerChoice {
    /// Every view renders with this sampler, whatever its Jacobian.
    Fixed(Sampler),
    /// Each view renders with the sampler [`rule_sampler`] picks from its own
    /// Jacobian, with `anisotropic_threshold` as the rule's `a`.
    PerView {
        /// The rule's threshold `a` on [`minor_axis_loss`].
        anisotropic_threshold: f64,
    },
}

impl Default for SamplerChoice {
    /// The sampler rule at [`DEFAULT_ANISOTROPIC_THRESHOLD`].
    fn default() -> Self {
        Self::per_view()
    }
}

impl From<Sampler> for SamplerChoice {
    fn from(sampler: Sampler) -> Self {
        Self::Fixed(sampler)
    }
}

impl SamplerChoice {
    /// The sampler rule at [`DEFAULT_ANISOTROPIC_THRESHOLD`].
    pub const fn per_view() -> Self {
        Self::PerView {
            anisotropic_threshold: DEFAULT_ANISOTROPIC_THRESHOLD,
        }
    }

    /// The choice's name as the bindings and the command line spell it: the
    /// fixed sampler's [`Sampler::name`], or `per_view`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fixed(sampler) => sampler.name(),
            Self::PerView { .. } => "per_view",
        }
    }

    /// The rule's threshold `a`, or `None` for a fixed sampler.
    pub fn anisotropic_threshold(self) -> Option<f64> {
        match self {
            Self::Fixed(_) => None,
            Self::PerView {
                anisotropic_threshold,
            } => Some(anisotropic_threshold),
        }
    }

    /// The sampler for a view whose Jacobian has the singular values
    /// `singular_values` (`[σ_major, σ_minor]`). Pass `None` for a view with
    /// no Jacobian (its tile's centre does not project); the rule then keeps
    /// [`Sampler::BilinearMip`].
    pub fn for_singular_values(self, singular_values: Option<[f64; 2]>) -> Sampler {
        match self {
            Self::Fixed(sampler) => sampler,
            Self::PerView {
                anisotropic_threshold,
            } => match singular_values {
                Some(values) => rule_sampler(values, anisotropic_threshold),
                None => Sampler::BilinearMip,
            },
        }
    }

    /// The sampler for a view whose patch grid has `jacobian` at the tile's
    /// centre, as [`patch_grid_jacobian`] returns it.
    pub fn for_jacobian(self, jacobian: Option<[[f64; 2]; 2]>) -> Sampler {
        match self {
            Self::Fixed(sampler) => sampler,
            Self::PerView { .. } => self.for_singular_values(jacobian.map(singular_values_2x2)),
        }
    }

    /// The sampler for the tile rendered through `placement` at `resolution`
    /// texels a side ([`WarpMap::from_patch`] with the same arguments).
    ///
    /// The rule reads the Jacobian at the tile's centre in photograph px per
    /// tile texel, so a wider tile at the same texel size (a localizer's
    /// context tile) chooses as the core tile at its centre does. A fixed
    /// choice computes nothing.
    pub fn for_placement(
        self,
        placement: &OrientedPatch,
        camera: &CameraIntrinsics,
        cam_from_world: &RigidTransform,
        resolution: u32,
    ) -> Sampler {
        match self {
            Self::Fixed(sampler) => sampler,
            Self::PerView { .. } => self.for_jacobian(patch_grid_jacobian(
                placement,
                camera,
                cam_from_world,
                resolution as usize,
            )),
        }
    }

    /// The sampler for one observation of `patch`: [`Self::for_placement`]
    /// on `patch` re-anchored on the observation's `keypoint`
    /// ([`OrientedPatch::anchored_at_keypoint`]) at the patch resolution
    /// `resolution`, or on `patch` itself where there is no keypoint or its
    /// ray does not meet the patch.
    ///
    /// The one placement every kernel reads the rule from, whatever frame it
    /// then renders through: the bench's tiles, the fuse, the localizer's and
    /// the refiner's search tiles, the gates and Track View all choose a
    /// view's sampler here, once per view, so they choose the same one for
    /// the same observation at the same keypoint. A kernel that rebuilds the
    /// frame it renders (an orthonormal one, a wider one, one shifted by a
    /// search) does not change that choice.
    ///
    /// Each kernel passes the keypoint it starts from: the fuse, the gates and
    /// view selection's track views the stored keypoint, the bench and the
    /// localizer the seed of the round, the sub-pixel refiner the seed the
    /// localizer handed it, and view selection's candidates, which have no
    /// keypoint, `None`. Moving the keypoint by a few pixels changes the
    /// Jacobian by a small fraction, so two of these can choose differently
    /// only for a view whose `L` lies that close to `a`, or whose `σ_major`
    /// lies that close to a level boundary `2^(l + 0.5)`, where `L` doubles.
    pub fn for_observation(
        self,
        patch: &OrientedPatch,
        camera: &CameraIntrinsics,
        cam_from_world: &RigidTransform,
        keypoint: Option<[f64; 2]>,
        resolution: u32,
    ) -> Sampler {
        match self {
            Self::Fixed(sampler) => sampler,
            Self::PerView { .. } => {
                let anchored =
                    keypoint.and_then(|kp| patch.anchored_at_keypoint(camera, cam_from_world, kp));
                let placement = anchored.as_ref().unwrap_or(patch);
                self.for_placement(placement, camera, cam_from_world, resolution)
            }
        }
    }
}

/// Render `map` out of `pyramid` with `sampler`, computing the map's per-pixel
/// SVD first where the sampler reads it.
///
/// The one dispatch every patch kernel renders a tile through, so a sampler
/// added or changed here reaches all of them.
pub fn render_tile(pyramid: &ImageU8Pyramid, map: &mut WarpMap, sampler: Sampler) -> ImageU8 {
    if sampler.needs_svd() && !map.has_svd() {
        map.compute_svd();
    }
    match sampler {
        Sampler::Anisotropic => remap_aniso_with_pyramid(pyramid, map, MAX_ANISOTROPY),
        Sampler::BilinearMip => remap_bilinear_mip(pyramid, map),
        Sampler::Bilinear => remap_bilinear(pyramid.level(0), map),
    }
}

/// [`render_tile`] with the analytic image gradient, into `out`. The map's
/// per-pixel Jacobians are computed either way (by the SVD, or alone for
/// [`Sampler::Bilinear`]), since a caller composing the gradient with the warp
/// reads them.
pub fn render_tile_with_grad_into(
    pyramid: &ImageU8Pyramid,
    map: &mut WarpMap,
    sampler: Sampler,
    out: &mut ImageF32WithGrad,
) {
    match sampler {
        Sampler::Anisotropic => {
            if !map.has_svd() {
                map.compute_svd();
            }
            remap_aniso_with_grad_into(pyramid, map, MAX_ANISOTROPY, out);
        }
        Sampler::BilinearMip => {
            if !map.has_svd() {
                map.compute_svd();
            }
            remap_bilinear_mip_with_grad_into(pyramid, map, out);
        }
        Sampler::Bilinear => {
            if !map.has_jacobians() {
                map.compute_jacobians();
            }
            remap_bilinear_with_grad_into(pyramid.level(0), map, out);
        }
    }
}

/// Render each of `maps` out of the matching pyramid with the matching
/// sampler, and return the tiles in the order of `maps`.
///
/// The renders are grouped by sampler, and each group is timed in its own
/// [`render_phase`], so a detailed run reads one phase per sampler with the
/// number of views it rendered. The order of the renders does not change any
/// tile.
///
/// # Panics
///
/// Panics if `pyramids`, `maps` and `samplers` differ in length.
pub fn render_tiles(
    pyramids: &[&ImageU8Pyramid],
    maps: &mut [WarpMap],
    samplers: &[Sampler],
    progress: &Progress<'_>,
) -> Vec<ImageU8> {
    assert_eq!(pyramids.len(), maps.len(), "one pyramid per map");
    assert_eq!(samplers.len(), maps.len(), "one sampler per map");
    let mut tiles: Vec<Option<ImageU8>> = (0..maps.len()).map(|_| None).collect();
    for sampler in [
        Sampler::Bilinear,
        Sampler::BilinearMip,
        Sampler::Anisotropic,
    ] {
        let count = samplers.iter().filter(|&&s| s == sampler).count();
        if count == 0 {
            continue;
        }
        let _phase = render_phase(progress, sampler, count);
        for (i, map) in maps.iter_mut().enumerate() {
            if samplers[i] == sampler {
                tiles[i] = Some(render_tile(pyramids[i], map, sampler));
            }
        }
    }
    tiles
        .into_iter()
        .map(|tile| tile.expect("every map has a sampler"))
        .collect()
}

/// Open the [`Progress::detail_phase`] that times `views` renders under
/// `sampler`, with a note giving the count. Inert unless `progress` is
/// detailed.
pub fn render_phase<'a>(progress: &Progress<'a>, sampler: Sampler, views: usize) -> Phase<'a> {
    let mut phase = progress.detail_phase(sampler.render_phase());
    if views == 1 {
        progress_note!(phase, "1 view");
    } else {
        progress_note!(phase, "{views} views");
    }
    phase
}

#[cfg(test)]
mod tests;
