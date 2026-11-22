// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The 3D position of a measurement endpoint, a pixel in one image: the
//! observations in that image whose square holds the pixel, each mapping it to
//! an exact texel `(s, t)` of its point's patch, and the one picked to stand
//! for it.
//!
//! An observation's **square** is its point's patch slid onto the
//! observation's keypoint ([`OrientedPatch::seen_at_keypoint`]), the square
//! every render of the observation draws and its stored readings were taken
//! on. Where exactly one square holds the pixel, that observation is the
//! answer, with no ranking and no stored reading read. Where several hold it,
//! the front layer is ranked by a cost in scene units read from each
//! candidate's own stored readings, in the manner of the reference-view rule
//! ([`crate::patch::reference_view`]).
//!
//! [`endpoint_views`] runs the mapping backwards: for each candidate, the
//! pixels in the point's other observations that hold the same texel, where
//! the endpoint can move to without moving on the surface.
//!
//! `specs/core/reconstruction/measurement-endpoint.md` is the design.

use nalgebra::{Matrix3, Point3, Vector3};

use crate::geometry::RigidTransform;
use crate::patch::cloud::OrientedPatch;
use crate::patch::normal_refine::viewing_angle;
use crate::patch::observation_reading::stored_ellipse;
use crate::patch::reference_view::ReferenceViewOptions;
use crate::patch::self_similarity::SelfSimilarityEllipse;
use crate::reconstruction::{EditedReconstruction, PointView};
use crate::RotQuaternion;

#[cfg(test)]
mod tests;

/// Where one endpoint of a measurement lies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndpointPosition {
    /// A point in scene units.
    Finite {
        /// `X_p + s·u + t·v` on the point's own patch.
        position: [f64; 3],
    },
    /// The texel lies on the patch of a point at infinity: a unit direction.
    /// No distance can be measured to it.
    AtInfinity {
        /// The unit direction of `p + s·u + t·v`, `p` the point's direction.
        direction: [f64; 3],
    },
}

impl EndpointPosition {
    /// The position, `None` at infinity.
    pub fn finite(&self) -> Option<[f64; 3]> {
        match self {
            EndpointPosition::Finite { position } => Some(*position),
            EndpointPosition::AtInfinity { .. } => None,
        }
    }
}

/// One observation in the endpoint's image whose square holds the pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointCandidate {
    /// The point's live index in the edited reconstruction.
    pub point: u32,
    /// The observation's index within the point's own track.
    pub observation: usize,
    /// `(s, t)` on the point's patch, each in `[-1, 1]`: the texel of the
    /// observation's render that shows the pixel.
    pub patch_coords: [f64; 2],
    /// The 3D point that texel stands for.
    pub position: EndpointPosition,
    /// The distance along the pixel's unit ray from the camera centre to the
    /// square; `∞` for the square of a point at infinity.
    pub depth: f64,
    /// Whether it was ranked, and how.
    pub standing: CandidateStanding,
}

/// What the operation decided about one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CandidateStanding {
    /// The only square that holds the pixel: no ranking was made and no
    /// stored reading was read.
    Alone,
    /// One of several, in the front layer, with its cost.
    Ranked(EndpointCost),
    /// Not eligible.
    Excluded(EndpointExclusion),
}

/// One ranked candidate's cost and the factors it is the product of:
/// `cost = localization · capped_radius · offset · angle · agreement`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndpointCost {
    /// `ℓ`, the semi-major axis of the observation's self-similarity ellipse
    /// on the patch ([`SelfSimilarityEllipse::on_patch`]): in scene units for
    /// a finite point, in degrees for a point at infinity.
    pub localization: f64,
    /// Whether [`Self::localization`] is read from the observation's own
    /// stored reading, rather than the stand-in
    /// [`ReferenceViewOptions::unread_radius`].
    pub localization_read: bool,
    /// [`ReferenceViewOptions::capped_radius_factor`] where the reading's
    /// semi-major axis is at the cap of its reading, else `1`.
    pub capped_radius: f64,
    /// `O = 1 + w · ρ²`, `ρ = max(|s|, |t|)`.
    pub offset: f64,
    /// `A = 1 + k · tan θ`, the reference-view rule's angle factor.
    pub angle: f64,
    /// `Z`, the reference-view rule's agreement penalty on the shortfall of
    /// the observation's blur-matched ZNCC against the stored bitmap below
    /// `1`.
    pub agreement: f64,
    /// The product.
    pub cost: f64,
}

/// One of the factors an endpoint cost is the product of
/// ([`EndpointCost::ln_parts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EndpointTerm {
    /// The localization length `ℓ`.
    Localization,
    /// The step on a radius read at the cap.
    CappedRadius,
    /// The offset factor.
    Offset,
    /// The angle factor.
    Angle,
    /// The agreement penalty.
    Agreement,
}

impl EndpointTerm {
    /// The term's name as the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            EndpointTerm::Localization => "localization",
            EndpointTerm::CappedRadius => "capped_radius",
            EndpointTerm::Offset => "offset",
            EndpointTerm::Angle => "angle",
            EndpointTerm::Agreement => "agreement",
        }
    }
}

impl std::fmt::Display for EndpointTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl EndpointCost {
    /// `ln cost` split into one part per term, so that a caller can name the
    /// term that cost a candidate most against the pick, as
    /// [`ReferenceStanding::costliest_term`](crate::patch::reference_view::ReferenceStanding::costliest_term)
    /// does for a view.
    pub fn ln_parts(&self) -> [(EndpointTerm, f64); 5] {
        [
            (EndpointTerm::Localization, self.localization.ln()),
            (EndpointTerm::CappedRadius, self.capped_radius.ln()),
            (EndpointTerm::Offset, self.offset.ln()),
            (EndpointTerm::Angle, self.angle.ln()),
            (EndpointTerm::Agreement, self.agreement.ln()),
        ]
    }
}

/// Why a candidate was not eligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EndpointExclusion {
    /// The square's outward normal faces away from the camera along the
    /// pixel's ray (`n · d ≥ 0`): the square is seen from behind.
    BackFacing,
    /// Several squares hold the pixel and this one lies behind the front
    /// layer, the surface nearest the camera.
    BehindFrontLayer,
}

impl EndpointExclusion {
    /// The exclusion's name as the bindings spell it.
    pub fn name(self) -> &'static str {
        match self {
            EndpointExclusion::BackFacing => "back_facing",
            EndpointExclusion::BehindFrontLayer => "behind_front_layer",
        }
    }
}

impl std::fmt::Display for EndpointExclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Where an endpoint resolved to: every observation whose square holds the
/// pixel, and the one picked.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointResolution {
    /// The pick, an index into [`Self::candidates`].
    pub chosen: usize,
    /// Every observation in the image whose square holds the pixel, in
    /// ascending order of point index and then observation index.
    pub candidates: Vec<EndpointCandidate>,
}

impl EndpointResolution {
    /// The picked candidate.
    pub fn chosen(&self) -> &EndpointCandidate {
        &self.candidates[self.chosen]
    }

    /// The picked candidate's position.
    pub fn position(&self) -> EndpointPosition {
        self.chosen().position
    }
}

/// Why an endpoint has no position. Its `Display` is a sentence that names
/// the remedy.
#[derive(Debug, Clone, PartialEq)]
pub enum EndpointRefusal {
    /// The image index is past the reconstruction's images.
    NoSuchImage {
        /// The index asked for.
        image: u32,
        /// How many images the reconstruction has.
        image_count: usize,
    },
    /// The image carries no pose, so its pixels state no ray.
    NoPose {
        /// The image.
        image: u32,
    },
    /// The pixel is not finite, lies outside `[0, width) × [0, height)`, or
    /// lies outside the camera model's domain.
    PixelOutsideImage {
        /// The image.
        image: u32,
        /// The pixel.
        pixel: [f64; 2],
    },
    /// The reconstruction stores no patch frames, so no observation has a
    /// square.
    NoPatchFrames,
    /// No live point is observed in the image.
    NoObservations {
        /// The image.
        image: u32,
    },
    /// No observation's square in the image holds the pixel.
    NoCoveringSquare {
        /// The image.
        image: u32,
        /// The pixel.
        pixel: [f64; 2],
    },
    /// Squares hold the pixel, but none is eligible: each is seen from
    /// behind.
    AllExcluded {
        /// Every square that holds the pixel, each with its exclusion.
        candidates: Vec<EndpointCandidate>,
    },
}

impl std::fmt::Display for EndpointRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EndpointRefusal::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not in the reconstruction, which has {image_count} images; \
                 place the endpoint in one of them"
            ),
            EndpointRefusal::NoPose { image } => write!(
                f,
                "image {image} has no pose, so its pixels have no rays; resect the image to \
                 give the endpoint a position"
            ),
            EndpointRefusal::PixelOutsideImage { image, pixel } => write!(
                f,
                "pixel ({}, {}) is not inside image {image} or its camera model's domain; \
                 place the endpoint inside the image",
                pixel[0], pixel[1]
            ),
            EndpointRefusal::NoPatchFrames => write!(
                f,
                "the reconstruction stores no patch frames, so no observation has a square to \
                 hold an endpoint; convert it with sfm xform --to-embedded-patches"
            ),
            EndpointRefusal::NoObservations { image } => write!(
                f,
                "no point is observed in image {image}; add tracks to the image, for example \
                 with Create Track Here at the endpoint's pixel"
            ),
            EndpointRefusal::NoCoveringSquare { image, pixel } => write!(
                f,
                "no observation's square in image {image} holds pixel ({}, {}); use Create \
                 Track Here at the endpoint's pixel to build a track observed there",
                pixel[0], pixel[1]
            ),
            EndpointRefusal::AllExcluded { candidates } => write!(
                f,
                "the {} square(s) that hold the endpoint's pixel are all seen from behind; \
                 fix the normals of their points or use Create Track Here at the pixel",
                candidates.len()
            ),
        }
    }
}

impl std::error::Error for EndpointRefusal {}

/// The settings of [`resolve_measurement_endpoint`]: how deep the front layer
/// reaches, the weight of the offset factor, and the reference-view rule's
/// settings that the cost reads the same readings with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementEndpointOptions {
    /// The front layer holds the candidates whose depth is within this many
    /// of the nearest candidate's patch half-sizes (`max(|u|, |v|)`) of the
    /// nearest depth. Default `1`.
    pub layer_depth_factor: f64,
    /// `w` of the offset factor `1 + w · ρ²`. Default `1`.
    pub offset_weight: f64,
    /// The factor on [`Self::offset_weight`] for a point whose stored normal
    /// confidence is `0`, a placeholder normal. Default `4`.
    pub placeholder_normal_factor: f64,
    /// The reference-view rule's settings. The cost reads `angle_weight`,
    /// `capped_radius_factor`, `agreement_band`, `agreement_scale` and
    /// `unread_radius`; the rule's other fields do not apply to it. Default
    /// [`ReferenceViewOptions::default`].
    pub reference_view: ReferenceViewOptions,
}

impl Default for MeasurementEndpointOptions {
    fn default() -> Self {
        Self {
            layer_depth_factor: 1.0,
            offset_weight: 1.0,
            placeholder_normal_factor: 4.0,
            reference_view: ReferenceViewOptions::default(),
        }
    }
}

/// Both endpoints of one stored distance measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedDistanceMeasurement {
    /// Each endpoint's resolution, or why it has none.
    pub endpoints: [Result<EndpointResolution, EndpointRefusal>; 2],
    /// The distance between the two picked positions, in scene units, where
    /// both resolve to finite positions; `None` otherwise.
    pub scene_distance: Option<f64>,
}

/// A square that holds the pixel, before its standing is decided.
struct Hit {
    candidate: EndpointCandidate,
    /// The point's own patch.
    patch: OrientedPatch,
    /// The observation's square: the patch slid onto its keypoint.
    square: OrientedPatch,
    back_facing: bool,
}

/// Where the pixel's ray `origin + λ·dir` meets `square`, as `(λ, [s, t])`,
/// where it does so in front of the camera and inside the square.
///
/// A finite square is `c + s·u + t·v`; a square at infinity is the directions
/// `p + s·u + t·v`, met by a ray parallel to one of them. Both are the same
/// 3×3 system `λ·d − s·u − t·v = c − o`, with `o = 0` at infinity.
fn meet_square(
    square: &OrientedPatch,
    origin: &Point3<f64>,
    dir: &Vector3<f64>,
) -> Option<(f64, [f64; 2])> {
    let u = square.u_axis * square.half_extent[0];
    let v = square.v_axis * square.half_extent[1];
    let rhs = if square.w == 0.0 {
        square.center.coords
    } else {
        square.center - origin
    };
    let system = Matrix3::from_columns(&[*dir, -u, -v]);
    let x = system.lu().solve(&rhs)?;
    let (lambda, s, t) = (x[0], x[1], x[2]);
    (lambda > 0.0 && s.abs() <= 1.0 && t.abs() <= 1.0).then_some((lambda, [s, t]))
}

/// Resolve one endpoint: the pixel `pixel` of image `image` to a texel of a
/// patch and the 3D position that texel stands for.
///
/// The candidates are the observations in `image` whose square holds the
/// pixel. With one, it is the answer (`Alone`) unless it is seen from behind.
/// With several, the back-facing ones are excluded, the front layer is formed
/// by [`MeasurementEndpointOptions::layer_depth_factor`], and the candidate in
/// it with the lowest cost is picked, a tie going to the nearer depth and then
/// to the lower point index. The position is the texel's place on the point's
/// own patch, `X_p + s·u + t·v`.
///
/// One call visits every observation of the image once, and reads stored
/// readings only where two or more squares hold the pixel.
///
/// # Errors
///
/// An [`EndpointRefusal`] whose sentence names the remedy.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::reconstruction::measurement_endpoint::{
///     resolve_measurement_endpoint, EndpointRefusal, MeasurementEndpointOptions,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::SfmrReconstruction;
///
/// // The demo reconstruction stores no patch frames, so no square holds a pixel.
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let refusal = resolve_measurement_endpoint(
///     &edited,
///     0,
///     [960.0, 540.0],
///     &MeasurementEndpointOptions::default(),
/// )
/// .unwrap_err();
/// assert_eq!(refusal, EndpointRefusal::NoPatchFrames);
/// assert!(refusal.to_string().contains("--to-embedded-patches"));
/// ```
pub fn resolve_measurement_endpoint(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
    options: &MeasurementEndpointOptions,
) -> Result<EndpointResolution, EndpointRefusal> {
    let image_set = &edited.base.image_set;
    let i = image as usize;
    let Some(posed) = image_set.images.get(i) else {
        return Err(EndpointRefusal::NoSuchImage {
            image,
            image_count: image_set.image_count(),
        });
    };
    if !posed.is_posed() {
        return Err(EndpointRefusal::NoPose { image });
    }
    let outside = || EndpointRefusal::PixelOutsideImage { image, pixel };
    let camera = image_set
        .cameras
        .get(posed.camera_index as usize)
        .ok_or_else(outside)?;
    let inside = |x: f64, size: u32| x.is_finite() && x >= 0.0 && x < f64::from(size);
    if !(inside(pixel[0], camera.width) && inside(pixel[1], camera.height)) {
        return Err(outside());
    }
    let dir = image_set.world_ray(i, pixel).ok_or_else(outside)?;
    if !edited.has_patch_frames() {
        return Err(EndpointRefusal::NoPatchFrames);
    }

    let origin = posed.camera_center();
    let cam_from_world = RigidTransform::new(
        RotQuaternion::from_nalgebra(posed.quaternion_wxyz),
        posed.translation_xyz,
    );
    let intrinsics = image_set.image_intrinsics(i);
    let mut observed = false;
    let mut hits: Vec<Hit> = Vec::new();
    for index in edited.live_indexes() {
        let Some(view) = edited.point(index) else {
            continue;
        };
        let mut patch: Option<Option<OrientedPatch>> = None;
        for (k, obs) in view.observations().iter().enumerate() {
            if obs.image_index != image {
                continue;
            }
            observed = true;
            let Some(patch) = patch.get_or_insert_with(|| view.patch_frame()).as_ref() else {
                continue;
            };
            let keypoint = view
                .keypoint_xy(k)
                .map(|xy| [f64::from(xy[0]), f64::from(xy[1])]);
            let square = patch.seen_at_keypoint(&intrinsics, &cam_from_world, keypoint);
            let Some((lambda, [s, t])) = meet_square(&square, &origin, &dir) else {
                continue;
            };
            let (position, depth, back_facing) = if patch.w == 0.0 {
                let (direction, _) = patch.corner_homogeneous(s, t);
                let direction = direction.normalize();
                (
                    EndpointPosition::AtInfinity {
                        direction: [direction.x, direction.y, direction.z],
                    },
                    f64::INFINITY,
                    false,
                )
            } else {
                let x = patch.to_world(s, t);
                (
                    EndpointPosition::Finite {
                        position: [x.x, x.y, x.z],
                    },
                    lambda,
                    square.normal().dot(&dir) >= 0.0,
                )
            };
            hits.push(Hit {
                candidate: EndpointCandidate {
                    point: index,
                    observation: k,
                    patch_coords: [s, t],
                    position,
                    depth,
                    standing: CandidateStanding::Alone,
                },
                patch: patch.clone(),
                square,
                back_facing,
            });
        }
    }
    if !observed {
        return Err(EndpointRefusal::NoObservations { image });
    }
    match hits.len() {
        0 => Err(EndpointRefusal::NoCoveringSquare { image, pixel }),
        1 => {
            let hit = hits.pop().expect("one hit");
            let mut candidate = hit.candidate;
            if hit.back_facing {
                candidate.standing = CandidateStanding::Excluded(EndpointExclusion::BackFacing);
                return Err(EndpointRefusal::AllExcluded {
                    candidates: vec![candidate],
                });
            }
            Ok(EndpointResolution {
                chosen: 0,
                candidates: vec![candidate],
            })
        }
        _ => rank(edited, hits, &cam_from_world, options),
    }
}

/// The patch resolution `R` the localization length reads grid px at: the
/// stored readings' resolution, else the stored bitmaps' side, else `1`, a
/// grid px then being the patch's whole width.
fn grid_resolution(edited: &EditedReconstruction) -> usize {
    edited
        .observation_reading_options()
        .map(|o| o.resolution as usize)
        .or_else(|| {
            edited
                .base
                .point_set
                .patch_bitmaps_y_x_rgba
                .as_ref()
                .map(|b| b.shape()[1])
        })
        .filter(|&r| r > 0)
        .unwrap_or(1)
}

/// Pick among two or more squares: exclude the back-facing ones and those
/// behind the front layer, cost the rest, and pick the lowest.
fn rank(
    edited: &EditedReconstruction,
    hits: Vec<Hit>,
    cam_from_world: &RigidTransform,
    options: &MeasurementEndpointOptions,
) -> Result<EndpointResolution, EndpointRefusal> {
    // The nearest finite square that faces the camera sets the front layer; a
    // square at infinity is in it only where no finite square faces the camera.
    let nearest = hits
        .iter()
        .filter(|h| !h.back_facing && h.patch.w != 0.0)
        .min_by(|a, b| a.candidate.depth.total_cmp(&b.candidate.depth));
    let layer_end = nearest.map(|h| {
        let half = h.patch.half_extent[0].max(h.patch.half_extent[1]);
        h.candidate.depth + options.layer_depth_factor * half
    });
    let resolution = grid_resolution(edited);
    let max_radius = edited
        .observation_reading_options()
        .map(|o| f64::from(o.max_radius));

    let mut candidates = Vec::with_capacity(hits.len());
    for hit in hits {
        let mut candidate = hit.candidate;
        let in_layer = match layer_end {
            Some(end) => hit.patch.w != 0.0 && candidate.depth <= end,
            None => true,
        };
        candidate.standing = if hit.back_facing {
            CandidateStanding::Excluded(EndpointExclusion::BackFacing)
        } else if !in_layer {
            CandidateStanding::Excluded(EndpointExclusion::BehindFrontLayer)
        } else {
            let view = edited
                .point(candidate.point)
                .expect("a candidate's point is live");
            CandidateStanding::Ranked(cost_of(
                &view,
                candidate.observation,
                &hit.patch,
                &hit.square,
                candidate.patch_coords,
                cam_from_world,
                resolution,
                max_radius,
                options,
            ))
        };
        candidates.push(candidate);
    }

    let key = |c: &EndpointCandidate| match c.standing {
        CandidateStanding::Ranked(cost) => Some((cost.cost, c.depth, c.point, c.observation)),
        _ => None,
    };
    let chosen = candidates
        .iter()
        .enumerate()
        .filter_map(|(i, c)| key(c).map(|k| (i, k)))
        .min_by(|(_, a), (_, b)| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.cmp(&b.2))
                .then(a.3.cmp(&b.3))
        })
        .map(|(i, _)| i);
    match chosen {
        Some(chosen) => Ok(EndpointResolution { chosen, candidates }),
        None => Err(EndpointRefusal::AllExcluded { candidates }),
    }
}

/// The cost of one candidate in the front layer, from its observation's own
/// stored reading.
#[allow(clippy::too_many_arguments)]
fn cost_of(
    view: &PointView<'_>,
    observation: usize,
    patch: &OrientedPatch,
    square: &OrientedPatch,
    [s, t]: [f64; 2],
    cam_from_world: &RigidTransform,
    resolution: usize,
    max_radius: Option<f64>,
    options: &MeasurementEndpointOptions,
) -> EndpointCost {
    let rule = &options.reference_view;
    let row = view
        .observation_readings()
        .and_then(|rows| rows.get(observation));
    let read = row.and_then(stored_ellipse);
    let localization_read = read.is_some();
    let at_cap = read
        .as_ref()
        .is_some_and(|e| max_radius.is_some_and(|cap| e.axes[0] >= cap));
    let ellipse = read.unwrap_or_else(|| {
        SelfSimilarityEllipse::from_axes(
            [rule.unread_radius, rule.unread_radius],
            [false, false],
            f64::NAN,
        )
    });
    let localization = ellipse
        .on_patch(patch, resolution)
        .map_or(f64::NAN, |e| e.ellipse().axes[0]);
    let capped_radius = if at_cap {
        rule.capped_radius_factor
    } else {
        1.0
    };
    let placeholder = view.normal_confidence() == Some(0);
    let weight = options.offset_weight
        * if placeholder {
            options.placeholder_normal_factor
        } else {
            1.0
        };
    let rho = s.abs().max(t.abs());
    let offset = 1.0 + weight * rho * rho;
    let angle =
        viewing_angle(square, cam_from_world).map_or(1.0, |a| rule.angle_factor(a.angle_deg));
    let agreement = row
        .map(|r| f64::from(r.blur_matched_bitmap_zncc))
        .filter(|z| !z.is_nan())
        .map_or(1.0, |z| rule.agreement_penalty(1.0 - z));
    EndpointCost {
        localization,
        localization_read,
        capped_radius,
        offset,
        angle,
        agreement,
        cost: localization * capped_radius * offset * angle * agreement,
    }
}

/// Another observation of a candidate's point, at which an endpoint could sit
/// instead without moving on the surface: the pixel of that observation's
/// image that its square holds the candidate's texel at.
///
/// What [`endpoint_views`] offers.
#[derive(Debug, Clone, PartialEq)]
pub struct EndpointView {
    /// The candidate's point, its live index.
    pub point: u32,
    /// The observation's index within the point's own track.
    pub observation: usize,
    /// The observation's image.
    pub image: u32,
    /// The pixel of [`Self::image`] where the observation's square holds the
    /// texel.
    pub pixel: [f64; 2],
    /// The texel `(s, t)`: the candidate's.
    pub patch_coords: [f64; 2],
    /// `ℓ`, the semi-major axis of the observation's own stored
    /// self-similarity ellipse on the patch, as the resolver's cost reads it
    /// ([`EndpointCost::localization`]): scene units for a finite point,
    /// degrees for a point at infinity. `None` where the observation has no
    /// stored reading.
    pub localization: Option<f64>,
    /// The viewing angle of the observation's square from its camera, in
    /// degrees; `None` where the camera sits on the square's centre.
    pub viewing_angle_deg: Option<f64>,
    /// The point the endpoint resolves to at [`Self::image`] and
    /// [`Self::pixel`], or `None` where the resolver refuses it there.
    pub resolves_to: Option<u32>,
}

impl EndpointView {
    /// Whether moving the endpoint here makes it resolve to a point other
    /// than [`Self::point`], or to none: another square holds the pixel and
    /// wins the ranking, or none is eligible.
    pub fn resolves_elsewhere(&self) -> bool {
        self.resolves_to != Some(self.point)
    }
}

/// The pixels in other images that hold the same texel of each candidate's
/// patch as `(image, pixel)` does: where an endpoint can move to without
/// moving on the surface.
///
/// The candidates are the resolver's at `(image, pixel)`
/// ([`resolve_measurement_endpoint`]), chosen or not, excluded or not. For
/// each, every other observation of its point gives one view: the pixel where
/// that observation's square, the patch slid onto its keypoint and seen
/// through its image's own intrinsics, holds the candidate's texel `(s, t)`.
/// This is the resolver's mapping from a pixel to a texel run backwards, so
/// the resolver at a view's pixel finds that observation's square holding the
/// same texel. A view is dropped where its image has no pose, where the
/// texel's place projects outside `[0, width) × [0, height)` of the image or
/// behind its camera, or where the square faces away from that camera, the
/// resolver's back-facing test.
///
/// The views of the chosen candidate's point come first, then the others'
/// in the resolver's order of candidates, each in the order of the point's
/// track. Each carries the observation's stored `ℓ` and its viewing angle,
/// and the point the resolver picks at its pixel ([`EndpointView::resolves_to`]),
/// which differs from the view's own point where another square holds the
/// pixel there and wins.
///
/// One resolver call per view, run in parallel, so a call costs about as many
/// resolutions as the candidates' tracks have observations.
///
/// # Errors
///
/// The resolver's refusal at `(image, pixel)`, so a caller shows the same
/// sentence as for the endpoint itself.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::reconstruction::measurement_endpoint::{
///     endpoint_views, EndpointRefusal, MeasurementEndpointOptions,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::SfmrReconstruction;
///
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let refused = endpoint_views(&edited, 0, [960.0, 540.0], &MeasurementEndpointOptions::default());
/// assert_eq!(refused, Err(EndpointRefusal::NoPatchFrames));
/// ```
pub fn endpoint_views(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
    options: &MeasurementEndpointOptions,
) -> Result<Vec<EndpointView>, EndpointRefusal> {
    use rayon::prelude::*;

    let resolution = resolve_measurement_endpoint(edited, image, pixel, options)?;
    let chosen = resolution.chosen;
    let order =
        std::iter::once(chosen).chain((0..resolution.candidates.len()).filter(|&i| i != chosen));
    let grid = grid_resolution(edited);
    let mut views = Vec::new();
    for index in order {
        let candidate = &resolution.candidates[index];
        let Some(view) = edited.point(candidate.point) else {
            continue;
        };
        let Some(patch) = view.patch_frame() else {
            continue;
        };
        for (k, obs) in view.observations().iter().enumerate() {
            if k == candidate.observation {
                continue;
            }
            if let Some(offer) = view_at(edited, &view, &patch, k, obs.image_index, candidate, grid)
            {
                views.push(offer);
            }
        }
    }
    views.par_iter_mut().for_each(|offer| {
        offer.resolves_to = resolve_measurement_endpoint(edited, offer.image, offer.pixel, options)
            .ok()
            .map(|r| r.chosen().point);
    });
    Ok(views)
}

/// Observation `k` of `view`, in image `image`, as a view of `candidate`'s
/// texel, or `None` where it is dropped ([`endpoint_views`]). Its
/// `resolves_to` is left for the caller.
fn view_at(
    edited: &EditedReconstruction,
    view: &PointView<'_>,
    patch: &OrientedPatch,
    k: usize,
    image: u32,
    candidate: &EndpointCandidate,
    grid: usize,
) -> Option<EndpointView> {
    let image_set = &edited.base.image_set;
    let i = image as usize;
    let posed = image_set.images.get(i)?;
    if !posed.is_posed() {
        return None;
    }
    let camera = image_set.cameras.get(posed.camera_index as usize)?;
    let cam_from_world = RigidTransform::new(
        RotQuaternion::from_nalgebra(posed.quaternion_wxyz),
        posed.translation_xyz,
    );
    let intrinsics = image_set.image_intrinsics(i);
    let keypoint = view
        .keypoint_xy(k)
        .map(|xy| [f64::from(xy[0]), f64::from(xy[1])]);
    let square = patch.seen_at_keypoint(&intrinsics, &cam_from_world, keypoint);
    let [s, t] = candidate.patch_coords;
    let (xyz, w) = square.corner_homogeneous(s, t);
    if w != 0.0 {
        // The resolver's back-facing test, along this camera's ray to the
        // texel's place.
        let sight = (Point3::from(xyz) - posed.camera_center()).try_normalize(0.0)?;
        if square.normal().dot(&sight) >= 0.0 {
            return None;
        }
    }
    let local = cam_from_world.transform_point_homogeneous(xyz, w);
    let (x, y) = intrinsics.ray_to_pixel([local.x, local.y, local.z])?;
    let inside = |v: f64, size: u32| v.is_finite() && v >= 0.0 && v < f64::from(size);
    if !(inside(x, camera.width) && inside(y, camera.height)) {
        return None;
    }
    let localization = view
        .observation_readings()
        .and_then(|rows| rows.get(k))
        .and_then(stored_ellipse)
        .and_then(|ellipse| ellipse.on_patch(patch, grid))
        .map(|e| e.ellipse().axes[0]);
    Some(EndpointView {
        point: candidate.point,
        observation: k,
        image,
        pixel: [x, y],
        patch_coords: candidate.patch_coords,
        localization,
        viewing_angle_deg: viewing_angle(&square, &cam_from_world).map(|a| a.angle_deg),
        resolves_to: None,
    })
}

/// Resolve both endpoints of every distance measurement the reconstruction
/// stores ([`MeasurementsData::distance`](crate::MeasurementsData::distance)), in
/// stored order, with the distance between them in scene units where both
/// are finite. Empty where it stores none.
///
/// ```
/// use std::sync::Arc;
/// use sfmtool_core::reconstruction::measurement_endpoint::{
///     resolve_distance_measurements, MeasurementEndpointOptions,
/// };
/// use sfmtool_core::reconstruction::EditedReconstruction;
/// use sfmtool_core::SfmrReconstruction;
///
/// let edited = EditedReconstruction::new(Arc::new(SfmrReconstruction::demo(10)));
/// let opts = MeasurementEndpointOptions::default();
/// for (i, m) in resolve_distance_measurements(&edited, &opts).iter().enumerate() {
///     match (&m.endpoints[0], &m.endpoints[1], m.scene_distance) {
///         (Ok(_), Ok(_), Some(d)) => println!("measurement {i}: {d:.4} scene units"),
///         (Err(e), _, _) | (_, Err(e), _) => println!("measurement {i}: {e}"),
///         _ => println!("measurement {i}: an endpoint is at infinity"),
///     }
/// }
/// ```
pub fn resolve_distance_measurements(
    edited: &EditedReconstruction,
    options: &MeasurementEndpointOptions,
) -> Vec<ResolvedDistanceMeasurement> {
    let Some(m) = edited.measurements().distance.as_ref() else {
        return Vec::new();
    };
    (0..m.len())
        .map(|i| {
            // A row missing from a column (a value built by hand, which a load
            // refuses) reads as an image past the set or a pixel that is not a
            // number, and is refused as such.
            let endpoint = |k: usize| {
                let image = m.image_indexes.get((i, k)).copied().unwrap_or(u32::MAX);
                let xy = |c: usize| {
                    m.endpoints_xy
                        .get((i, k, c))
                        .map_or(f64::NAN, |&v| f64::from(v))
                };
                (image, [xy(0), xy(1)])
            };
            resolve_distance_endpoints(edited, [endpoint(0), endpoint(1)], options)
        })
        .collect()
}

/// Resolve the two endpoints of one distance measurement, each an image and
/// a pixel in it, with the distance between them in scene units where both
/// are finite: what [`resolve_distance_measurements`] does for each stored
/// row, for a measurement that is not stored, such as one being edited on the
/// bench (`specs/core/bench/editable-measurement.md`).
pub fn resolve_distance_endpoints(
    edited: &EditedReconstruction,
    endpoints: [(u32, [f64; 2]); 2],
    options: &MeasurementEndpointOptions,
) -> ResolvedDistanceMeasurement {
    let endpoints =
        endpoints.map(|(image, pixel)| resolve_measurement_endpoint(edited, image, pixel, options));
    let finite = |e: &Result<EndpointResolution, EndpointRefusal>| {
        e.as_ref().ok().and_then(|r| r.position().finite())
    };
    let scene_distance = match (finite(&endpoints[0]), finite(&endpoints[1])) {
        (Some(a), Some(b)) => Some((Vector3::from(a) - Vector3::from(b)).norm()),
        _ => None,
    };
    ResolvedDistanceMeasurement {
        endpoints,
        scene_distance,
    }
}
