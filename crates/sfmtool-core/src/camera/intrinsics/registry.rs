// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera model registry: every model declared once, both
//! [`SfmrCamera`] conversions derived from that declaration.
//!
//! # Why there are two camera types
//!
//! [`SfmrCamera`] is the **wire type** — a `model: String` plus a
//! `BTreeMap<String, f64>`, which is literally the shape of the
//! `cameras/metadata.json.zst` payload, with COLMAP's parameter names. It
//! lives in `sfmr-format`, which does not depend on this crate, so
//! `sfmr-colmap` and `camrig-format` can move cameras across the disk
//! boundary without acquiring the geometry layer.
//!
//! [`CameraModel`] is the **computation type** — a closed enum, so every
//! projection, Jacobian and distortion path is exhaustively matched by the
//! compiler.
//!
//! Neither absorbs the other. Pushing the enum down into `sfmr-format` would
//! invert the workspace layering; computing over the map would trade
//! compile-time exhaustiveness for runtime lookups. And the map can carry a
//! model this build does not know, which is exactly why
//! `TryFrom<&SfmrCamera>` is fallible while `From<&CameraIntrinsics>` is not.
//!
//! # The naming invariant
//!
//! For every fixed-arity model, **the struct field name is byte-identical to
//! the serialized parameter name**: `CameraModel::Pinhole { focal_length_x }`
//! serializes as `"focal_length_x"` and nothing else. That invariant holds
//! across all 13 fixed-arity variants in both directions, which is what lets
//! the `camera_models!` macro below recover every key from the field
//! identifier with `stringify!` instead of respelling it.
//!
//! For a fixed-arity model the serialized key is therefore never written as a
//! string at all: the name appears twice as an *identifier* — the field
//! declaration in [`CameraModel`] and the registry entry below — and the
//! compiler checks the two against each other, so they cannot drift. Before
//! this registry each key was a string literal in two independent 100+ line
//! matches that nothing compared, and a one-sided edit produced a camera that
//! wrote but would not read back, with nothing watching.
//!
//! The string literals that remain belong to the one `custom` model, whose
//! hand-written arms are exempt by design.
//!
//! See `specs/core/camera-model-registry.md`; the authoritative
//! human-readable model/parameter table is `specs/formats/sfmr-file-format.md`
//! §3.

use std::borrow::Cow;
use std::collections::BTreeMap;

use sfmr_format::SfmrCamera;

use crate::camera::distortion::bspline::MIN_BSPLINE_COEFFS;

use super::{CameraIntrinsics, CameraIntrinsicsError, CameraModel};

/// Declare every camera model once and derive the `SfmrCamera` boundary.
///
/// Two blocks:
///
/// - `fixed_arity` — models whose parameter list is a fixed set of `f64`
///   fields. `Variant => "MODEL_NAME" { field, … }`. Both serialization
///   directions are generated, with each key coming from `stringify!` on the
///   field identifier.
/// - `custom` — models whose parameter list is not a fixed set of `f64`
///   fields. `Variant => "MODEL_NAME" as CONST_NAME`. Only the name is
///   registered; serialization is hand-written and intercepts the conversion
///   before the fixed-arity path is reached.
///
/// The invocation is compiler-checked against the enum: a field the variant
/// lacks, a field it has but the table omits, or a variant missing from the
/// table altogether are each a build failure — the last because
/// `model_name` and `fixed_arity_params` would become non-exhaustive matches.
macro_rules! camera_models {
    (
        fixed_arity {
            $( $fx_variant:ident => $fx_name:literal { $( $field:ident ),+ $(,)? } ),+ $(,)?
        }
        custom {
            $( $cu_variant:ident => $cu_name:literal as $cu_const:ident ),+ $(,)?
        }
    ) => {
        $(
            /// The `.sfmr` model-name string for the custom model of the same
            /// name, so the dispatch site need not respell it.
            pub(crate) const $cu_const: &str = $cu_name;
        )+

        /// Number of camera models the registry knows about, fixed-arity and
        /// custom together.
        ///
        /// Because the generated matches are exhaustive, this is also the
        /// number of [`CameraModel`] variants. Its whole job is to let the
        /// test corpus assert its own completeness — a variant cannot be
        /// registered (which the compiler forces) and then silently left
        /// untested — so it is test-only rather than dead weight in a release
        /// build.
        #[cfg(test)]
        pub(crate) const MODEL_COUNT: usize =
            [$( $fx_name ),+].len() + [$( $cu_name ),+].len();

        impl CameraModel {
            /// Return the COLMAP model name string for this camera model.
            pub fn model_name(&self) -> &'static str {
                match self {
                    $( CameraModel::$fx_variant { .. } => $fx_name, )+
                    $( CameraModel::$cu_variant { .. } => $cu_name, )+
                }
            }
        }

        /// Serialize a fixed-arity model's parameters, keyed by field name.
        ///
        /// Custom models are intercepted by the caller and reaching this with
        /// one is a bug, not bad input.
        fn fixed_arity_params(model: &CameraModel) -> BTreeMap<String, f64> {
            let mut parameters = BTreeMap::new();
            match model {
                $(
                    CameraModel::$fx_variant { $( $field ),+ } => {
                        $( parameters.insert(stringify!($field).to_string(), *$field); )+
                    }
                )+
                $(
                    CameraModel::$cu_variant { .. } => unreachable!(
                        "`{}` has a variable-length parameter list and is serialized by \
                         its own arm in `From<&CameraIntrinsics> for SfmrCamera`",
                        $cu_name
                    ),
                )+
            }
            parameters
        }

        /// A fixed-arity model's parameter names in declaration order — the
        /// same field identifiers `fixed_arity_params` keys its map with, in
        /// the order the registry lists them rather than the order a
        /// `BTreeMap` hands them back.
        ///
        /// Custom models are intercepted by [`CameraModel::parameter_names`]
        /// and reaching this with one is a bug, not bad input.
        fn fixed_arity_param_names(model: &CameraModel) -> &'static [&'static str] {
            match model {
                $(
                    CameraModel::$fx_variant { .. } => &[$( stringify!($field) ),+],
                )+
                $(
                    CameraModel::$cu_variant { .. } => unreachable!(
                        "`{}` has a variable-length parameter list and names its \
                         parameters in its own arm of `CameraModel::parameter_names`",
                        $cu_name
                    ),
                )+
            }
        }

        /// Deserialize a fixed-arity model, reading each field by its own name.
        ///
        /// A name that is not registered is [`CameraIntrinsicsError::UnknownModel`];
        /// custom model names reach here only if the caller failed to
        /// intercept them, and are reported the same way.
        fn fixed_arity_from_sfmr(
            model: &str,
            params: &BTreeMap<String, f64>,
        ) -> Result<CameraModel, CameraIntrinsicsError> {
            match model {
                $(
                    $fx_name => Ok(CameraModel::$fx_variant {
                        $( $field: get_param(params, model, stringify!($field))?, )+
                    }),
                )+
                other => Err(CameraIntrinsicsError::UnknownModel(other.to_string())),
            }
        }
    };
}

camera_models! {
    fixed_arity {
        Pinhole => "PINHOLE" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
        },
        SimplePinhole => "SIMPLE_PINHOLE" {
            focal_length, principal_point_x, principal_point_y,
        },
        SimpleRadial => "SIMPLE_RADIAL" {
            focal_length, principal_point_x, principal_point_y, radial_distortion_k1,
        },
        Radial => "RADIAL" {
            focal_length, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
        },
        OpenCV => "OPENCV" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
            tangential_distortion_p1, tangential_distortion_p2,
        },
        OpenCVFisheye => "OPENCV_FISHEYE" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
            radial_distortion_k3, radial_distortion_k4,
        },
        SimpleRadialFisheye => "SIMPLE_RADIAL_FISHEYE" {
            focal_length, principal_point_x, principal_point_y, radial_distortion_k1,
        },
        RadialFisheye => "RADIAL_FISHEYE" {
            focal_length, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
        },
        ThinPrismFisheye => "THIN_PRISM_FISHEYE" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
            tangential_distortion_p1, tangential_distortion_p2,
            radial_distortion_k3, radial_distortion_k4,
            thin_prism_sx1, thin_prism_sy1,
        },
        RadTanThinPrismFisheye => "RAD_TAN_THIN_PRISM_FISHEYE" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
            radial_distortion_k0, radial_distortion_k1, radial_distortion_k2,
            radial_distortion_k3, radial_distortion_k4, radial_distortion_k5,
            tangential_distortion_p0, tangential_distortion_p1,
            thin_prism_s0, thin_prism_s1, thin_prism_s2, thin_prism_s3,
        },
        FullOpenCV => "FULL_OPENCV" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
            radial_distortion_k1, radial_distortion_k2,
            tangential_distortion_p1, tangential_distortion_p2,
            radial_distortion_k3, radial_distortion_k4,
            radial_distortion_k5, radial_distortion_k6,
        },
        Equirectangular => "EQUIRECTANGULAR" {
            focal_length_x, focal_length_y, principal_point_x, principal_point_y,
        },
        EquidistantFisheye => "EQUIDISTANT_FISHEYE" {
            focal_length, principal_point_x, principal_point_y,
        },
    }
    custom {
        SfmtoolFisheye => "SFMTOOL_FISHEYE" as SFMTOOL_FISHEYE,
        SfmtoolPinhole => "SFMTOOL_PINHOLE" as SFMTOOL_PINHOLE,
    }
}

impl CameraModel {
    /// This model's parameter names in **declaration order** — the order the
    /// registry declares them in, which is the order a parameter table should
    /// print them in.
    ///
    /// [`SfmrCamera`]'s `parameters` is a `BTreeMap`, so reading the names off
    /// a serialized camera gives lexicographic order instead. That is wrong
    /// twice over: it separates related terms (a principal point lands between
    /// two focal lengths), and it orders the spline models' `bspline_c10`
    /// before `bspline_c2`.
    ///
    /// The result is always a permutation of the keys
    /// `SfmrCamera::from(&CameraIntrinsics)` writes for the same camera, which
    /// is what keeps a table built from it from silently dropping a parameter
    /// when a model gains one.
    ///
    /// Owned `Cow`s rather than a `&'static [&'static str]` because the two
    /// spline models' trailing `bspline_c{i}` names depend on the coefficient
    /// count and so cannot be static. Every fixed-arity model borrows
    /// throughout and allocates only the `Vec`.
    pub fn parameter_names(&self) -> Vec<Cow<'static, str>> {
        // Custom models first, exactly as `From<&CameraIntrinsics>` does: a
        // variable-length parameter list is not recoverable from field
        // identifiers, so it never reaches the generated table.
        match self {
            CameraModel::SfmtoolFisheye { bspline, .. } => {
                spline_parameter_names("bspline_theta_max", bspline.len())
            }
            CameraModel::SfmtoolPinhole { bspline, .. } => {
                spline_parameter_names("bspline_rho_max", bspline.len())
            }
            fixed => fixed_arity_param_names(fixed)
                .iter()
                .copied()
                .map(Cow::Borrowed)
                .collect(),
        }
    }
}

/// The parameter names of a spline model in declaration order: the three named
/// parameters both carry, the domain end under `domain_end_key`, then the
/// declared coefficient count and `bspline_c0..bspline_c{n−1}` in **index**
/// order.
///
/// The mirror of what the two spline arms of `From<&CameraIntrinsics> for
/// SfmrCamera` and [`insert_bspline`] write, in the order a `BTreeMap` cannot
/// keep. The permutation test over the registry corpus is what holds the two
/// lists together, since these names are string literals on both sides.
fn spline_parameter_names(domain_end_key: &'static str, n: usize) -> Vec<Cow<'static, str>> {
    let mut names: Vec<Cow<'static, str>> = vec![
        Cow::Borrowed("focal_length"),
        Cow::Borrowed("principal_point_x"),
        Cow::Borrowed("principal_point_y"),
        Cow::Borrowed(domain_end_key),
        Cow::Borrowed("bspline_coeff_count"),
    ];
    names.extend((0..n).map(|i| Cow::Owned(format!("bspline_c{i}"))));
    names
}

/// Read one parameter out of an [`SfmrCamera`] parameter map.
fn get_param(
    params: &BTreeMap<String, f64>,
    model: &str,
    name: &str,
) -> Result<f64, CameraIntrinsicsError> {
    params
        .get(name)
        .copied()
        .ok_or_else(|| CameraIntrinsicsError::MissingParameter {
            model: model.to_string(),
            parameter: name.to_string(),
        })
}

/// Read the variable-length radial spline both sfmtool spline models carry:
/// the domain end under `domain_end_key` (`bspline_theta_max` for the fisheye,
/// `bspline_rho_max` for the pinhole) and the coefficients
/// `bspline_c0..bspline_c{N−1}`. Returns `(domain_end, coefficients)`.
///
/// `bspline_coeff_count` DECLARES the number of coefficients, and the map must
/// carry exactly `bspline_c0..bspline_c{N−1}`. An index below the declared
/// count that is absent is a hole; a `bspline_c*` key at or beyond it is a
/// stray. Either way the parameter map is corrupt — counting the keys instead
/// would silently accept both as a shorter or longer spline.
///
/// The count must also be a length the models define: zero (the empty spline,
/// the identity) or at least [`MIN_BSPLINE_COEFFS`], since a clamped cubic
/// basis needs them. Exactly one coefficient is no spline they can evaluate,
/// so it is rejected rather than silently read as the identity.
///
/// The domain end must be a real interval: zero, negative or non-finite leaves
/// the basis nothing to live on (`+∞` would put every knot at infinity), so it
/// is corrupt rather than a camera with a degenerate spline.
fn get_bspline(
    params: &BTreeMap<String, f64>,
    model: &str,
    domain_end_key: &str,
) -> Result<(f64, Vec<f64>), CameraIntrinsicsError> {
    let declared = get_param(params, model, "bspline_coeff_count")?;
    if !declared.is_finite()
        || declared < 0.0
        || declared.fract() != 0.0
        || (declared > 0.0 && declared < MIN_BSPLINE_COEFFS as f64)
    {
        return Err(CameraIntrinsicsError::InvalidParameter {
            model: model.to_string(),
            parameter: "bspline_coeff_count".to_string(),
        });
    }
    let n = declared as usize;
    // Cap the reservation by the map size: the declared count is untrusted,
    // and any index it claims beyond what the map holds errors out as a
    // missing coefficient below.
    let mut bspline = Vec::with_capacity(n.min(params.len()));
    for i in 0..n {
        bspline.push(get_param(params, model, &format!("bspline_c{i}"))?);
    }
    let stray = params.keys().find(|k| {
        // `bspline_coeff_count` shares the coefficient prefix but is the
        // declaration itself, never a coefficient.
        k.as_str() != "bspline_coeff_count"
            && k.strip_prefix("bspline_c")
                .is_some_and(|i| !matches!(i.parse::<usize>(), Ok(i) if i < n))
    });
    if let Some(stray) = stray {
        return Err(CameraIntrinsicsError::InvalidParameter {
            model: model.to_string(),
            parameter: stray.clone(),
        });
    }
    let domain_end = get_param(params, model, domain_end_key)?;
    if !(domain_end > 0.0 && domain_end.is_finite()) {
        return Err(CameraIntrinsicsError::InvalidParameter {
            model: model.to_string(),
            parameter: domain_end_key.to_string(),
        });
    }
    Ok((domain_end, bspline))
}

/// Write the declared coefficient count and one key per coefficient — the
/// half of the spline head that is identical across both spline models (the
/// domain end, whose key name differs, is written by the caller). Read back by
/// [`get_bspline`].
fn insert_bspline(parameters: &mut BTreeMap<String, f64>, bspline: &[f64]) {
    parameters.insert("bspline_coeff_count".to_string(), bspline.len() as f64);
    for (i, c) in bspline.iter().enumerate() {
        parameters.insert(format!("bspline_c{i}"), *c);
    }
}

impl TryFrom<&SfmrCamera> for CameraIntrinsics {
    type Error = CameraIntrinsicsError;

    fn try_from(cam: &SfmrCamera) -> Result<Self, Self::Error> {
        let p = &cam.parameters;
        let m = cam.model.as_str();

        // Custom models first: their parameter lists are not derivable from
        // field identifiers, so they never reach the fixed-arity table.
        let model = match m {
            SFMTOOL_FISHEYE => {
                let (bspline_theta_max, bspline) = get_bspline(p, m, "bspline_theta_max")?;
                CameraModel::SfmtoolFisheye {
                    focal_length: get_param(p, m, "focal_length")?,
                    principal_point_x: get_param(p, m, "principal_point_x")?,
                    principal_point_y: get_param(p, m, "principal_point_y")?,
                    bspline_theta_max,
                    bspline,
                }
            }
            SFMTOOL_PINHOLE => {
                let (bspline_rho_max, bspline) = get_bspline(p, m, "bspline_rho_max")?;
                CameraModel::SfmtoolPinhole {
                    focal_length: get_param(p, m, "focal_length")?,
                    principal_point_x: get_param(p, m, "principal_point_x")?,
                    principal_point_y: get_param(p, m, "principal_point_y")?,
                    bspline_rho_max,
                    bspline,
                }
            }
            other => fixed_arity_from_sfmr(other, p)?,
        };

        Ok(CameraIntrinsics {
            model,
            width: cam.width,
            height: cam.height,
        })
    }
}

impl From<&CameraIntrinsics> for SfmrCamera {
    fn from(cam: &CameraIntrinsics) -> Self {
        // Matching the custom models here — rather than inside
        // `fixed_arity_params` — is what keeps this exhaustive over the enum:
        // a new variant that the registry does not cover fails to compile in
        // the generated match, and one it does cover lands in `fixed`.
        let parameters = match &cam.model {
            CameraModel::SfmtoolFisheye {
                focal_length,
                principal_point_x,
                principal_point_y,
                bspline_theta_max,
                bspline,
            } => {
                let mut parameters = BTreeMap::new();
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("bspline_theta_max".to_string(), *bspline_theta_max);
                insert_bspline(&mut parameters, bspline);
                parameters
            }
            CameraModel::SfmtoolPinhole {
                focal_length,
                principal_point_x,
                principal_point_y,
                bspline_rho_max,
                bspline,
            } => {
                let mut parameters = BTreeMap::new();
                parameters.insert("focal_length".to_string(), *focal_length);
                parameters.insert("principal_point_x".to_string(), *principal_point_x);
                parameters.insert("principal_point_y".to_string(), *principal_point_y);
                parameters.insert("bspline_rho_max".to_string(), *bspline_rho_max);
                insert_bspline(&mut parameters, bspline);
                parameters
            }
            fixed => fixed_arity_params(fixed),
        };

        SfmrCamera {
            model: cam.model_name().to_string(),
            width: cam.width,
            height: cam.height,
            parameters,
        }
    }
}
