// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Image Detail panel's controls, as one document an agent reads and
//! writes.
//!
//! Everything the panel's toolbar decides — the seven feature overlay modes,
//! the three filters on the features, and the intrinsics layer with its own
//! sub-toggles — is scene-level state on `AppState` (`feature_display`,
//! `intrinsics_display`) rather than a property of any image or
//! reconstruction. That is why it is one document and one pair of tools rather
//! than fields on `set_reconstruction_display`: it describes how the panel
//! looks at *whatever* is selected, and a `screenshot` of `image_detail` shows
//! whichever mode was last picked.
//!
//! Three things about the shape here are worth knowing before reading it:
//!
//! - **Every refusal is at the parse.** The vocabularies are static — seven
//!   modes, two ladders, the bounds on a size filter — so [`parse_change`]
//!   validates the whole call before a `Command` exists and [`set`] cannot
//!   fail. A call naming a good field and a bad one changes nothing.
//! - **`feature_size_px` is one thing here because it is one checkbox in the
//!   toolbar.** See [`FeatureSize`].
//! - **The Action Log entries are not written here.** They come from
//!   [`crate::state::record_image_detail_changes`], the one differ the panel's
//!   own frame goes through as well, which is what keeps the human's row and
//!   the agent's row for one control identical.

use serde_json::{json, Value};

use super::tools::Args;
use super::{ImageDetailDisplayChange, IntrinsicsChange, JsonReply, ToolError};
use crate::state::{
    record_image_detail_changes, AppState, FeatureDisplaySettings, ImageDetailDisplay,
    IntrinsicsDisplaySettings, OverlayMode,
};

/// A `feature_size_px` a call carried: the two bounds, or the filter off.
///
/// One value rather than two optional bounds, because the toolbar re-derives
/// `min_feature_size` and `max_feature_size` from its single `Min/max size:`
/// checkbox **every frame** — ticked, both are written from the persisted drag
/// values; unticked, both are cleared. So the two are never independently
/// `Some` while the panel is open, and a tool that let an agent set one
/// without the other would have its half undone by the next frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FeatureSize {
    /// Both bounds, in pixels. Writing this writes **all four** fields — the
    /// two options and the two drag values — so the toolbar's next frame
    /// re-derives exactly what the agent asked for.
    Between { min: f32, max: f32 },
    /// The filter off, which is what unticking the checkbox does: the two
    /// options are cleared and the drag values stay where they were.
    Off,
}

/// `get_image_detail_display`: the whole document, no arguments.
pub(super) fn get(state: &AppState) -> JsonReply {
    Ok(document(state))
}

/// `set_image_detail_display`: write the fields the call named, record what
/// that changed, and answer with the whole document.
///
/// The reply is exactly what [`get`] would return, so the agent reads back the
/// state rather than the fields it happened to set — the same rule as
/// `set_reconstruction_display`.
pub(super) fn set(state: &mut AppState, change: &ImageDetailDisplayChange) -> JsonReply {
    let before = ImageDetailDisplay::snapshot(&state.feature_display, &state.intrinsics_display);
    apply_feature(&mut state.feature_display, change);
    apply_intrinsics(&mut state.intrinsics_display, &change.intrinsics);
    let after = ImageDetailDisplay::snapshot(&state.feature_display, &state.intrinsics_display);
    record_image_detail_changes(&mut state.action_log, &before, &after);
    Ok(document(state))
}

fn apply_feature(feature: &mut FeatureDisplaySettings, change: &ImageDetailDisplayChange) {
    if let Some(mode) = change.overlay_mode {
        feature.overlay_mode = mode;
    }
    if let Some(max_features) = change.max_features {
        feature.max_features = max_features;
    }
    match change.feature_size_px {
        None => {}
        Some(FeatureSize::Between { min, max }) => {
            feature.min_feature_size = Some(min);
            feature.max_feature_size = Some(max);
            feature.min_feature_size_value = min;
            feature.max_feature_size_value = max;
        }
        Some(FeatureSize::Off) => {
            feature.min_feature_size = None;
            feature.max_feature_size = None;
        }
    }
    if let Some(tracked_only) = change.tracked_only {
        feature.tracked_only = tracked_only;
    }
}

fn apply_intrinsics(intrinsics: &mut IntrinsicsDisplaySettings, change: &IntrinsicsChange) {
    if let Some(enabled) = change.enabled {
        intrinsics.enabled = enabled;
    }
    if let Some(axes) = change.axes {
        intrinsics.axes = axes;
    }
    if let Some(rings) = change.rings {
        intrinsics.rings = rings;
    }
    if let Some(distortion) = change.distortion {
        intrinsics.distortion = distortion;
    }
    if let Some(distortion_scale) = change.distortion_scale {
        intrinsics.distortion_scale = distortion_scale;
    }
    if let Some(grid_cols) = change.grid_cols {
        intrinsics.grid_cols = grid_cols;
    }
}

/// The document both tools answer with.
///
/// The feature overlay and its filters at the top level and the intrinsics
/// layer as an `intrinsics` sub-block, which is how the toolbar draws them: the
/// feature controls in a row, the layer behind one checkbox and a gear.
fn document(state: &AppState) -> Value {
    let feature = &state.feature_display;
    let intrinsics = &state.intrinsics_display;
    json!({
        "image_detail_display": {
            "overlay_mode": feature.overlay_mode.wire_name(),
            "max_features": feature.max_features,
            "feature_size_px": feature_size_px(feature),
            "tracked_only": feature.tracked_only,
            "intrinsics": {
                "enabled": intrinsics.enabled,
                "axes": intrinsics.axes,
                "rings": intrinsics.rings,
                "distortion": intrinsics.distortion,
                "distortion_scale": intrinsics.distortion_scale.map(f64::from),
                "grid_cols": intrinsics.grid_cols,
            },
        }
    })
}

/// The size filter as one object, or `null` for no size filter.
///
/// A bound somehow set on its own falls back to the drag value beside it,
/// which is the number the toolbar would show for it — the object always
/// carries both, because both is what setting it takes.
fn feature_size_px(feature: &FeatureDisplaySettings) -> Value {
    match (feature.min_feature_size, feature.max_feature_size) {
        (None, None) => Value::Null,
        (min, max) => json!({
            "min": f64::from(min.unwrap_or(feature.min_feature_size_value)),
            "max": f64::from(max.unwrap_or(feature.max_feature_size_value)),
        }),
    }
}

/// Every argument `set_image_detail_display` takes, and the whole of what it
/// refuses.
///
/// Called from [`super::tools::parse`], so a bad mode name, an off-ladder
/// value or an inverted size filter is turned away before a `Command` exists —
/// which is what makes a refusal atomic without [`set`] needing a rollback.
pub(super) fn parse_change(args: &Args) -> Result<ImageDetailDisplayChange, ToolError> {
    args.reject_unknown(&[
        "overlay_mode",
        "max_features",
        "feature_size_px",
        "tracked_only",
        "intrinsics",
    ])?;
    let change = ImageDetailDisplayChange {
        overlay_mode: parse_overlay_mode(args)?,
        // Doubly optional: absent leaves the cap alone, an explicit null lifts
        // it. Refused at 0, since "show no features" is what
        // `overlay_mode: "none"` says, and the `Max:` dropdown cannot show a
        // second spelling of it.
        max_features: match args.get("max_features") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(_) => {
                let max = args.required_usize("max_features")?;
                if max == 0 {
                    return Err(args.error(
                        "wants max_features to be 1 or more — \"show no features\" is \
                         overlay_mode \"none\".",
                    ));
                }
                Some(Some(max))
            }
        },
        feature_size_px: parse_feature_size(args)?,
        tracked_only: args.optional_bool("tracked_only")?,
        intrinsics: parse_intrinsics(args)?,
    };
    if change == ImageDetailDisplayChange::default() {
        return Err(args.error("was given nothing to change."));
    }
    Ok(change)
}

fn parse_overlay_mode(args: &Args) -> Result<Option<OverlayMode>, ToolError> {
    let Some(name) = args.optional_string("overlay_mode")? else {
        return Ok(None);
    };
    OverlayMode::from_wire_name(&name).map(Some).ok_or_else(|| {
        args.error(format!(
            "does not know the overlay mode {name:?} — the modes are {}.",
            OverlayMode::all_wire_names()
        ))
    })
}

/// The size filter, as an object of two bounds or an explicit `null`.
fn parse_feature_size(args: &Args) -> Result<Option<FeatureSize>, ToolError> {
    let value = match args.get("feature_size_px") {
        None => return Ok(None),
        Some(Value::Null) => return Ok(Some(FeatureSize::Off)),
        Some(value) => value,
    };
    let map = value.as_object().ok_or_else(|| {
        args.error("wants feature_size_px to be an object with min and max, or null.")
    })?;
    let inner = Args::new("set_image_detail_display.feature_size_px", map);
    inner.reject_unknown(&["min", "max"])?;
    let min = inner.required_f64("min")?;
    let max = inner.required_f64("max")?;
    for (name, bound) in [("min", min), ("max", max)] {
        if !bound.is_finite() || bound < 0.0 {
            return Err(inner.error(format!(
                "wants {name} to be a size in pixels, zero or more — got {bound}."
            )));
        }
    }
    if min > max {
        return Err(inner.error(format!(
            "was given min {min} above max {max}, which selects no feature at all."
        )));
    }
    Ok(Some(FeatureSize::Between {
        min: min as f32,
        max: max as f32,
    }))
}

fn parse_intrinsics(args: &Args) -> Result<IntrinsicsChange, ToolError> {
    let value = match args.get("intrinsics") {
        None | Some(Value::Null) => return Ok(IntrinsicsChange::default()),
        Some(value) => value,
    };
    let map = value
        .as_object()
        .ok_or_else(|| args.error("wants intrinsics to be an object of the layer's controls."))?;
    let inner = Args::new("set_image_detail_display.intrinsics", map);
    inner.reject_unknown(&[
        "enabled",
        "axes",
        "rings",
        "distortion",
        "distortion_scale",
        "grid_cols",
    ])?;
    Ok(IntrinsicsChange {
        enabled: inner.optional_bool("enabled")?,
        axes: inner.optional_bool("axes")?,
        rings: inner.optional_bool("rings")?,
        distortion: inner.optional_bool("distortion")?,
        // Doubly optional, and off its ladder it is refused: those are the
        // exaggerations the gear popup offers, and a value the popup cannot
        // show is a value the human cannot see they are looking at.
        distortion_scale: match inner.get("distortion_scale") {
            None => None,
            Some(Value::Null) => Some(None),
            Some(_) => {
                let scale = inner.required_f64("distortion_scale")? as f32;
                if !IntrinsicsDisplaySettings::SCALE_LADDER.contains(&scale) {
                    return Err(inner.error(format!(
                        "does not offer the distortion scale {scale} — the ladder is {}, or null \
                         for auto.",
                        ladder(&IntrinsicsDisplaySettings::SCALE_LADDER)
                    )));
                }
                Some(Some(scale))
            }
        },
        grid_cols: match inner.optional_usize("grid_cols")? {
            None => None,
            Some(cols) => {
                if !IntrinsicsDisplaySettings::GRID_LADDER.contains(&cols) {
                    return Err(inner.error(format!(
                        "does not offer the grid density {cols} — the ladder is {}.",
                        ladder(&IntrinsicsDisplaySettings::GRID_LADDER)
                    )));
                }
                Some(cols)
            }
        },
    })
}

/// A ladder as an error message lists it: `1, 2, 3, 5, 10, 20, 50`.
fn ladder<T: std::fmt::Display>(values: &[T]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

// ── Schema fragments ────────────────────────────────────────────────────
//
// Here rather than in `tools`, beside the parse that has to accept every one
// of them: the two are one statement, and the catalog walk in `mcp::tests`
// checks that they stay one.

/// The `intrinsics` sub-block of `set_image_detail_display`.
pub(super) fn intrinsics_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "description":
            "The intrinsics overlay layer, which composes with whichever feature overlay_mode is \
             active — including \"none\", for the camera model alone. Every key is optional.",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description":
                    "Draw the layer at all: the principal point always, and whatever of axes, \
                     rings and distortion is on. On by default.",
            },
            "axes": {
                "type": "boolean",
                "description": "The angular axes through the principal point, with their ticks.",
            },
            "rings": {
                "type": "boolean",
                "description": "Iso-angle rings, at the same angular ladder as the axis ticks.",
            },
            "distortion": {
                "type": "boolean",
                "description":
                    "The displacement field. Ignored where the camera model has no distortion.",
            },
            "distortion_scale": {
                "type": ["number", "null"],
                "enum": scale_ladder_with_null(),
                "description":
                    "How far the displacement arrows are exaggerated, from the ladder the gear \
                     popup offers, or null to fit the scale to the lens automatically.",
            },
            "grid_cols": {
                "type": "integer",
                "enum": IntrinsicsDisplaySettings::GRID_LADDER,
                "description": "Arrows across the image width, from the popup's own ladder.",
            },
        },
    })
}

/// The `feature_size_px` argument: both bounds, or `null` for no filter.
pub(super) fn feature_size_schema() -> Value {
    json!({
        "type": ["object", "null"],
        "additionalProperties": false,
        "required": ["min", "max"],
        "description":
            "Show only features whose size falls between these two, in pixels, or null to turn \
             the filter off. Both bounds together, because the toolbar's one checkbox derives \
             both from its two persisted values every frame.",
        "properties": {
            "min": { "type": "number", "minimum": 0 },
            "max": { "type": "number", "minimum": 0 },
        },
    })
}

/// The ladder plus `null`, as `distortion_scale`'s enum spells it.
fn scale_ladder_with_null() -> Vec<Value> {
    IntrinsicsDisplaySettings::SCALE_LADDER
        .iter()
        .map(|scale| json!(f64::from(*scale)))
        .chain(std::iter::once(Value::Null))
        .collect()
}
