// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `get_viewer_3d_display` and `set_viewer_3d_display`: the 3D viewport's
//! display controls, the checkboxes and sliders of its HUD, as one flat
//! document.
//!
//! The fields, their wire names, their ranges and their Action Log texts all
//! come from [`crate::viewer_3d::display::Field`], the list the HUD draws its
//! widgets from, so a range here and a slider's range are one constant, and the
//! row an agent's change leaves is the row a person's click leaves.
//!
//! As with `set_image_detail_display`, **every refusal is at the parse**: the
//! ranges are static, so [`parse_change`] checks the whole call before a
//! `Command` exists and [`set`] cannot fail. A call naming a good field and a
//! bad one changes nothing.

use serde_json::{json, Map, Value};

use super::tools::Args;
use super::{JsonReply, ToolError, Viewer3dDisplayChange};
use crate::state::AppState;
use crate::viewer_3d::display::{
    record_viewer_3d_display_changes, Control, Field, FieldValue, Viewer3dDisplay,
};
use crate::viewer_3d::Viewer3D;

/// `get_viewer_3d_display`: the whole document, no arguments.
pub(super) fn get(state: &AppState, viewer: &Viewer3D) -> JsonReply {
    Ok(document(state, viewer))
}

/// `set_viewer_3d_display`: write the fields the call named, record what that
/// changed, and answer with the whole document.
///
/// The record is the diff of the document before and after the write, so a
/// field set to the value it had leaves no row.
pub(super) fn set(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    change: &Viewer3dDisplayChange,
) -> JsonReply {
    let before = Viewer3dDisplay::snapshot(state, viewer);
    for &(field, value) in &change.fields {
        field.write(state, viewer, value);
    }
    let after = Viewer3dDisplay::snapshot(state, viewer);
    record_viewer_3d_display_changes(&mut state.action_log, &before, &after);
    Ok(document(state, viewer))
}

/// The document both tools answer with: every field under its wire name, in
/// the HUD's order.
fn document(state: &AppState, viewer: &Viewer3D) -> Value {
    let fields: Map<String, Value> = Viewer3dDisplay::snapshot(state, viewer)
        .fields()
        .map(|(field, value)| (field.wire_name().to_string(), wire_value(value)))
        .collect();
    json!({ "viewer_3d_display": fields })
}

/// A value as the wire carries it.
///
/// A number goes through its shortest `f32` spelling, so a point size of 1.5
/// reads `1.5` and an EDL width of 2.4 reads `2.4` rather than the `f64` the
/// `f32` widens to, `2.4000000953674316`.
fn wire_value(value: FieldValue) -> Value {
    match value {
        FieldValue::Flag(on) => json!(on),
        FieldValue::Number(number) => {
            let widened = number
                .to_string()
                .parse::<f64>()
                .unwrap_or(f64::from(number));
            json!(widened)
        }
    }
}

/// Every argument `set_viewer_3d_display` takes, and the whole of what it
/// refuses: a value of the wrong type, a number that is not finite or is
/// outside its slider's range, and a call that names nothing.
///
/// A number inside the range is rounded to the decimals its slider shows,
/// which is what the slider itself does to the value it holds whenever the HUD
/// is drawn.
pub(super) fn parse_change(args: &Args) -> Result<Viewer3dDisplayChange, ToolError> {
    let mut fields = Vec::new();
    for field in Field::ALL {
        let name = field.wire_name();
        let value = match field.control() {
            Control::Checkbox => args.optional_bool(name)?.map(FieldValue::Flag),
            Control::Slider(range) => match args.optional_f64(name)? {
                None => None,
                Some(number) => {
                    if !range.contains(number) {
                        return Err(args.error(format!(
                            "wants {name} to be a number from {} — got {number}.",
                            range.describe()
                        )));
                    }
                    Some(FieldValue::Number(range.round(number)))
                }
            },
        };
        if let Some(value) = value {
            fields.push((field, value));
        }
    }
    if fields.is_empty() {
        return Err(args.error("was given nothing to change."));
    }
    Ok(Viewer3dDisplayChange { fields })
}

/// The schema of `set_viewer_3d_display`'s arguments: one optional property per
/// field, a slider's carrying its range as `minimum` and `maximum`.
///
/// Built from the same list the parse walks, so the catalog walk in
/// `mcp::tests` holds the two to one statement.
pub(super) fn properties() -> Vec<(&'static str, Value)> {
    Field::ALL
        .into_iter()
        .map(|field| (field.wire_name(), property(field)))
        .collect()
}

fn property(field: Field) -> Value {
    let description = description(field);
    match field.control() {
        Control::Checkbox => json!({ "type": "boolean", "description": description }),
        Control::Slider(range) => {
            let (minimum, maximum) = range.decimal_ends();
            json!({
                "type": "number",
                "minimum": minimum,
                "maximum": maximum,
                "description": format!(
                    "{description} From {}; rounded to {} decimal{}, as the HUD's slider shows it.",
                    range.describe(),
                    range.decimals,
                    if range.decimals == 1 { "" } else { "s" },
                ),
            })
        }
    }
}

/// What each field does, for the schema.
fn description(field: Field) -> &'static str {
    match field {
        Field::ShowPoints => "Draw the 3D points (the HUD's Points).",
        Field::ShowCameraImages => "Draw the camera frustums and their images (Camera Images).",
        Field::ShowGrid => "Draw the ground grid (Grid).",
        Field::ShowPatches => {
            "Draw patch surfels (Patches). Settable without patch data, where it draws nothing."
        }
        Field::ShowPointsAtInfinity => "Draw points at infinity, w = 0 (Points at ∞).",
        Field::ShowTargetIndicator => {
            "Draw the orbit target's indicator all the time (Target indicator). Off by default, \
             when it shows only while Alt is held and for a moment after the target moves."
        }
        Field::PointSizeLog2 => {
            "Point size as log2 of a multiplier on the automatic size: 0 is the automatic size, \
             1 twice it (the HUD's Points slider)."
        }
        Field::InfinityPointPx => "Size of a point at infinity on screen, in pixels.",
        Field::LengthScale => {
            "The scene scale, in scene units, that the grid, frustums and target are sized by \
             (the HUD's Scene slider)."
        }
        Field::PatchOpacity => "Opacity of the patch surfels.",
        Field::PatchSizeLog2 => "Patch size as log2 of a multiplier on the stored size.",
        Field::PatchAlphaCutoff => {
            "Patch texels with coverage below this are discarded (Edge cutoff)."
        }
        Field::MaintainZUp => {
            "Turn the view back to +Z up whenever it is not looking through a camera (Maintain \
             Z-up). Turning it on eases a rolled view back to level over the next frames."
        }
        Field::EdlLineThickness => "Width of the depth-edge shading, in pixels (EDL width).",
        Field::FrustumSizeMultiplier => "Frustum depth, as a multiple of the scene scale.",
        Field::TargetSizeMultiplier => "Target indicator radius, as a multiple of the scene scale.",
        Field::TargetFogMultiplier => {
            "How slowly the target indicator fades where scene geometry is in front of it; \
             larger fades it more slowly."
        }
        Field::ShowControlsHelp => "Paint the controls cheat sheet (Controls help).",
        Field::ShowFps => "Show the frame rate in the scene stats line (Frame rate).",
    }
}
