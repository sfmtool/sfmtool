// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Catalog families in wire order; schema fragments shared by those families.

use super::{ToolKind, ToolSpec};
use crate::dock::Tab;
use crate::window::WindowState;
use serde_json::{json, Map, Value};

mod background;
mod bench;
mod edit;
mod read;
mod viewer;

pub(super) fn build_catalog() -> Vec<ToolSpec> {
    let mut specs = read::specs();
    specs.extend(viewer::specs());
    specs.extend(edit::specs());
    specs.extend(bench::specs());
    specs.extend(background::specs());
    specs
}

// ── Schema fragments ─────────────────────────────────────────────────────

/// An object schema from its optional and required properties.
fn object(optional: &[(&str, Value)], required: &[(&str, Value)]) -> Value {
    let mut properties = Map::new();
    for (name, schema) in optional.iter().chain(required) {
        properties.insert((*name).to_string(), schema.clone());
    }
    json!({
        "type": "object",
        "properties": properties,
        "required": required.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        // Closed on purpose. A misspelled argument that is silently ignored
        // leaves the agent believing it asked for something it did not.
        "additionalProperties": false,
    })
}

fn flag(description: &str) -> Value {
    json!({ "type": "boolean", "description": description })
}

fn reconstruction_label_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which reconstruction, by the label get_scene reports. Omit for the selected one. A \
             label is unique across the scene and survives every edit, which is why it rather \
             than any internal id is the handle.",
    })
}

/// The reconstruction argument of a tool that edits one, reads its history or
/// writes it out.
///
/// Required rather than defaulting to the selection, which is the one place
/// this surface departs from "omit for the selected one": the selection is the
/// *human's*, it moves under the agent between calls, and an edit that landed
/// on whatever was last clicked would be an edit the agent could not check it
/// had asked for. A read of the history is required for the same reason its
/// answer would otherwise be about a node the caller did not name.
fn edited_label_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which reconstruction, by the label get_scene reports. Named rather than defaulting \
             to the selected one: the selection belongs to the human at the window and can move \
             between calls.",
    })
}

/// A pixel in one camera image's own pixel coordinates.
fn pixel_schema() -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 2,
        "maxItems": 2,
        "description":
            "A pixel [x, y] in the camera image's own coordinates, as get_point reports an \
             observation's xy.",
    })
}

/// A camera image argument, which takes either of the two handles the surface
/// hands out.
///
/// The field is named for the entity rather than for an attribute, because it
/// has no single spelling: a track observation reports an index, a
/// `list_camera_images` row reports both, and an agent arrives holding
/// whichever it read. Contrast `camera_intrinsics_index`, which can name its
/// attribute because an intrinsics record has exactly one handle.
fn camera_image_schema() -> Value {
    json!({
        "description":
            "Which camera image: its index in the reconstruction, or its name — the .sfmr \
             relative path, as in \"images/IMG_0042.jpg\".",
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
        ],
    })
}

/// The photograph a bench patch tool's pixel is in, named as a camera image in
/// place of an observation: the ghost outline's square rather than a
/// sighting's.
fn pixel_view_schema() -> Value {
    json!({
        "description":
            "The camera image the pixel is in, by index or by .sfmr relative path, in place \
             of observation: the pixel is read against the patch as it stands (the ghost \
             outline Image Detail draws where the track has no sighting), not re-anchored on \
             a keypoint. Give this or observation, not both.",
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
        ],
    })
}

/// An item on the bench, by its label.
///
/// A label and not an index: an item is named by its label everywhere -- in the
/// Scene tree, on the panel's tabs and in every Action Log row -- and the
/// bench's own refusal for one that names nothing is in those terms.
fn bench_item_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which item on the bench, by the label get_bench reports. A label is minted from \
             what the item was made from — a point id, or an image and a pixel — until \
             rename_bench_item gives it one of your own.",
    })
}

/// The track a bench tool acts on, which is optional: a call that names none
/// acts on the active track, as a gesture in Track View's edit mode does.
fn bench_track_schema() -> Value {
    json!({
        "type": "string",
        "description":
            "Which track on the bench, by its label. Omit for the active track, which is the \
             item Track View is editing and what a create or an activate last made active; with \
             nothing active, a call that omits it is refused.",
    })
}

/// One sighting's affine shape: the detector's canonical keypoint frame mapped
/// onto that image's pixels.
fn affine_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "array",
            "items": { "type": "number" },
            "minItems": 2,
            "maxItems": 2,
        },
        "minItems": 2,
        "maxItems": 2,
        "description":
            "The affine shape, as [[a11, a12], [a21, a22]]: the detector's canonical keypoint \
             frame mapped onto this image's pixels, which is what a .sift feature's shape is. A \
             shape is a scale rather than a size, so it is read over the cluster's own radius.",
    })
}

/// How far around each observation a reading looks for its correlation peak.
fn search_px_schema() -> Value {
    json!({
        "type": "number",
        "description":
            "How far from each observation's own pixel the correlation peak is looked for, in \
             patch-grid px. Omit for the reading's own radius. A wider window finds a feature \
             the sighting sits further from, and says so in seed_shift_px.",
    })
}

/// One observation of a bench track, by its position in the track's list.
fn observation_schema() -> Value {
    json!({
        "type": "integer",
        "minimum": 0,
        "description":
            "Which observation, by its position in get_bench_track's observations list. \
             Observations are appended and never renumbered, so the position is stable for the \
             life of the track.",
    })
}

/// Which edge of a patch's square a resize drags.
///
/// Named rather than numbered, and by side as well as axis, because a resize
/// holds the **opposite** edge still: which of the two edges of an axis moves
/// is the whole of the difference between the patch growing one way and the
/// other.
fn edge_schema() -> Value {
    json!({
        "type": "string",
        "enum": ["+u", "-u", "+v", "-v"],
        "description":
            "Which edge of the patch's square to drag: +u and -u are the two edges across the \
             patch's first axis, +v and -v the two across its second. The opposite edge stays \
             where it is, so the patch grows or shrinks toward the one you name.",
    })
}

/// One bar of the threshold painting: a number, or absent to leave the bar
/// where the track has it.
fn threshold_schema(description: &str) -> Value {
    json!({ "type": "number", "description": description })
}

/// The seed forms `create_bench_cluster` and `add_bench_track_observation`
/// share, as optional properties.
///
/// Optional rather than required because the three forms are alternatives and a
/// schema cannot say "one of these"; which combinations are a seed at all is
/// settled in [`super::parse_seed`], which sees what the call named.
fn seed_properties() -> Vec<(&'static str, Value)> {
    vec![
        ("pixel", pixel_schema()),
        (
            "radius_px",
            json!({
                "type": "number",
                "exclusiveMinimum": 0,
                "description":
                    "The patch's half-width at that pixel, in this image's own pixels. With \
                     create_bench_cluster, omit for the median radius the image's existing \
                     patches project to; with add_bench_track_observation, omit for the scale \
                     the track already works at.",
            }),
        ),
        (
            "affine",
            json!({
                "type": "array",
                "items": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 2,
                "maxItems": 2,
                "description":
                    "The affine shape at that pixel, as [[a11, a12], [a21, a22]]: the detector's \
                     canonical keypoint frame mapped onto this image's pixels, which is what a \
                     .sift feature's shape is. A shape is a scale rather than a size, so it is \
                     read over the cluster's own radius.",
            }),
        ),
        (
            "feature",
            json!({
                "type": "integer",
                "minimum": 0,
                "description":
                    "Seed from a .sift feature of this camera image instead, by its index in \
                     that file. It carries its own position and its own shape, so pixel, \
                     radius_px and affine are not accepted with it.",
            }),
        ),
    ]
}

fn point_schema() -> Value {
    json!({
        "description":
            "Which 3D point: a bare index into the selected reconstruction, or a full \
             pt3d_<hash>_<index> id, which names its own reconstruction and so resolves \
             wherever the selection happens to be.",
        "anyOf": [
            { "type": "integer", "minimum": 0 },
            { "type": "string" },
        ],
    })
}

/// A panel argument: one of the seven names, spelled as the layout file spells
/// them.
///
/// `panel_name` rather than `panel`, because the field carries a name and not
/// a panel — the same rule that makes the reconstruction argument
/// `reconstruction_label`.
fn panel_name_schema() -> Value {
    json!({
        "type": "string",
        "enum": Tab::ALL.map(|tab| tab.wire_name()),
        "description":
            "Which panel, by the name get_window_layout and the layout file use.",
    })
}

fn window_state_schema() -> Value {
    json!({
        "type": "string",
        "enum": WindowState::ALL.map(|state| state.wire_name()),
        "description":
            "What the window should be. \"normal\" restores it from all three of minimized, \
             maximized and fullscreen.",
    })
}

/// The `window` section of a layout document: the same five keys the file
/// carries, since the file and the wire are one document with one parser.
fn window_section_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "description":
            "Where the window goes. Every key is optional and what the section does not carry is \
             preserved; omit the section to leave the window alone.",
        "properties": {
            "state": window_state_schema(),
            "outer_position": int_pair_schema(
                "The top-left corner of the window's normal rectangle, in physical pixels in \
                 desktop coordinates.",
                None,
            ),
            "inner_size": int_pair_schema(
                "The drawable area of the window's normal rectangle, in physical pixels. The \
                 platform has the last word on it, so read the reply rather than assuming the \
                 request.",
                Some(1),
            ),
            "monitor": {
                "type": "object",
                "additionalProperties": false,
                "description":
                    "The monitor the rectangle was measured on, as get_window_layout writes it. \
                     A rectangle that lands nowhere visible on this desktop is mapped onto the \
                     current monitor by the share of it the window occupied. Needs a rectangle \
                     to fit.",
                "properties": {
                    "position": int_pair_schema(
                        "The monitor's top-left corner, physical pixels.",
                        None,
                    ),
                    "size": int_pair_schema("The monitor's size, physical pixels.", Some(1)),
                },
                "required": ["position", "size"],
            },
            "focus": {
                "type": "boolean",
                "description":
                    "Bring the window to the front. A platform may decline to let an application \
                     take focus; the reply's focused says whether it worked.",
            },
        },
    })
}

/// A two-element array of whole numbers: a size, or a position.
fn int_pair_schema(description: &str, minimum: Option<i64>) -> Value {
    let mut items = json!({ "type": "integer" });
    if let Some(minimum) = minimum {
        items["minimum"] = json!(minimum);
    }
    json!({
        "type": "array",
        "items": items,
        "minItems": 2,
        "maxItems": 2,
        "description": description,
    })
}

fn tint_names_with_null() -> Vec<Value> {
    crate::scene::TINT_PALETTE
        .iter()
        .map(|color| json!(color.name))
        .chain(std::iter::once(Value::Null))
        .collect()
}

fn vec3_schema(description: &str) -> Value {
    json!({
        "type": "array",
        "items": { "type": "number" },
        "minItems": 3,
        "maxItems": 3,
        "description": description,
    })
}

fn set_view_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "fit": {
                "type": ["string", "null"],
                "description":
                    "Frame the scene: a reconstruction label to frame that one, or null to frame \
                     everything drawn.",
            },
            "look_through": {
                "type": "object",
                "description": "Look through one camera image, as double-clicking its frustum does.",
                "properties": {
                    "reconstruction_label": reconstruction_label_schema(),
                    "camera_image": camera_image_schema(),
                },
                "required": ["camera_image"],
                "additionalProperties": false,
            },
            "exit_camera_view": {
                "type": "boolean",
                "description": "Leave camera view, keeping the camera where it is.",
            },
            "position": vec3_schema(
                "Camera position in world coordinates. On its own it moves the camera and \
                 carries the target along, keeping the orientation.",
            ),
            "target": vec3_schema(
                "The point the camera looks at. With position, the look-at form; on its own it \
                 re-centres the view on that point, keeping the orientation and the distance.",
            ),
            "forward": vec3_schema(
                "The direction the camera looks, the derived.forward the view block reports. \
                 Need not be a unit vector. On its own it swings the camera around what it is \
                 looking at rather than turning it in place.",
            ),
            "up": vec3_schema(
                "The roll, where the orientation is being derived -- with forward, or with \
                 position and target. Defaults to the view's current world_up.",
            ),
            "orientation_wxyz": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 4,
                "maxItems": 4,
                "description":
                    "World-to-camera rotation. Present makes this the exact form, which restores \
                     a view read from get_scene verbatim.",
            },
            "target_distance": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description":
                    "Distance to the orbit target along the camera's forward axis. On its own it \
                     dollies: the target stays put and the camera moves. Not accepted alongside \
                     position and target, whose separation is already the distance.",
            },
            "world_up": vec3_schema(
                "Navigation up, which carries the roll, for the exact form. Elsewhere the roll \
                 is up.",
            ),
            "fov_short_axis_deg": {
                "type": "number",
                "minimum": 5,
                "maximum": 160,
                "description":
                    "Field of view of the shorter viewport dimension — vertical in a landscape \
                     window, horizontal in a portrait one. May accompany an explicit camera \
                     placement, or be sent alone.",
            },
        },
        "additionalProperties": false,
    })
}
