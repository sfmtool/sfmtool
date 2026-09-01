// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The tool table, and the parse from a `tools/call` argument object to a
//! [`Command`].
//!
//! Tool names and argument names *are* the API — they live in client configs
//! and in the prompts people write against them — so the vocabulary here obeys
//! one rule without exception: **one entity, one spelled-out word, in tool
//! names, arguments and reply fields alike.** No abbreviations, and no word
//! that names two things. See "The wire vocabulary" in
//! `specs/gui/mcp-server.md` for what that buys and what it costs.
//!
//! The catalog and the parser sit in one module because they are two halves of
//! one statement: [`catalog`] advertises what a tool accepts and [`parse`]
//! accepts it, and a test walks the pair so a schema and its parser cannot
//! drift.

use serde_json::{json, Map, Value};

use super::{
    CameraImageSel, CloseTarget, Command, DisplayChange, SelectionScope, ToolError, ViewCommand,
};
use crate::goto_point::{parse_point_query, PointQuery};

/// What a tool does to the viewer, which is all the MCP annotations need to
/// know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolKind {
    /// Changes nothing.
    Read,
    /// Changes the scene, the selection or the view. Never a file on disk.
    Write,
}

/// One advertised tool.
pub(crate) struct ToolSpec {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) kind: ToolKind,
    pub(crate) schema: Value,
}

/// Every tool this surface advertises, in the order `tools/list` reports them:
/// the reads first, then the writes, then the one that hands back a picture.
pub(crate) fn catalog() -> Vec<ToolSpec> {
    use ToolKind::{Read, Write};
    vec![
        ToolSpec {
            name: "get_scene",
            description: "The whole scene graph: every loaded reconstruction with its counts and \
                          display state, the current selection, which reconstruction is soloed, \
                          the 3D viewport camera, and the window title. Call this first — the \
                          labels it reports are the handles every other tool takes. Counts only: \
                          no tool here returns point arrays or track tables in bulk, so read the \
                          .sfmr file itself (or `sfm inspect`) for data and ask the viewer for \
                          state.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "list_camera_images",
            description: "One reconstruction's camera images — index, name, the intrinsics record \
                          each was shot through, camera centre, and observation count — a page at \
                          a time.",
            kind: Read,
            schema: object(
                &[
                    ("reconstruction_label", reconstruction_label_schema()),
                    (
                        "offset",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description": "First image index to return. Defaults to 0.",
                        }),
                    ),
                    (
                        "limit",
                        json!({
                            "type": "integer",
                            "minimum": 1,
                            "maximum": super::read::MAX_LIMIT,
                            "description":
                                "How many images to return. Defaults to 50, capped at 500.",
                        }),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "get_camera_image",
            description: "One camera image: its pose (world-to-camera quaternion and translation, \
                          plus the camera centre those imply), the intrinsics record it was shot \
                          through, how many observations it carries, and a reprojection-error \
                          summary. The error summary is null when it cannot be computed — an \
                          embedded-patches reconstruction has no .sift file to read features \
                          from.",
            kind: Read,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[("camera_image", camera_image_schema())],
            ),
        },
        ToolSpec {
            name: "get_camera_intrinsics",
            description: "One camera intrinsics record — the lens: model, sensor size, every \
                          stored parameter by name, and the camera images that use it. The \
                          parameters are a name-to-value map in the model's own declaration \
                          order, which is the order `sfm inspect` prints.",
            kind: Read,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[(
                    "camera_intrinsics_index",
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "description":
                            "Index of the intrinsics record. An intrinsics record has no name to \
                             address it by; get_camera_image reports the index each image uses.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "get_point",
            description: "One 3D point: position, colour, RMS error, whether it is a point at \
                          infinity, and its full track — every observing camera image with the \
                          pixel it was seen at and that observation's reprojection error.",
            kind: Read,
            schema: object(&[], &[("point", point_schema())]),
        },
        ToolSpec {
            name: "open_reconstruction",
            description: "Load an .sfmr file into the scene as a new reconstruction, and select \
                          it. Opening a path that is already open reloads that reconstruction in \
                          place instead, keeping its label, display state and transform; \
                          `reloaded` says which happened. Read the returned `label` back rather \
                          than assuming it — a colliding file stem is disambiguated as \
                          \"name (2)\".",
            kind: Write,
            schema: object(
                &[],
                &[(
                    "path",
                    json!({
                        "type": "string",
                        "description": "Path to an .sfmr file, as the viewer's process can see it.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "close_reconstruction",
            description: "Unload one reconstruction, or all of them. This removes it from the \
                          viewer; it does not delete or modify any file.",
            kind: Write,
            schema: json!({
                "type": "object",
                "description":
                    "Name one reconstruction to close, or pass all: true to clear the scene.",
                "properties": {
                    "reconstruction_label": {
                        "type": "string",
                        "description": "The reconstruction to close.",
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Close every loaded reconstruction.",
                    },
                },
                "additionalProperties": false,
            }),
        },
        ToolSpec {
            name: "select_reconstruction",
            description: "Make one reconstruction the one the file- and sequence-shaped panels \
                          follow. Selections belonging to other reconstructions are dropped, \
                          which is the invariant that stops two panels showing two different \
                          files' selections.",
            kind: Write,
            schema: object(
                &[],
                &[(
                    "reconstruction_label",
                    json!({ "type": "string", "description": "The reconstruction to select." }),
                )],
            ),
        },
        ToolSpec {
            name: "select_camera_image",
            description: "Select a camera image — and with it the intrinsics record it was shot \
                          through, and the reconstruction that owns it. The selected image's \
                          frustum is drawn in cyan, so this is visible in a screenshot.",
            kind: Write,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[("camera_image", camera_image_schema())],
            ),
        },
        ToolSpec {
            name: "select_camera_intrinsics",
            description: "Select a camera intrinsics record. Clears the selected camera image \
                          unless that image uses these intrinsics.",
            kind: Write,
            schema: object(
                &[("reconstruction_label", reconstruction_label_schema())],
                &[(
                    "camera_intrinsics_index",
                    json!({
                        "type": "integer",
                        "minimum": 0,
                        "description": "Index of the intrinsics record to select.",
                    }),
                )],
            ),
        },
        ToolSpec {
            name: "select_point",
            description: "Select a 3D point, and with it the reconstruction that owns it. A \
                          qualified pt3d_<hash>_<index> id names its own reconstruction, so this \
                          can move the selection to a different one. The selected point's track \
                          rays are drawn in orange.",
            kind: Write,
            schema: object(&[], &[("point", point_schema())]),
        },
        ToolSpec {
            name: "clear_selection",
            description: "Drop the selection, wholly or one kind of it. Selection is visible — a \
                          selected image tints its frustum, a selected point draws its track — so \
                          this is how to get a clean render before a screenshot. Clearing just \
                          the camera image keeps its intrinsics selected: dismissing a photograph \
                          says nothing about the lens.",
            kind: Write,
            schema: object(
                &[(
                    "scope",
                    json!({
                        "type": "string",
                        "enum": ["all", "camera_image", "camera_intrinsics", "point"],
                        "description": "How much to clear. Defaults to all.",
                    }),
                )],
                &[],
            ),
        },
        ToolSpec {
            name: "set_reconstruction_display",
            description: "Change how one reconstruction is drawn: its master eye, whether pointer \
                          picks reach it, the per-group eyes, and its comparison tint. Every \
                          field is optional and every omitted one is left alone. None of this \
                          touches the reconstruction's data.",
            kind: Write,
            schema: object(
                &[
                    ("visible", flag("Master eye. Off draws nothing of it.")),
                    (
                        "interactive",
                        flag(
                            "Whether pointer hover and click-pick in the 3D viewport reach this \
                             reconstruction. Off is display-only: it still renders and occludes.",
                        ),
                    ),
                    ("show_points", flag("Group eye: the 3D points.")),
                    (
                        "show_camera_images",
                        flag("Group eye: the camera frustums and image quads."),
                    ),
                    ("show_patches", flag("Group eye: the patch surfels.")),
                    (
                        "show_points_at_infinity",
                        flag("Sub-toggle of show_points: the w = 0 directions."),
                    ),
                    (
                        "tint",
                        json!({
                            "type": ["string", "null"],
                            "enum": tint_names_with_null(),
                            "description":
                                "A comparison tint mixed into everything this reconstruction \
                                 draws, from a fixed colour-blind-safe palette, or null for its \
                                 own colours.",
                        }),
                    ),
                ],
                &[(
                    "reconstruction_label",
                    json!({ "type": "string", "description": "The reconstruction to restyle." }),
                )],
            ),
        },
        ToolSpec {
            name: "set_solo",
            description: "Draw only one reconstruction, or end the solo. At most one is soloed at \
                          a time and soloing a second moves the solo. Solo is independent of \
                          selection and never writes the per-reconstruction eyes, so ending it \
                          restores exactly the visibility that was set by hand.",
            kind: Write,
            schema: object(
                &[(
                    "reconstruction_label",
                    json!({
                        "type": ["string", "null"],
                        "description":
                            "The reconstruction to draw alone, or null to end the solo.",
                    }),
                )],
                &[],
            ),
        },
        ToolSpec {
            name: "set_view",
            description: "Move the 3D viewport camera — the tool to call immediately before \
                          screenshot. Five forms, exactly one per call: frame everything or one \
                          reconstruction (fit), look through a camera image (look_through), leave \
                          camera view (exit_camera_view), place the camera by position and target \
                          (the look-at form), or restore a view read from get_scene (the exact \
                          form). fov_short_axis_deg may ride along with the last two or be sent \
                          alone. View changes jump rather than animating, so a screenshot taken \
                          straight afterward shows the new view.",
            kind: Write,
            schema: set_view_schema(),
        },
        ToolSpec {
            name: "screenshot",
            description: "A PNG of the 3D viewport as it is drawn right now — the 3D view itself, \
                          not the surrounding panels. Answered after the next frame has been \
                          rendered, so it reflects any change made in the same batch of calls.",
            kind: Read,
            schema: object(
                &[(
                    "max_dimension",
                    json!({
                        "type": "integer",
                        "minimum": 16,
                        "description":
                            "Scale the image down so neither side exceeds this many pixels. \
                             Omit for the viewport's native size.",
                    }),
                )],
                &[],
            ),
        },
    ]
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
             label is unique across the scene and survives a reload, which is why it rather than \
             any internal id is the handle.",
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
            "position": vec3_schema("Camera position in world coordinates."),
            "target": vec3_schema(
                "The point the camera looks at. Present makes this the look-at form.",
            ),
            "up": vec3_schema(
                "The roll, for the look-at form. Defaults to the view's current world_up.",
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
                "description": "Distance to the orbit target along the camera's forward axis.",
            },
            "world_up": vec3_schema("Navigation up, which carries the roll, for the exact form."),
            "fov_short_axis_deg": {
                "type": "number",
                "minimum": 5,
                "maximum": 160,
                "description":
                    "Field of view of the shorter viewport dimension — vertical in a landscape \
                     window, horizontal in a portrait one. May accompany either explicit form, \
                     or be sent alone.",
            },
        },
        "additionalProperties": false,
    })
}

// ── The parse ────────────────────────────────────────────────────────────

/// Build the [`Command`] a `tools/call` asked for.
///
/// The schemas above are closed and typed, so a compliant client will not
/// reach most of these errors; they are here because a tool call arrives from
/// whatever the agent actually sent, and "silently did something else" is the
/// one answer this surface must never give.
pub(crate) fn parse(
    name: &str,
    arguments: Option<&Map<String, Value>>,
) -> Result<Command, ToolError> {
    static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
    let map = arguments.unwrap_or_else(|| EMPTY.get_or_init(Map::new));
    let args = Args { tool: name, map };

    let command = match name {
        "get_scene" => {
            args.reject_unknown(&[])?;
            Command::GetScene
        }
        "list_camera_images" => {
            args.reject_unknown(&["reconstruction_label", "offset", "limit"])?;
            Command::ListCameraImages {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                offset: args.optional_usize("offset")?.unwrap_or(0),
                limit: args
                    .optional_usize("limit")?
                    .unwrap_or(super::read::DEFAULT_LIMIT),
            }
        }
        "get_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image"])?;
            Command::GetCameraImage {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "get_camera_intrinsics" => {
            args.reject_unknown(&["reconstruction_label", "camera_intrinsics_index"])?;
            Command::GetCameraIntrinsics {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
            }
        }
        "get_point" => {
            args.reject_unknown(&["point"])?;
            Command::GetPoint {
                point: args.point("point")?,
            }
        }
        "open_reconstruction" => {
            args.reject_unknown(&["path"])?;
            Command::OpenReconstruction {
                path: std::path::PathBuf::from(args.required_string("path")?),
            }
        }
        "close_reconstruction" => {
            args.reject_unknown(&["reconstruction_label", "all"])?;
            let all = args.optional_bool("all")?.unwrap_or(false);
            let label = args.optional_string("reconstruction_label")?;
            match (all, label) {
                (true, None) => Command::CloseReconstruction {
                    target: CloseTarget::All,
                },
                (false, Some(label)) => Command::CloseReconstruction {
                    target: CloseTarget::One(label),
                },
                (true, Some(_)) => {
                    return Err(args.error(
                        "takes either reconstruction_label or all: true, not both — \"close this \
                         one\" and \"close everything\" are different requests.",
                    ))
                }
                (false, None) => {
                    return Err(args
                        .error("needs a reconstruction_label, or all: true to clear the scene."))
                }
            }
        }
        "select_reconstruction" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::SelectReconstruction {
                reconstruction_label: args.required_string("reconstruction_label")?,
            }
        }
        "select_camera_image" => {
            args.reject_unknown(&["reconstruction_label", "camera_image"])?;
            Command::SelectCameraImage {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_image: args.camera_image("camera_image")?,
            }
        }
        "select_camera_intrinsics" => {
            args.reject_unknown(&["reconstruction_label", "camera_intrinsics_index"])?;
            Command::SelectCameraIntrinsics {
                reconstruction_label: args.optional_string("reconstruction_label")?,
                camera_intrinsics_index: args.required_usize("camera_intrinsics_index")?,
            }
        }
        "select_point" => {
            args.reject_unknown(&["point"])?;
            Command::SelectPoint {
                point: args.point("point")?,
            }
        }
        "clear_selection" => {
            args.reject_unknown(&["scope"])?;
            let scope = match args.optional_string("scope")?.as_deref() {
                None | Some("all") => SelectionScope::All,
                Some("camera_image") => SelectionScope::CameraImage,
                Some("camera_intrinsics") => SelectionScope::CameraIntrinsics,
                Some("point") => SelectionScope::Point,
                Some(other) => {
                    return Err(args.error(format!(
                        "does not know the scope {other:?} — expected all, camera_image, \
                         camera_intrinsics or point."
                    )))
                }
            };
            Command::ClearSelection { scope }
        }
        "set_reconstruction_display" => {
            args.reject_unknown(&[
                "reconstruction_label",
                "visible",
                "interactive",
                "show_points",
                "show_camera_images",
                "show_patches",
                "show_points_at_infinity",
                "tint",
            ])?;
            let change = DisplayChange {
                visible: args.optional_bool("visible")?,
                interactive: args.optional_bool("interactive")?,
                show_points: args.optional_bool("show_points")?,
                show_camera_images: args.optional_bool("show_camera_images")?,
                show_patches: args.optional_bool("show_patches")?,
                show_points_at_infinity: args.optional_bool("show_points_at_infinity")?,
                // Doubly optional: absent leaves the tint alone, an explicit
                // null clears it.
                tint: match args.map.get("tint") {
                    None => None,
                    Some(Value::Null) => Some(None),
                    Some(Value::String(name)) => Some(Some(name.clone())),
                    Some(_) => return Err(args.error("wants tint to be a palette name or null.")),
                },
            };
            if change == DisplayChange::default() {
                return Err(args.error("was given nothing to change."));
            }
            Command::SetReconstructionDisplay {
                reconstruction_label: args.required_string("reconstruction_label")?,
                change,
            }
        }
        "set_solo" => {
            args.reject_unknown(&["reconstruction_label"])?;
            Command::SetSolo {
                reconstruction_label: args.optional_string("reconstruction_label")?,
            }
        }
        "set_view" => parse_set_view(&args)?,
        "screenshot" => {
            args.reject_unknown(&["max_dimension"])?;
            Command::Screenshot {
                max_dimension: args
                    .optional_usize("max_dimension")?
                    .map(|d| d.min(u32::MAX as usize) as u32),
            }
        }
        other => {
            return Err(ToolError::new(format!(
                "There is no tool named {other:?}. Call tools/list for what this viewer offers."
            )))
        }
    };
    Ok(command)
}

/// `set_view`'s five forms, told apart by which field is present.
///
/// The forms are exclusive and the check is up front, because they are
/// *intents* rather than representations: a call carrying both `fit` and
/// `position` has no answer, and guessing one would move the camera somewhere
/// the agent did not ask for.
fn parse_set_view(args: &Args) -> Result<Command, ToolError> {
    args.reject_unknown(&[
        "fit",
        "look_through",
        "exit_camera_view",
        "position",
        "target",
        "up",
        "orientation_wxyz",
        "target_distance",
        "world_up",
        "fov_short_axis_deg",
    ])?;

    let present = |key: &str| args.map.contains_key(key);
    let forms: Vec<&str> = ["fit", "look_through", "exit_camera_view", "position"]
        .into_iter()
        .filter(|key| present(key))
        .collect();
    if forms.len() > 1 {
        return Err(args.error(format!(
            "was given {} at once — fit, look_through, exit_camera_view and the two explicit \
             camera forms are exclusive, one per call.",
            forms.join(" and ")
        )));
    }

    let fov = args.optional_f64("fov_short_axis_deg")?;

    if present("fit") {
        return Ok(Command::SetView {
            view: ViewCommand::Fit {
                reconstruction_label: args.optional_string("fit")?,
            },
        });
    }
    if let Some(look_through) = args.map.get("look_through") {
        let map = look_through.as_object().ok_or_else(|| {
            args.error("wants look_through to be an object naming a camera image.")
        })?;
        let inner = Args {
            tool: "set_view.look_through",
            map,
        };
        inner.reject_unknown(&["reconstruction_label", "camera_image"])?;
        return Ok(Command::SetView {
            view: ViewCommand::LookThrough {
                reconstruction_label: inner.optional_string("reconstruction_label")?,
                camera_image: inner.camera_image("camera_image")?,
            },
        });
    }
    if present("exit_camera_view") {
        if args.optional_bool("exit_camera_view")? != Some(true) {
            return Err(args.error(
                "reads exit_camera_view: false as no request at all — omit it, or pass true.",
            ));
        }
        return Ok(Command::SetView {
            view: ViewCommand::ExitCameraView,
        });
    }
    if present("position") {
        let position = args.required_vec3("position")?;
        let exact = present("orientation_wxyz");
        if exact && present("target") {
            return Err(args.error(
                "was given both target and orientation_wxyz — the look-at form and the exact \
                 form are exclusive.",
            ));
        }
        if exact {
            return Ok(Command::SetView {
                view: ViewCommand::Exact {
                    position,
                    orientation_wxyz: args.required_vec4("orientation_wxyz")?,
                    target_distance: args.required_f64("target_distance")?,
                    world_up: args.optional_vec3("world_up")?,
                    fov_short_axis_deg: fov,
                },
            });
        }
        return Ok(Command::SetView {
            view: ViewCommand::LookAt {
                position,
                target: args.required_vec3("target")?,
                up: args.optional_vec3("up")?,
                fov_short_axis_deg: fov,
            },
        });
    }
    match fov {
        Some(fov_short_axis_deg) => Ok(Command::SetView {
            view: ViewCommand::Fov { fov_short_axis_deg },
        }),
        None => Err(args.error(
            "was given nothing to do — pass fit, look_through, exit_camera_view, a position with \
             a target or an orientation_wxyz, or fov_short_axis_deg alone.",
        )),
    }
}

/// One tool call's argument object, with the accessors that turn a JSON value
/// into a typed argument or into a message saying what was wrong with it.
struct Args<'a> {
    tool: &'a str,
    map: &'a Map<String, Value>,
}

impl Args<'_> {
    /// `"<tool> <complaint>"`, so every message from this module reads as a
    /// sentence about the tool that was called.
    fn error(&self, complaint: impl std::fmt::Display) -> ToolError {
        ToolError::new(format!("{} {complaint}", self.tool))
    }

    fn wrong_type(&self, key: &str, expected: &str, got: &Value) -> ToolError {
        self.error(format!(
            "wants {key} to be {expected} — got {}.",
            describe(got)
        ))
    }

    /// Refuse an argument the tool does not have.
    ///
    /// The schemas say `additionalProperties: false`, but a schema is only
    /// enforced by clients that enforce it. An ignored typo would leave the
    /// agent believing it asked for something it did not, and the whole reason
    /// this surface returns its resulting state is so that never happens.
    fn reject_unknown(&self, allowed: &[&str]) -> Result<(), ToolError> {
        let unknown: Vec<String> = self
            .map
            .keys()
            .filter(|key| !allowed.contains(&key.as_str()))
            .map(|key| format!("{key:?}"))
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        let known = if allowed.is_empty() {
            "it takes none".to_string()
        } else {
            format!("it takes {}", allowed.join(", "))
        };
        Err(self.error(format!("has no argument {} — {known}.", unknown.join(", "))))
    }

    fn optional_string(&self, key: &str) -> Result<Option<String>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(other) => Err(self.wrong_type(key, "a string", other)),
        }
    }

    fn required_string(&self, key: &str) -> Result<String, ToolError> {
        self.optional_string(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_bool(&self, key: &str) -> Result<Option<bool>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Bool(b)) => Ok(Some(*b)),
            Some(other) => Err(self.wrong_type(key, "true or false", other)),
        }
    }

    fn optional_usize(&self, key: &str) -> Result<Option<usize>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_u64()
                .map(|n| Some(n as usize))
                .ok_or_else(|| self.wrong_type(key, "a whole number, zero or more", value)),
        }
    }

    fn required_usize(&self, key: &str) -> Result<usize, ToolError> {
        self.optional_usize(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_f64(&self, key: &str) -> Result<Option<f64>, ToolError> {
        match self.map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .map(Some)
                .ok_or_else(|| self.wrong_type(key, "a number", value)),
        }
    }

    fn required_f64(&self, key: &str) -> Result<f64, ToolError> {
        self.optional_f64(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_vec3(&self, key: &str) -> Result<Option<[f64; 3]>, ToolError> {
        self.optional_numbers::<3>(key)
    }

    fn required_vec3(&self, key: &str) -> Result<[f64; 3], ToolError> {
        self.optional_vec3(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn required_vec4(&self, key: &str) -> Result<[f64; 4], ToolError> {
        self.optional_numbers::<4>(key)?
            .ok_or_else(|| self.error(format!("needs {key}.")))
    }

    fn optional_numbers<const N: usize>(&self, key: &str) -> Result<Option<[f64; N]>, ToolError> {
        let value = match self.map.get(key) {
            None | Some(Value::Null) => return Ok(None),
            Some(value) => value,
        };
        let expected = format!("an array of {N} numbers");
        let array = value
            .as_array()
            .ok_or_else(|| self.wrong_type(key, &expected, value))?;
        if array.len() != N {
            return Err(self.error(format!(
                "wants {key} to be {expected} — got {}.",
                array.len()
            )));
        }
        let mut out = [0.0; N];
        for (slot, element) in out.iter_mut().zip(array) {
            *slot = element
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| self.wrong_type(key, &expected, value))?;
        }
        Ok(Some(out))
    }

    /// A camera image argument, in either of its two spellings.
    fn camera_image(&self, key: &str) -> Result<CameraImageSel, ToolError> {
        match self.map.get(key) {
            Some(Value::String(name)) => Ok(CameraImageSel::Name(name.clone())),
            Some(value) if value.as_u64().is_some() => Ok(CameraImageSel::Index(
                value.as_u64().expect("just checked") as usize,
            )),
            Some(value) => Err(self.wrong_type(key, "an image index or an image name", value)),
            None => Err(self.error(format!(
                "needs {key} — an index, or the image's .sfmr relative path."
            ))),
        }
    }

    /// A point argument, through the same parser the Go to Point dialog uses.
    ///
    /// A bare JSON integer is the index form spelled as a number rather than as
    /// a string, which is what a caller reading an index out of a track will
    /// naturally send; everything else goes to
    /// [`parse_point_query`], whose error messages already show both accepted
    /// shapes.
    fn point(&self, key: &str) -> Result<PointQuery, ToolError> {
        match self.map.get(key) {
            Some(value) if value.as_u64().is_some() => Ok(PointQuery::Index(
                value.as_u64().expect("just checked") as usize,
            )),
            Some(Value::String(text)) => parse_point_query(text).map_err(ToolError),
            Some(value) => Err(self.wrong_type(key, "a point index or a point id", value)),
            None => Err(self.error(format!(
                "needs {key} — an index, or a pt3d_<hash>_<index> id."
            ))),
        }
    }
}

/// What a value is, for a message that has to say what arrived instead.
///
/// The kind and not the value: an argument that was wrong is usually long, and
/// a message that quotes the whole of it buries the part that matters.
fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "true or false",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
