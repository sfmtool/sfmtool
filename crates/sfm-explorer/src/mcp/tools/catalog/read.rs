// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::Read;

pub(super) fn specs() -> Vec<ToolSpec> {
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
                            "maximum": crate::mcp::read::MAX_LIMIT,
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
            description: "One camera intrinsics record — the lens: camera_model, sensor size, every \
                          stored parameter by name, and the camera images that use it. The \
                          parameters are a name-to-value map in the model's own declaration \
                          order, which is the order `sfm inspect` prints. outermost_keypoint \
                          is the keypoint of those images furthest from the principal point, \
                          as a radius in pixels and an incidence angle under this model: \
                          observed among the reconstruction's observations, and detected \
                          among every feature of the images' .sift files (null when none is \
                          readable). It is the angle to set switch_camera_model's \
                          spline_domain_deg from.",
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
            name: "get_action_log",
            description: "What has happened in the viewer, oldest first — the human's selections \
                          and file loads, the agent's own calls, and every refusal, each with the \
                          revision it was written at. Pass the revision from a previous reply back \
                          as since_revision to read only what has happened since; the log's \
                          revision goes up on every entry and on every fold of a run of like \
                          entries into one, so a line that changed is reported again. \
                          oldest_revision says how far back the log still goes: a since_revision \
                          below it means entries were missed. actors filters by who did it, and \
                          actors: [\"user\"] is the read that answers \"what did the human do while \
                          I was working\". An entry carries took_ms once the viewer has drawn the \
                          frame that showed its result, and detail: true adds the stage-by-stage \
                          breakdown of where that time went.",
            kind: Read,
            schema: object(
                &[
                    (
                        "since_revision",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description":
                                "Return entries written or changed after this revision. Defaults \
                                 to 0, the start of the log.",
                        }),
                    ),
                    (
                        "limit",
                        json!({
                            "type": "integer",
                            "minimum": 1,
                            "maximum": crate::mcp::read::ACTION_LOG_MAX_LIMIT,
                            "description":
                                "How many entries to return. Defaults to 200, capped at 1000; \
                                 truncated in the reply says there are more, and the last entry's \
                                 revision is where to continue from.",
                        }),
                    ),
                    (
                        "actors",
                        json!({
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": crate::action_log::Actor::ALL.map(|a| a.wire_name()),
                            },
                            "minItems": 1,
                            "description":
                                "Which actors' entries to return. Omit for all of them; an empty \
                                 array is refused, since it can return nothing by construction.",
                        }),
                    ),
                    (
                        "detail",
                        flag(
                            "Carry each entry's breakdown: the stages it spent its time in and \
                             what it said, in the order the Action Log panel draws them, with \
                             elsewhere_ms beside took_ms for the time no stage claimed. Off by \
                             default, since a breakdown is several times the size of the row it \
                             hangs off. The read this is for: find a slow row, then ask again \
                             with detail set and since_revision just below that row's revision. \
                             It reports what was recorded, which is what set_timing_detail \
                             decided at the time.",
                        ),
                    ),
                ],
                &[],
            ),
        },
        ToolSpec {
            name: "get_timing_detail",
            description: "Whether the viewer is recording the finer stages inside each \
                          operation: the Action Log toolbar's Detailed timing checkbox, wherever \
                          it was last left. Off by default, and it lives only as long as the \
                          session.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_window_layout",
            description: "Where the window is and how its panels are arranged, as one document — \
                          the layout file the Panels menu saves and the viewer reads at startup, \
                          which set_window_layout takes back unchanged. Beside it: the live \
                          window block (state, focus, scale factor, current position and sizes in \
                          physical pixels, and every monitor, the current one first) and one \
                          entry per panel saying whether it is open and whether it is the front \
                          tab of its node. The document's window section is the *normal* \
                          rectangle — what the window restores to — so it and the live block \
                          differ for a maximized window, on purpose.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_image_detail_display",
            description: "What the Image Detail panel draws over its photograph: the feature \
                          overlay mode, the three filters on the features, and the intrinsics \
                          layer with its own sub-toggles, as one document. These are scene-level \
                          controls rather than properties of any image, so they decide what a \
                          screenshot of the image_detail panel shows whichever camera image is \
                          selected.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_image_detail_view",
            description: "Where the Image Detail panel is looking: the camera image on screen, \
                          the zoom (1.0 is the whole photograph fitted to the panel), the \
                          rectangle of the photograph the panel shows in its own pixels, the \
                          panel's size in points and the photograph's size in pixels. This is \
                          what the last drawn frame settled on, so it is what a screenshot of \
                          the panel would show — a select_camera_image sent a moment ago is not \
                          in it until the next frame. Every field is null before the panel has \
                          drawn a photograph at all.",
            kind: Read,
            schema: object(&[], &[]),
        },
        ToolSpec {
            name: "get_history",
            description: "One reconstruction's versions, oldest first: every edit anyone has \
                          made to it this session, with the sentence the edit recorded as each \
                          version's label and the time it was made. cursor names the version the \
                          viewer is showing, disk_serial the one its file holds (null when it \
                          came from no file), and dirty says the two differ. A version whose \
                          value the history budget released is listed with held: false: what \
                          happened there is still known, but the cursor can no longer go to it, \
                          and neither can jump_to_version.",
            kind: Read,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "get_bench",
            description: "The bench beside one reconstruction: every item on it by label, with \
                          its kind, the stage it is at, the point it came from where it came \
                          from one, its observation and verdict counts, and which item of each \
                          kind is active. The bench is a place beside the reconstruction rather \
                          than part of it — nothing on it is saved, and a commit is how it \
                          reaches the file.",
            kind: Read,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "get_bench_track",
            description: "One track on the bench, as its Track View table: the stage, the point \
                          it came from, the thresholds, and every observation with what put it \
                          there, the verdict on it, where it sits and whatever each stage has \
                          measured about it. An observation's pixel is where it sits whether or \
                          not anything has read it: the keypoint a reading wrote, else the \
                          refined cluster position, else the seed it was proposed at. So a \
                          candidate a search has just added says where it is without being \
                          evaluated first. An observation is addressed by its position in the \
                          list, which is \
                          stable for the life of the track — observations are appended and never \
                          renumbered, so an index read here still names the same observation \
                          after a verdict or an evaluation.",
            kind: Read,
            schema: object(
                &[("track", bench_track_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
    ]
}
