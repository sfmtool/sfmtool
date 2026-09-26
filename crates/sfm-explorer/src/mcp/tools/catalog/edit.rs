// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::{Save, Write};

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "undo",
            description: "Step one reconstruction's history back a version, exactly as the Edit \
                          menu's Undo does. Refused when it is already on its first version. The \
                          reply is the version now showing.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "redo",
            description: "Step one reconstruction's history forward a version. Refused when \
                          there is nothing ahead of the cursor, since an edit made after an undo \
                          discards what was ahead. The reply is the version now showing.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "jump_to_version",
            description: "Move one reconstruction's cursor straight to a version, which is what \
                          clicking a row of the Edit History panel does. The move is the run of \
                          undos or redos between the two, so the selection lands where stepping \
                          would have put it, and it is refused whole when any version on the way, \
                          the destination included, has had its value released.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "serial",
                        json!({
                            "type": "string",
                            "description":
                                "Which version, spelled as get_history and the Action Log spell \
                                 it: \"v12\".",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "save_reconstruction",
            description: "Write one reconstruction to disk. With no path it is written over the \
                          file it came from, and a reconstruction that came from no file is \
                          refused; with a path it is written there and the node is re-pointed at \
                          it, taking that file's name as its label, so read the reply's \
                          reconstruction_label back before the next call. The version written is \
                          the one at the cursor, and it becomes the version the history calls \
                          clean. With minimal: true and a path, it instead writes a minimal copy \
                          there, the file sfm xform --minimal writes: no thumbnails, no patch \
                          bitmaps, no lineage and no absolute workspace path. The node keeps its \
                          path, label and history, and is no cleaner than before; a minimal copy \
                          over the node's own file is refused. workspace_path states the \
                          workspace.relative_path the file records instead of the measured one.",
            kind: Save,
            schema: object(
                &[
                    (
                        "path",
                        json!({
                            "type": "string",
                            "description":
                                "Where to write it, as the viewer's process can see it. Omit \
                                 to write over the file the reconstruction came from; \
                                 required with minimal.",
                        }),
                    ),
                    (
                        "minimal",
                        json!({
                            "type": "boolean",
                            "description":
                                "Write a minimal copy to path instead of saving the node. \
                                 Default false.",
                        }),
                    ),
                    (
                        "workspace_path",
                        json!({
                            "type": "string",
                            "description":
                                "The workspace.relative_path to record in the file, as the \
                                 reader will walk it from the file's own directory, instead \
                                 of the path measured from where the file is written. \
                                 \".\" for a file written inside its workspace. Requires \
                                 path: a save of the node's own file leaves it where it is, \
                                 so the measured path is already right, and an override \
                                 only means something for a copy written elsewhere.",
                        }),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "delete_point",
            description: "Delete one 3D point and its whole track. A point edit: every other \
                          point keeps the index it had, so indexes an agent is holding stay \
                          good, and undo puts it back.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "retriangulate_point",
            description: "Re-solve one 3D point from its own observations, at the poses and the \
                          lens the reconstruction already holds. Moves no camera and no other \
                          point. A point edit, so every other index stays good; the point itself \
                          is deleted and re-added, so it takes a new index, which the reply's \
                          report names along with the verdict its observations supported - \
                          finite, at infinity, behind a camera that sees it, too thin to place. \
                          A point the reconstruction holds at a fixed coordinate is refused, and \
                          a point fewer than two of whose observations state a usable ray keeps \
                          the geometry it had. Undo puts it back.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("point", point_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "retriangulate_all_points",
            description: "Re-solve every 3D point of one reconstruction from its own \
                          observations, at the poses and the lens it already holds. Moves no \
                          camera and no lens: this is the structure re-read at a geometry \
                          somebody else decided. A bulk edit giving the node a whole new base, \
                          but it deletes no point and creates none, so every index still means \
                          what it meant. Points the reconstruction holds at a fixed coordinate \
                          are left alone, a ranged point keeps its distance and only its \
                          direction is re-read, and a point too thinly seen to place keeps the \
                          geometry it had. One version, and it runs on a worker thread, so a \
                          retriangulation still going after 200 ms replies with running: true \
                          and an operation_id instead of the version; cancel_background_task \
                          stops it. Needs a pixel per observation and one shared camera.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "prune_covered_observations",
            description: "Retire every observation of one reconstruction that a finer tracked \
                          observation covers in the same image, and drop the points left with \
                          fewer than two. Each observation's footprint is its point's patch frame \
                          projected into the image that saw it; an observation goes when another \
                          one, on another point, sits inside that footprint with a radius at \
                          least ratio times smaller. The coarse side is the one retired, never \
                          the fine one. Nothing is re-solved and nothing moves: a surviving point \
                          keeps its position, frame, bitmap, colour and constraint, and only its \
                          observation list is shorter. Points the reconstruction holds or ranges \
                          are never retired, though they still cover. A bulk edit giving the node \
                          a whole new base, so the surviving points are renumbered and point \
                          indexes read before the call no longer mean what they meant. A prune \
                          that retires nothing pushes no version and says so. Runs on a worker \
                          thread, so one still going after 200 ms replies with running: true and \
                          an operation_id instead of the version; cancel_background_task stops \
                          it. Needs a patch frame per point and a pixel per observation.",
            kind: Write,
            schema: object(
                &[
                    (
                        "footprint_fraction",
                        json!({
                            "type": "number",
                            "description":
                                "What fraction of an observation's projected patch radius its \
                                 footprint is. Defaults to 0.5. A patch embedded at patch size 11 \
                                 spans 5.5 feature sizes and a keypoint's support is stated at \
                                 2.5 of them, so 0.4545 is the faithful value on such a file.",
                        }),
                    ),
                    (
                        "ratio",
                        json!({
                            "type": "number",
                            "description":
                                "How many times finer the covering observation has to be. \
                                 Defaults to 2.0, which is one octave; the comparison is \
                                 non-strict.",
                        }),
                    ),
                    (
                        "min_fine_radius_px",
                        json!({
                            "type": "number",
                            "description":
                                "A covering observation whose projected radius is below this says \
                                 nothing, because a feature that projects to a fraction of a \
                                 pixel is a collapsed measurement rather than finer evidence. \
                                 Defaults to 1.0; 0 turns the floor off.",
                        }),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "delete_camera_image",
            description: "Delete one camera image, its observations, and any track left with \
                          none. A bulk edit: every image index at or after the deleted one moves \
                          down by one and the surviving points are renumbered, so indexes read \
                          before the call no longer mean what they meant. Deleting the only image \
                          is refused.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "move_camera_image",
            description: "Put one camera image at a pose, as one version of its reconstruction. \
                          The pose is world-from-camera in the reconstruction's own frame: \
                          quaternion_wxyz carries camera axes onto world axes and translation is \
                          the camera centre, which is what get_camera_image reports as center. \
                          The tracks that image observes are re-triangulated around the new pose \
                          where two or more pixels see them; a bearing only it sees turns with \
                          it, and anything else keeps its position. The image table stays put and \
                          nothing is renumbered. Undo (Ctrl+Z in the window) puts the stored pose \
                          back.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                    (
                        "world_from_camera",
                        json!({
                            "type": "object",
                            "description":
                                "Where to put the camera, in the reconstruction's own frame.",
                            "properties": {
                                "quaternion_wxyz": {
                                    "type": "array",
                                    "items": { "type": "number" },
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description":
                                        "Camera-to-world rotation, WXYZ. Normalised on arrival.",
                                },
                                "translation": vec3_schema(
                                    "The camera centre in world coordinates.",
                                ),
                            },
                            "required": ["quaternion_wxyz", "translation"],
                            "additionalProperties": false,
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "resect_camera_image",
            description: "Re-estimate one camera image's pose against structure held out from \
                          it, and install the answer as the reconstruction's next version. The \
                          correspondences are the reconstruction's tracks and the clusters of \
                          its cluster patches file, so that file must be current; a missing or \
                          stale one is refused with the reason, and build_index_files makes \
                          it. A bulk edit: the points the image observes are re-triangulated \
                          and the surviving ones are renumbered, while the image table stays \
                          put. A refused estimate pushes no version.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "add_camera_image_to_tracks",
            description: "Add one camera image's observations of the points it sees and does \
                          not already observe, as one version of its reconstruction. For each \
                          such point the patch is projected into the image; where it is in \
                          frame, not grazing and facing the camera, the image is searched for \
                          it against the consensus of the point's existing observations, and \
                          the sighting is added when its ZNCC reaches the image's pooled bar or \
                          the point's own track's bar and its keypoint lies within the image's \
                          positional bound of the projection. Nothing else moves: no point, \
                          frame, bitmap or camera, and nothing is re-triangulated, so every \
                          index still means what it meant. The step to take after \
                          resect_camera_image. Runs on a worker thread, so one still going \
                          after 200 ms replies with running: true and an operation_id instead \
                          of the version; cancel_background_task stops it. A call that adds \
                          nothing pushes no version. The Action Log row names how many tracks \
                          were joined and why the other candidates were refused. Needs a posed \
                          image whose photograph can be read, and a reconstruction with \
                          embedded patches and a patch frame per point.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "bundle_adjust",
            description: "Refine every pose and every point of one reconstruction against its \
                          observations, as one version. A bulk edit, and it runs on a worker \
                          thread, so the window stays live while it solves. An adjustment that \
                          finishes quickly replies with the version it pushed and a report \
                          carrying the counts and the median residual before and after; one that \
                          is still going after 200 ms replies instead with running: true and an \
                          operation_id, and the outcome is then read out of get_action_log or \
                          stopped with cancel_background_task. Needs inline keypoints. Each \
                          camera the posed images use is solved through its own lens, and what \
                          each camera releases is decided camera by camera: release_focal and \
                          release_distortion are the defaults every camera takes, and cameras \
                          overrides them for the cameras it names. A camera with neither is held. \
                          The label says what each camera released, and the report names each \
                          released camera's focal before and after.",
            kind: Write,
            schema: object(
                &[
                    (
                        "release_focal",
                        flag(
                            "The default for every camera: solve its focal length as well as \
                             the poses and points. Refused, naming the camera, when a camera \
                             the posed images use that takes it has a model whose focal the \
                             adjustment cannot solve. Defaults to false, which holds each focal \
                             where it is.",
                        ),
                    ),
                    (
                        "release_distortion",
                        flag(
                            "The default for every camera: solve its lens distortion together \
                             with its focal: k1 on SIMPLE_RADIAL_FISHEYE, the radial spline on \
                             SFMTOOL_FISHEYE and SFMTOOL_PINHOLE. Refused, naming the camera, \
                             for a camera that takes it without its focal, since neither k1 nor \
                             the spline can change the scale at the centre of the image, and \
                             for a camera the posed images use whose model has no such \
                             distortion; hold such a camera through cameras. Defaults to false.",
                        ),
                    ),
                    (
                        "cameras",
                        json!({
                            "type": "array",
                            "description":
                                "Per-camera overrides of release_focal and release_distortion. \
                                 Each entry names a camera by camera_intrinsics_index, the index \
                                 get_camera_intrinsics and get_camera_image report, and states \
                                 what it releases; a field left out takes the call's default. A \
                                 camera no entry names takes the defaults. Refused for an index \
                                 the reconstruction has no camera at, or one named twice. A \
                                 camera no posed image uses is not in the solve, and its release \
                                 is ignored.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "camera_intrinsics_index": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "description": "The camera's index in the reconstruction's camera table.",
                                    },
                                    "release_focal": flag("Solve this camera's focal length. Omit for the call's release_focal."),
                                    "release_distortion": flag("Solve this camera's lens distortion, together with its focal. Omit for the call's release_distortion."),
                                },
                                "required": ["camera_intrinsics_index"],
                                "additionalProperties": false,
                            },
                        }),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "switch_camera_model",
            description: "Switch one camera intrinsics record to a camera model fitted to it, \
                          and install the answer as the reconstruction's next version. The new \
                          model is fitted to the old one where the old one is trusted; poses, \
                          points, keypoints and tracks do not move, and the stored errors of \
                          the points the camera's images observe are recomputed. With \
                          camera_model omitted the target is the camera's own model, which for an \
                          SFMTOOL_FISHEYE or SFMTOOL_PINHOLE camera is a refit of its spline \
                          to another coefficient count or domain end: fitted over the whole \
                          new domain and kept monotone, the domain end kept exactly unless \
                          spline_domain_deg is given. This is how a spline's count or domain \
                          changes; bundle_adjust then refines the coefficients it has. The \
                          reply carries the version, the Action Log sentence, and fit: the \
                          fit's rms and max pixel distance from the old camera, \
                          theta_fit_deg and its source, the spline domain, the range of \
                          incidence angles where the monotonicity constraint bound, and the \
                          median reprojection error of the camera's observations before and \
                          after. A refusal names the camera and the rule.",
            kind: Write,
            schema: object(
                &[
                    (
                        "camera_model",
                        json!({
                            "type": "string",
                            "description":
                                "The target model, case-insensitive: SFMTOOL_FISHEYE, \
                                 SFMTOOL_PINHOLE, EQUIDISTANT_FISHEYE or a COLMAP lens model. \
                                 Omit for the camera's own model.",
                        }),
                    ),
                    (
                        "coeff_count",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 32,
                            "description":
                                "The spline coefficient count of a spline target, 0 or 2 to \
                                 32. Omit for the camera's own count when the target is its \
                                 own spline model, and 8 otherwise. Refused for a model \
                                 without a spline.",
                        }),
                    ),
                    (
                        "spline_domain_deg",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 180,
                            "description":
                                "Where a spline target's domain ends, as an incidence angle in \
                                 degrees (below 90 for SFMTOOL_PINHOLE). Omit to keep the \
                                 camera's own in a refit of its spline, and for the far image \
                                 corner otherwise. get_camera_intrinsics reports the outermost \
                                 keypoint's angle to set it from.",
                        }),
                    ),
                    (
                        "theta_fit_deg",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 180,
                            "description":
                                "The largest incidence angle the fit samples. Omit for the \
                                 camera's trusted bound, or its observations' extent for a \
                                 model without one; a refit of a spline samples its whole \
                                 domain. Given, even a refit of a spline is fitted over this \
                                 angle alone.",
                        }),
                    ),
                ],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "camera_intrinsics_index",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description": "The camera's index in the reconstruction's camera \
                                            table, as get_camera_intrinsics takes it.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "convert_to_embedded_patches",
            description: "Change how one reconstruction locates its observations: from a                           feature index into a .sift file to a patch frame per point with the                           keypoint carried inline. Every point keeps its index, position and                           track; its (u, v) frame uses the mean viewing direction and 2.5x                           the median projected keypoint scale. Each observation's keypoint                           and each image's identity hash come from its .sift file. The viewer                           then renders persistent reference bitmaps from readable photographs                           without photometric adaptation. If no photographs can be read, the                           conversion still succeeds without bitmaps. One version, and it runs                           on a worker thread, so a conversion still going after 200 ms replies                           with running: true and an operation_id instead of the version.                           Refused on a reconstruction that already carries embedded patches.                           Needs the workspace's .sift files where the reconstruction was made.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
    ]
}
