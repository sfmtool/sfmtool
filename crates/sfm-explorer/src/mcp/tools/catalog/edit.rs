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
                          camera the posed images use is solved through its own lens, and the \
                          report names each released camera's focal before and after.",
            kind: Write,
            schema: object(
                &[
                    (
                        "release_focal",
                        flag(
                            "Solve each camera's focal length as well as the poses and \
                             points. Refused when a camera the posed images use has a model \
                             whose focal the adjustment cannot solve. Defaults to false, which \
                             holds them where they are.",
                        ),
                    ),
                    (
                        "release_distortion",
                        flag(
                            "Solve each camera's lens distortion together with its focal, \
                             where its model has distortion the adjustment can free: k1 on \
                             SIMPLE_RADIAL_FISHEYE, the radial spline on SFMTOOL_FISHEYE and \
                             SFMTOOL_PINHOLE. Cameras of other models keep theirs. Refused \
                             without release_focal, since neither k1 nor the spline can \
                             change the scale at the centre of the image, and refused when no \
                             camera the posed images use has such a model. Defaults to false.",
                        ),
                    ),
                    (
                        "spline_coeff_count",
                        json!({
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 32,
                            "description":
                                "Refit every SFMTOOL_FISHEYE or SFMTOOL_PINHOLE camera the posed                                  images use to this many spline coefficients before the solve,                                  over its whole spline domain, and start the solve from the                                  refitted cameras; a camera already at the count is left alone.                                  Refused without release_distortion, since the new coefficients                                  only approximate the old curve until the solve fits them, and                                  when no camera is a spline model. The report names each refit                                  and its largest pixel distance from the old curve. Omit to keep                                  each count.",
                        }),
                    ),
                    (
                        "spline_domain_deg",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 180,
                            "description":
                                "Move the domain end of every SFMTOOL_FISHEYE or SFMTOOL_PINHOLE \
                                 camera the posed images use to this incidence angle, in \
                                 degrees, before the solve, in the same refit as \
                                 spline_coeff_count and over the whole new domain; a camera \
                                 already there is left alone. Past its domain the model is a \
                                 straight line the solve cannot bend. get_camera_intrinsics \
                                 reports the outermost keypoint's angle to set it from. Refused \
                                 as spline_coeff_count is, and for an angle the model cannot \
                                 end at (90 or more for SFMTOOL_PINHOLE). Omit to keep each \
                                 domain.",
                        }),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
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
