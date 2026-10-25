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
                                "Which version, spelled as get_reconstruction_history and \
                                 the Action Log spell it: \"v12\".",
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
                          bitmaps and no absolute workspace path. The node keeps its \
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
                          good, and undo puts it back. The deleted point's slot stays open only \
                          until the next edit that gives the node a whole new base, or a save \
                          that is not a minimal copy: \
                          that edit closes it up first, so every point after it moves down by \
                          one, a point retriangulate_point or commit_bench_track re-added goes \
                          back to the index it had, and a point new to the base follows the last \
                          surviving one. Re-read point indexes after such an edit. Image indexes \
                          do not move.",
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
            name: "delete_points",
            description: "Delete many 3D points of one reconstruction, and their tracks, as one \
                          version that one undo reverses. The same point edit as delete_point, \
                          with the same reply: every other point keeps its index, and the \
                          deleted slots close up first under a later edit that gives the node a \
                          new base, as delete_point describes. The list need not be sorted, and \
                          a point named twice counts once. Every index is checked first: a list \
                          naming a point that is out of range or already deleted is refused and \
                          deletes nothing, and so is an empty list.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "points",
                        json!({
                            "type": "array",
                            "items": { "type": "integer", "minimum": 0 },
                            "description":
                                "The points to delete, by their indexes in the reconstruction \
                                 reconstruction_label names.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "retriangulate_point",
            description: "Re-solve one 3D point from its own observations, at the poses and the \
                          lenses the reconstruction already holds. Moves no camera and no other \
                          point. A point edit, so every other index stays good; a point whose \
                          geometry changes is deleted and re-added, so it takes a new index, \
                          which the reply's report names ('now point N') along with what its \
                          observations supported - a verdict such as finite, at infinity or \
                          'behind a camera that sees it, so at infinity' - and 'unchanged' where \
                          the answer is the geometry it had. A point the reconstruction holds at a fixed \
                          coordinate is refused, and a point fewer than two of whose \
                          observations state a usable ray keeps the geometry it had. Undo puts \
                          it back.",
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
                          observations, at the poses and the lenses it already holds. Moves no \
                          camera and no lens: this is the structure re-read at a geometry \
                          somebody else decided. A bulk edit giving the node a whole new base, \
                          but it deletes no point and creates none, so every index still means \
                          what it meant unless a point edit is pending, in which case the \
                          deleted slots close up first, as delete_point describes. Points the reconstruction holds at a fixed coordinate \
                          are left alone, a ranged point keeps its distance and only its \
                          direction is re-read, and a point fewer than two of whose observations \
                          state a usable ray keeps the geometry it had. The report counts the \
                          points moved, crossed to or from infinity, refused by each rule, \
                          kept and held. One version, and it runs on a worker thread, so a \
                          retriangulation still going after 200 ms replies with running: true \
                          and an operation_id instead of the version; cancel_background_tasks \
                          stops it. Needs a pixel per observation and at least one posed image.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "refit_all_points",
            description: "Fit every 3D point of one reconstruction to the render of its \
                          reference observation, as Track View's Fit followed by Commit does \
                          for one track with every observation kept: each observation's \
                          keypoint is aligned to that render again, the point is \
                          re-triangulated (or, at infinity, its direction re-read), its patch \
                          is placed at the answer and its stored bitmap rendered again, and \
                          each observation's readings are taken again. Moves no camera and no \
                          lens, and reads every photograph the points observe; one that cannot \
                          be read refuses the run. A bulk edit giving the node a whole new \
                          base, but it deletes no point and creates none, so every index still \
                          means what it meant unless a point edit is pending, in which case the \
                          deleted slots close up first, as delete_point describes. Points the \
                          reconstruction holds at a fixed coordinate are left out, a ranged \
                          point keeps its distance and only its direction is refit, and a \
                          point the fit refuses keeps every column it had. The report counts \
                          the points refit, unchanged, ranged, released, crossed to or from \
                          infinity, left as they were and why, and held. One version, and it \
                          runs on a worker thread, so a refit still going after 200 ms replies \
                          with running: true and an operation_id instead of the version; \
                          cancel_background_tasks stops it. Needs inline keypoints, a patch \
                          frame per point and at least one posed image.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "refine_all_normals",
            description: "Turn the patch of every finite 3D point of one reconstruction to its \
                          grown-plane normal, as Track View's Grown Plane Normal followed by Commit does for \
                          one track with every observation kept: one plane is fitted to the \
                          photographs, the patch is turned to it whenever a view joined unless \
                          the turn is large on weak evidence, the keypoints of the views that \
                          joined the plane or had their shift fitted to it are written, and the \
                          stored bitmap is rendered again and each observation's readings taken \
                          again. The step trusts the keypoints it is given, so refit_all_points \
                          first where they may be off. Moves no camera, no lens and no point's \
                          centre, and reads every photograph the points observe; one that \
                          cannot be read refuses the run. Points at infinity are skipped and \
                          counted. Held and ranged points are refined too and keep their \
                          constraint, since the centre does not move. A point whose normal the \
                          step leaves unchanged, or that it refuses, keeps every column it had. \
                          A bulk edit giving the node a whole new base, but it deletes no point \
                          and creates none, so every index still means what it meant unless a \
                          point edit is pending, in which case the deleted slots close up \
                          first, as delete_point describes. The report counts the points \
                          turned, unchanged, with low confidence, the keypoints written, the \
                          points whose normal was left unchanged and why, the points left as \
                          they were and why, the points at infinity skipped, and the median \
                          turn. One version, and it runs on a worker thread, so a refinement \
                          still going after 200 ms replies with running: true and an \
                          operation_id instead of the version; cancel_background_tasks stops \
                          it. Needs inline keypoints, a patch frame per point and at least one \
                          posed image.",
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
                          an operation_id instead of the version; cancel_background_tasks stops \
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
                          before the call no longer mean what they meant. The bench moves with \
                          the image set in the same version: every bench observation in the \
                          deleted image is dropped and those in later images move down by one, \
                          so observation positions read from get_bench_track before the call may \
                          name other observations after it, and an item whose observations were \
                          all in the deleted image is discarded. Deleting the only image is \
                          refused.",
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
                          it, and anything else keeps its position. The image set stays put and \
                          nothing is renumbered unless a point edit is pending, in which case \
                          the deleted slots close up first, as delete_point describes. Undo (Ctrl+Z in the window) puts the stored pose \
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
                          and the surviving ones are renumbered, while the image set stays \
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
                          it against the point's stored bitmap or its reference observation's \
                          render, and \
                          the sighting is added when its ZNCC reaches the image's pooled bar or \
                          the point's own track's bar and its keypoint lies within the image's \
                          positional bound of the projection. Nothing else moves: no point, \
                          frame, bitmap or camera, and nothing is re-triangulated, so every \
                          index still means what it meant unless a point edit is pending, in \
                          which case the deleted slots close up first, as delete_point \
                          describes. The step to take after \
                          resect_camera_image. Runs on a worker thread, so one still going \
                          after 200 ms replies with running: true and an operation_id instead \
                          of the version; cancel_background_tasks stops it. A call that adds \
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
                          stopped with cancel_background_tasks. Needs inline keypoints. Each \
                          camera the posed images use is solved through its own lens, and what \
                          each camera releases is decided camera by camera: release_focal, \
                          release_distortion and release_principal_point are the defaults every \
                          camera takes, and cameras overrides them for the cameras it names. A \
                          camera with none of them is held. The label says what each camera \
                          released, and the report names each released camera's focal and \
                          principal point before and after. By default every free point \
                          is solved in inverse depth and stored as a position or a direction by \
                          the point-or-bearing test at the end of the solve, and the report says \
                          at what noise level and how many crossed each way; \
                          free_points_cross: false keeps each point as it is. schedule sets the \
                          trim schedule, [[50, 5], [12, 2], [4, 1]] by default; \
                          [[null, 1000]] keeps every observation that projects and solves \
                          close to least squares.",
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
                        "release_principal_point",
                        flag(
                            "The default for every camera: solve its principal point, the pixel \
                             its optical axis passes through, as well as the poses and points. \
                             Accepted on every camera model, with or without the focal. A step \
                             that would put the principal point off the image is rejected. With \
                             a narrow field of view the data barely tell a principal-point shift \
                             from a turn of every image, so read how far it moved in the report. \
                             Defaults to false.",
                        ),
                    ),
                    (
                        "free_points_cross",
                        flag(
                            "Solve every free point in inverse depth and store it as a position \
                             or a direction by the point-or-bearing test at the end of the \
                             solve, so a point whose rays give no depth becomes a direction and \
                             a direction whose rays give one a position. Defaults to true; false \
                             keeps each point in the representation it has.",
                        ),
                    ),
                    (
                        "schedule",
                        json!({
                            "type": "array",
                            "minItems": 1,
                            "description":
                                "The trim schedule, one [trim_px, loss_scale] pair per round, \
                                 the Bundle Adjust dialog's Schedule box. Before each round every \
                                 observation whose reprojection error is at or over trim_px is \
                                 left out of it, and the round solves the rest under a soft-L1 \
                                 loss of scale loss_scale, in pixels. trim_px is a positive \
                                 number, or null for no trim, which keeps every observation that \
                                 projects (JSON has no infinity); loss_scale is a positive \
                                 number. Omitted, the default [[50, 5], [12, 2], [4, 1]], \
                                 whatever the dialog last ran on the node. Refused, naming the \
                                 round, when a value is out of range. [[null, 1000]] keeps \
                                 every observation and solves close to least squares.",
                            "items": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": { "type": ["number", "null"] },
                            },
                        }),
                    ),
                    (
                        "cameras",
                        json!({
                            "type": "array",
                            "description":
                                "Per-camera overrides of release_focal, release_distortion and \
                                 release_principal_point. \
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
                                        "description": "The camera's index among the reconstruction's cameras.",
                                    },
                                    "release_focal": flag("Solve this camera's focal length. Omit for the call's release_focal."),
                                    "release_distortion": flag("Solve this camera's lens distortion, together with its focal. Omit for the call's release_distortion."),
                                    "release_principal_point": flag("Solve this camera's principal point. Omit for the call's release_principal_point."),
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
                          and install the answer as the reconstruction's next version. Two \
                          forms. The direct form, with camera_intrinsics_index and the fit's \
                          arguments, is a lens-only switch: the new model is fitted to the old \
                          one where the old one is trusted; poses, points, keypoints and \
                          tracks do not move, and the stored errors of the points the camera's \
                          images observe are recomputed. No index moves unless a point edit is \
                          pending, in which case the deleted slots close up first, as \
                          delete_point describes. With camera_model omitted the target is the \
                          camera's own model, which for an SFMTOOL_FISHEYE or SFMTOOL_PINHOLE \
                          camera is a refit of its spline to another coefficient count or \
                          domain end: fitted over the whole new domain and kept monotone, the \
                          domain end kept exactly unless spline_domain_deg is given. This is \
                          how a spline's count or domain changes; bundle_adjust then refines \
                          the coefficients it has. The direct form's reply carries the \
                          version, the Action Log sentence, and fit: the fit's rms and max \
                          pixel distance from the old camera, theta_fit_deg and its source, the \
                          spline domain, the range of incidence angles where the monotonicity \
                          constraint bound, and the median reprojection error of the camera's \
                          observations before and after. A camera model proposal open on the \
                          reconstruction (propose_camera_model) is applied first; cancel it \
                          first to make the direct switch alone. The proposal form, with \
                          proposal: true and none of the fit's arguments, applies the open \
                          proposal as the panel's Apply does: the switch and the \
                          retriangulation of every track the camera's images observe, in one \
                          version. Where the proposal's job already holds that version for the \
                          current generation (its triangulation state is ready) it is \
                          installed at once and the reply is the version; otherwise it runs \
                          on a worker thread, so one still going after 200 ms replies with \
                          running: true and an operation_id instead of the version; on a \
                          reconstruction whose observations carry no inline \
                          keypoints it is the switch alone, and the sentence says why. A \
                          refusal names the camera and the rule.",
            kind: Write,
            schema: object(
                &[
                    (
                        "camera_intrinsics_index",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "description": "The camera's index in the reconstruction's camera \
                                            table, as get_camera_intrinsics takes it. Required \
                                            in the direct form; in the proposal form, \
                                            optional and refused unless it is the proposal's \
                                            camera.",
                        }),
                    ),
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
                    (
                        "proposal",
                        flag(
                            "Apply the camera model proposal open on the reconstruction \
                             instead of fitting one: the switch and the retriangulation of the \
                             camera's tracks, as one version. Takes none of the fit's \
                             arguments.",
                        ),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "propose_camera_model",
            description: "Open a camera model proposal on one camera, or refit the one open on \
                          it, exactly as the Camera Intrinsics panel's Switch Model... and its \
                          controls do, and select that camera. The proposal is the camera \
                          fitted to another model, or a spline camera refitted to another \
                          coefficient count or domain end, shown in the panel beside the \
                          current camera before it is applied; it pushes no version. Each \
                          argument takes the default switch_camera_model gives it. A call whose \
                          fit arguments equal the open proposal's keeps it and its generation. \
                          A fit the core refuses still opens a proposal, which carries the \
                          refusal and cannot be applied; an unknown model, or a coeff_count on \
                          a model without a spline, opens nothing. Refused on a busy \
                          reconstruction, on one with no posed image, and while a proposal is \
                          open on another camera of the same reconstruction. A proposal open \
                          on another reconstruction is applied first, as selecting another \
                          reconstruction applies it. The reply is the proposed block \
                          get_camera_intrinsics also reports, which carries the observation \
                          comparison once the proposal's job has switched the camera, and a \
                          triangulation block whose state is off, pending, ready (with the \
                          preview's counts and lost_points, the points that lose their \
                          position) or refused. The job runs after the reply, so poll \
                          get_camera_intrinsics until triangulation.state is ready for the \
                          generation this call returned. Apply with switch_camera_model \
                          { proposal: true }, drop with cancel_camera_model_proposal; an undo, \
                          a redo or a history jump on the reconstruction drops it, and any \
                          other edit tool naming the reconstruction, a close, or selecting \
                          another reconstruction applies it.",
            kind: Write,
            schema: object(
                &[
                    (
                        "camera_model",
                        json!({
                            "type": "string",
                            "description":
                                "The target model, case-insensitive. Omit for the camera's own \
                                 model, which for a spline camera is the refit of its spline.",
                        }),
                    ),
                    (
                        "coeff_count",
                        json!({
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 32,
                            "description":
                                "The spline coefficient count of a spline target. Omit for the \
                                 camera's own count when the target is its own spline model, \
                                 and 8 otherwise.",
                        }),
                    ),
                    (
                        "spline_domain_deg",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 180,
                            "description":
                                "Where a spline target's domain ends, in degrees. Omit to keep \
                                 the camera's own in a refit of its spline, and for the far \
                                 image corner otherwise.",
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
                                 default; given, even a refit of a spline is fitted over this \
                                 angle alone.",
                        }),
                    ),
                    (
                        "preview_retriangulation",
                        json!({
                            "type": "boolean",
                            "description":
                                "Tick (true) or clear (false) the panel's Preview \
                                 retriangulation: the points and surfels of the camera's \
                                 tracks move in the 3D view to where Apply will put them, and \
                                 back when cleared. Omit to leave it as it is; off when a \
                                 proposal opens. Changing it alone keeps the generation.",
                        }),
                    ),
                    (
                        "animate",
                        json!({
                            "type": "boolean",
                            "description":
                                "Whether the preview's blend s moves over about 0.7 s (true, \
                                 the default) or is set to its goal at once (false), so the \
                                 next screenshot after triangulation.state reads ready shows \
                                 the end state. Omit to leave it as it is.",
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
                            "description": "The camera's index among the reconstruction's \
                                            cameras, as get_camera_intrinsics takes it.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "cancel_camera_model_proposal",
            description: "Drop the camera model proposal open on one reconstruction, as the \
                          panel's Cancel does. Nothing changes. Refused when none is open on it.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "convert_to_embedded_patches",
            description: "Change how one reconstruction locates its observations: from a \
                         feature index into a .sift file to a patch frame per point with the \
                         keypoint carried inline. Every point keeps its position and track, \
                         and its index unless a point edit is pending, in which case the \
                         deleted slots close up first, as delete_point describes; its (u, v) \
                         frame uses the mean viewing direction and 2.5x \
                         the median projected keypoint scale. Each observation's keypoint \
                         and each image's identity hash come from its .sift file. The viewer \
                         then renders persistent reference bitmaps from readable photographs \
                         without photometric adaptation. If no photographs can be read, the \
                         conversion still succeeds without bitmaps. One version, and it runs \
                         on a worker thread, so a conversion still going after 200 ms replies \
                         with running: true and an operation_id instead of the version. \
                         Refused on a reconstruction that already carries embedded patches. \
                         Needs the workspace's .sift files where the reconstruction was made.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
    ]
}
