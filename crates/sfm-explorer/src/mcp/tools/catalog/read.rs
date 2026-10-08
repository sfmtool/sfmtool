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
                          labels it reports are the handles every other tool takes. Each \
                          reconstruction carries the world_space_unit its file declares, and \
                          the view carries the unit its positions and distances are in; either \
                          is null for scene units, a length of whatever size the solve gave it. \
                          The view's unit is the selected reconstruction's, as its display \
                          transform draws it: null with no selection, for a selection that \
                          declares none, or for a display scale that lands between the units. \
                          Counts only: \
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
                          infinity, its patch placement (centre, unit u and v axes, outward \
                          normal and half_extent; null with no patch frame) with its \
                          normal_confidence, and its full track — every observing camera image \
                          with the pixel it was seen at and that observation's reprojection \
                          error. When the point is the one Track View is showing with Edit \
                          clear (the viewed point), the reply adds an evaluation block: the \
                          point read as a bench track and evaluated live, off the bench, with \
                          its rows in get_bench_track's shape, its state (current, evaluating, \
                          refused, failed) and reason, the read-only bars Track View's \
                          threshold boxes hold, and each row's verdict_by_bars (in, out, or \
                          null where unmeasured), and reference_observation, the row the \
                          reference-view rule picked. While evaluating, the measurements are the \
                          last ones landed. Any other point has no evaluation block.",
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
            name: "get_viewer_3d_display",
            description: "The 3D viewport's display controls, the checkboxes and sliders of its \
                          Display HUD, as one flat document named by the fields they are stored \
                          in: the layer toggles (show_points, show_camera_images, show_grid, \
                          show_patches, show_points_at_infinity, show_target_indicator), the \
                          sizes (point_size_log2, infinity_point_px, length_scale), the patch \
                          sliders, maintain_z_up, the Advanced sliders and the two Debug \
                          overlays. These decide what a screenshot of the viewer_3d panel \
                          shows. The field of view is in get_scene's view block instead.",
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
                          from one, its observation and verdict counts, and focused_item: the \
                          focused item's label when it is on this bench, else null. The bench is a place beside the reconstruction rather \
                          than part of it — nothing on it is saved, and a commit is how it \
                          reaches the file.",
            kind: Read,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "get_bench_track",
            description: "One track on the bench, as its Track View table: the stage and its \
                          data (at the track stage, the patch placement in get_point's shape, \
                          whose normal is the one tilt_bench_patch takes), the point \
                          it came from, the thresholds, the observations selected in Track \
                          View (selected_observations, which select_bench_observations sets; \
                          empty on a track that is not focused), and every observation with what put it \
                          there, the verdict on it, where it sits and whatever each stage has \
                          measured about it. The measurements are kept evaluated: every change \
                          to the track, its max_shift_px bar among them since that is the radius \
                          each peak is looked for within, or to the reconstruction under it \
                          evaluates it again on a worker, with no call to ask for it. evaluation.state says which numbers these are: current \
                          when they are the evaluation of the track as it stands, evaluating \
                          when an evaluation of the current inputs is running or about to start \
                          (evaluation.running says which) and the numbers are the previous \
                          evaluation's, so read again until it says current; refused or failed \
                          with evaluation.reason when the \
                          track cannot be evaluated as it stands. An observation's pixel is where it sits whether or \
                          not anything has measured it: the track-stage keypoint, else the \
                          refined cluster position, else the seed it was proposed at. So a \
                          observation a search has just added says where it is without being \
                          evaluated first. An observation is addressed by its position in the \
                          list. Observations are appended and no bench step renumbers them, so \
                          an index read here still names the same observation after a verdict \
                          or a fit; delete_camera_image is the one call that renumbers them, \
                          since it renumbers the images they are in. Both the cluster and the track block carry \
                          zncc_middle beside zncc: the same samples correlated over only the \
                          middle square of the patch, half its width (the middle 12 x 12 of a \
                          24 x 24 grid). A high zncc with a low zncc_middle is an agreement \
                          carried by the parts of the patch away from the pixel, such as a \
                          background behind a small near object, the far side of a depth edge, \
                          or a texture that repeats along the epipolar line. zncc_middle is null \
                          where zncc is, where the middle is flat, and on a track read back \
                          from a committed point before it is evaluated. Both blocks also carry \
                          zncc_grid: the same samples correlated over each cell of a three by \
                          three split of the patch (8 x 8 cells of a 24 x 24 grid), every pixel \
                          weighted equally, as three rows of three from the top-left, in the \
                          layout the patch tile is drawn in. It says where in the patch an \
                          agreement or a disagreement is; a cell is null where the patch is flat \
                          over it, and the grid is null where zncc is. \
                          zncc_self_similarity_radius is how far, in grid px, \
                          the tile can slide over itself and still match itself as well \
                          as a true match between two views would: the semi-major axis of the \
                          ellipse with the same second moments per unit area about the true position \
                          as the shifts where its ZNCC against itself, interpolated bilinearly \
                          between whole-pixel shifts, is at or above that level; under 1 on a \
                          corner or busy texture, and 3 meaning 3 or more, on a straight edge \
                          or a flat patch. zncc_self_similarity_radius_middle and \
                          zncc_self_similarity_radius_grid read the middle square and each cell \
                          of the same split the same way. zncc_self_similarity_surface is the \
                          tile's ZNCC against itself at every shift of the 7 x 7 square, seven \
                          rows of seven from (dx, dy) = (-3, -3), 1 at the centre and null where \
                          the tile is flat, and zncc_self_similarity_tolerance the ZNCC deficit \
                          the tile was judged by, so the region is the surface at or above \
                          1 - tolerance. zncc_self_similarity_ellipse and \
                          zncc_self_similarity_ellipse_middle are that region's ellipse for the \
                          whole tile and its middle, as grid_px in patch-grid px (x right, y \
                          down): at the track stage the grid of the reconstruction's patch \
                          resolution R, which stage_data reports as patch_resolution, and at the \
                          cluster stage the refinement kernel's grid; image_px in the \
                          photograph's px (x right, y down; null where the tile's centre does \
                          not project); and patch, along the patch's u and v, as {kind, unit, \
                          ellipse} with kind length and unit the reconstruction's \
                          world_space_unit (null for scene units, where the file names none), \
                          or kind angle and unit degrees for a patch at infinity; patch is null \
                          at the cluster stage. Each ellipse is {axes, axes_is_at_least, \
                          major_angle, matrix}: axes [semi-major, semi-minor], each capped at 3 \
                          in grid_px, which the other units map; axes_is_at_least true per axis where the true length may be \
                          larger, because the region runs off the square searched, a shift with \
                          no reading beside it could hide more of it, or the length reached the \
                          largest radius searched; major_angle the major axis's angle in \
                          radians in [0, pi) from the frame's first axis towards its second, \
                          null for a circle; and matrix the 2 x 2 matrix E, with d^T E^-1 d = 1 \
                          on the ellipse. zncc_self_similarity_ellipse_grid gives each cell's \
                          ellipse in grid px, three rows of three, null for a cell with no \
                          reading. The \
                          self-similarity fields are \
                          null where the tile could not be read. Each observation also carries \
                          patch_jacobian and patch_zoom, the geometry of the patch in its \
                          photograph, which needs no photograph and is reported whatever the \
                          evaluation says. patch_jacobian is [[dx/dcol, dx/drow], [dy/dcol, \
                          dy/drow]] at the patch's centre, in photograph px per patch-grid px \
                          at R, the grid the shift and the grid_px ellipses are in. patch_zoom \
                          is [least, most], patch-grid px per photograph px over the warp's \
                          two singular directions, the numbers the Zoom column prints; neither \
                          depends on the resolution Track View draws its tiles at. Both are \
                          read from the patch re-anchored where the observation sits, the \
                          placement the tile is rendered through. Both are null at the \
                          cluster stage, on a track with no patch yet, for an observation with \
                          nothing saying where it sits, and for a patch whose centre is behind \
                          the camera or outside the camera model's domain; a tile whose middle \
                          is off the photograph still has both. patch_zoom is null as well for \
                          a patch seen edge on, whose smaller singular value is at most 1e-9 of \
                          the larger. Each observation also carries sampler, the sampler its \
                          tile is rendered with by the evaluation, the stored bitmap and Track View: \
                          under the default sampler rule, anisotropic where the rule moves the \
                          view, because bilinear_mip would read its less compressed axis \
                          sampler_minor_axis_loss times too coarsely and that is at least the \
                          threshold 1.5 with the larger singular value at least sqrt 2, and \
                          bilinear_mip otherwise; both are read from patch_jacobian and are null \
                          where it is, and sampler_minor_axis_loss is null as well where it is \
                          not finite. A track-stage \
                          observation the last \
                          fit kept at its seed carries walked_px (how far the fit wanted to move \
                          it), walked_to (the pixel it would have reached), walked_zncc (the \
                          ZNCC scored there, beside the row's own zncc read at the seed) and \
                          walked_zncc_middle and walked_zncc_grid (the parts' readings there); \
                          sight_bench_observation with walked_to as the pixel accepts the walk. \
                          A track-stage observation also carries the reference view's readings: \
                          viewing_angle_deg, the angle between the patch's normal and the \
                          direction to the camera at the keypoint (0 facing the patch, 90 edge \
                          on); tilt_direction_deg, the direction in the patch's plane, in \
                          degrees from u towards v, that the ray from the camera leans along \
                          and the view foreshortens the patch along (null within 0.1 degrees of \
                          facing); coverage, the share of the R x R tile's samples on the \
                          photograph; and clipped_share, the share of the photograph's pixels \
                          inside the tile's outline that are 0 or 255 in any colour channel. \
                          An in observation also carries pair_zncc, the median of its pairwise \
                          ZNCCs with the other in observations from member coherence's matrix; \
                          pair_zncc_grid, per ninth of the tile, the median over the others of \
                          the pair's ZNCC there; cell_deficit, the most its pair_zncc_grid falls \
                          below the track's typical agreement (the median over the in \
                          observations) in a ninth where that typical agreement is at least \
                          0.5; and reference_view, the reference-view \
                          rule's decision: is_reference, rejected_by (null for the reference, \
                          else the first test that turned it away: coverage under 0.99, \
                          clipped over 0.05, angle over 65 degrees, or at or past 90 once that \
                          limit is dropped, cells for a cell deficit over 0.3, agreement \
                          for a pair ZNCC more than 0.15 below the best \
                          candidate's, or sharpness for a candidate a sharper one beat or one \
                          with no self-similarity radius), and \
                          fallback (none, or which tests the rule dropped because no view \
                          passed them: without_angle, without_angle_or_cells, without_any; \
                          without_angle drops only the 65 degree limit). \
                          These are null on an out observation, which the rule does not \
                          consider. stage_data.reference_observation is the index of the row \
                          the rule picked, or null. A fit stores that row's tile as the patch \
                          bitmap: stage_data.bitmap_observation is the index of the row the \
                          stored bitmap is the tile of, null for a bitmap that names none (a \
                          mean of the rows, or one stored before the reference was recorded), \
                          and has_bitmap says whether there is one. Every track-stage \
                          observation the track has a bitmap for also carries bitmap_zncc, its \
                          windowed ZNCC with the stored bitmap over the samples both have; \
                          blur_matched_bitmap_zncc, the same after the bitmap alone is blurred \
                          by a round Gaussian until its self-similarity semi-major axis reaches \
                          the row's semi-minor axis (at most 2 grid px), where that semi-minor \
                          axis is at least a quarter longer than the bitmap's semi-major axis, \
                          and bitmap_zncc where it is not; bitmap_blur_sigma, the width of that \
                          blur in grid px, 0 where the pair was read plain; and \
                          sharper_than_bitmap, true where the row's tile is sharper than the \
                          bitmap along every direction (read plain; a candidate to replace the \
                          reference). The bitmap's own row reads 1 and is not computed.",
            kind: Read,
            schema: object(
                &[("track", bench_track_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
    ]
}
