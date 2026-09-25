// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ToolKind::Write;

pub(super) fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "create_bench_cluster",
            description: "Start a cluster-stage track on the bench from a place in one camera \
                          image, and make it the active track. A cluster is a set of image \
                          patches that register onto one template, with no geometry behind them: \
                          it wants more observations and an evaluation, and set_bench_track_stage \
                          \"track\" is what triangulates it. Seed it with a pixel (and a \
                          radius_px or an affine shape for the patch), or with a .sift feature, \
                          which carries its own position and shape. The reply names the item the \
                          rest of the bench tools take.",
            kind: Write,
            schema: object(
                &seed_properties(),
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "create_bench_track",
            description: "Put one 3D point of the reconstruction on the bench as a track-stage \
                          track, and make it the active track, so its observations can be judged \
                          one at a time and the result committed back over the point. This is \
                          what Edit on Bench does at the window: the entry on a point's menu \
                          in the 3D viewport and on a feature's menu in Image Detail, and what \
                          double-clicking either of them does. Putting on a point a track \
                          already came from activates that track rather than putting a second \
                          one on. The reply names the item.",
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
            name: "activate_bench_item",
            description: "Make one item on the bench the active one, which is the item Track \
                          View is editing and the item a bench tool acts on when it names none. \
                          A track and a cluster share the one activation.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "deactivate_bench_item",
            description: "Stop editing: every item stays on the bench and none is active, which \
                          is Track View's Edit box cleared, so the panel shows the selected \
                          point's committed track. One version; with nothing active, a no-effect \
                          reply. get_bench then reports active.track as null.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "rename_bench_item",
            description: "Give one item on the bench a label of your own, which is what names it \
                          in every later call, in the Scene tree and in each Action Log row. The \
                          reply carries the new label. A label another item already holds is \
                          refused.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                    (
                        "label",
                        json!({
                            "type": "string",
                            "description":
                                "The label the item should take. Unique on this bench, and \
                                 something other than whitespace.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "discard_bench_item",
            description: "Take one item off the bench. It is a version like any other step, so \
                          undo puts the item back where it was and active as it was; nothing of \
                          the reconstruction is touched, since nothing on the bench is part of \
                          it.",
            kind: Write,
            schema: object(
                &[],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("item", bench_item_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "add_bench_track_observation",
            description: "Add one candidate observation of a bench track, in a camera image, at \
                          the place the seed names — a pixel, a pixel with a radius_px or an \
                          affine shape, or a .sift feature. It joins as a candidate and unpinned: \
                          something proposed it and nobody has ruled on it. A second observation \
                          in an image the track already sees is allowed and is measured like any \
                          other; what it cannot do is be turned in while the other is. The reply \
                          names the index it took.",
            kind: Write,
            schema: object(
                &{
                    let mut optional = seed_properties();
                    optional.push(("track", bench_track_schema()));
                    optional
                },
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("camera_image", camera_image_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "duplicate_bench_item",
            description: "Put a copy of one item on the bench beside it, and make the copy the \
                          active one — what a second patch over neighbouring ground is started \
                          from, since a patch already fitted to one piece of surface is most of \
                          the way to the piece next to it. The copy carries everything that \
                          describes the geometry and the judgements about it: the stage and its \
                          data, every observation with its keypoint, seed, shape, verdict and \
                          pin, the measurements, and the thresholds. The one thing it does not \
                          carry is the **origin**, so a commit of the copy creates a point rather \
                          than replacing the one the original came from. Its label is the \
                          original's with \" copy\" after it. Omit item for the active track. \
                          The reply names the copy, which is the handle every later call uses.",
            kind: Write,
            schema: object(
                &[("item", bench_item_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "translate_bench_patch",
            description: "Move a bench track's patch -- the dot drag on the Image Detail panel's \
                          bench layer with Track View's Lock ticked (a sighting's dot, or the \
                          ghost outline's centre in an image the track has no sighting in), and \
                          the normal-segment drag in either panel and the dot drag in the 3D \
                          viewer. A \
                          track-stage track has one patch and every observation is a view of it, \
                          so this moves the patch and not a sighting: the centre moves, the axes \
                          and the size are kept, and every observation's keypoint is carried by \
                          that same displacement, keeping its own offset from the centre's \
                          projection, which is what the tiles are cut on. Name where it goes in \
                          exactly one of two ways. `by` is [u, v, n] on the patch's OWN \
                          orthonormal axes, in the reconstruction's world units: u and v slide it \
                          across its own plane, n moves it along its outward normal, and both \
                          together are allowed. The n part is a statement no sighting can make \
                          -- a sighting says which ray the patch lies along and nothing about how \
                          far down it the surface is -- so it is where a patch's depth is settled, \
                          and the sightings then move by DIFFERENT amounts in their photographs, \
                          that spread being the parallax the old depth was wrong by. Or name a \
                          `pixel` and the photograph it is in, as exactly one of `observation` \
                          (the outline there is the patch re-anchored on that sighting, and the \
                          sighting lands under the pixel) or `camera_image` (an index or an \
                          .sfmr relative path; the outline there is the patch as it stands, and \
                          its own centre lands under the pixel; it is the only form that reaches \
                          an image the track has no sighting in). Nothing is pinned: where the \
                          patch is says \
                          nothing about whether a sighting belongs to it. A track at infinity \
                          refuses a `by` with an n part, its normal being its own bearing. A \
                          cluster-stage track has no shared geometry -- use \
                          sight_bench_observation there.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "by",
                        json!({
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 3,
                            "maxItems": 3,
                            "description":
                                "[u, v, n] on the patch's own orthonormal axes, in the \
                                 reconstruction's world units. Give this or a pixel with \
                                 its observation or camera_image, not both.",
                        }),
                    ),
                    ("observation", observation_schema()),
                    ("camera_image", pixel_view_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "sight_bench_observation",
            description: "Put ONE observation's own sighting of a bench track at a pixel, by \
                          hand, leaving every other where it is. At the track stage this writes \
                          its keypoint, which is the pixel a commit writes; at the cluster stage \
                          it moves its seed and keeps its shape. Either way the measurements read \
                          at the old pixel are dropped, because none of them says anything about \
                          the new one, and the observation is pinned: a sighting you placed is \
                          one you have ruled on, so the thresholds leave its verdict alone. The \
                          panel's dot drag is this at the cluster stage, and at the track stage \
                          with Track View's Lock cleared: the fix for one keypoint that settled on \
                          the wrong detail. With the Lock ticked the track stage's dot moves the \
                          whole patch instead (translate_bench_patch), because there the patch is \
                          the thing every sighting is a view of. The Lock is the panel's own \
                          setting and the wire has no copy of it: which of the two tools you call \
                          is the choice it makes.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "shape_bench_observation",
            description: "Give ONE cluster-stage sighting its affine shape outright: the 2x2 map \
                          from the detector's canonical keypoint frame onto that image's pixels, \
                          shear and all. The general form of the two gestures over a \
                          parallelogram, where spin_bench_shape only turns it and \
                          resize_bench_shape only scales it. A shape is a scale rather than a \
                          size, so it is read over the cluster's own radius. The sighting is \
                          re-seeded where it is already drawn and its refinement is dropped, that \
                          having been an answer about the shape it was run at; the verdict is NOT \
                          pinned, because a shape is not a ruling on whether the sighting \
                          belongs. A track-stage track is refused: there the square is the \
                          patch's.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                    ("shape", affine_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "resize_bench_patch",
            description: "Resize a bench track's patch. Name the size in exactly one of two \
                          ways. `half_length` is a world half-length in the reconstruction's own \
                          units, and `moved_edge` says what becomes of the sightings: named \
                          (\"+u\", \"-u\", \"+v\", \"-v\"), that edge moves and the opposite one \
                          is held, so the centre shifts and every sighting is carried with it; \
                          omitted, both edges move about a held centre and no sighting is touched \
                          at all. Or name an `edge`, a `pixel` and the photograph it is in, which \
                          is the edge drag on the Image Detail panel's bench layer: the pixel is \
                          unprojected onto the patch's own plane, so the edge lands there exactly \
                          through whatever distortion the lens has. The photograph is exactly \
                          one of `observation`, whose outline is the patch re-anchored on that \
                          sighting, or `camera_image`, whose outline is the patch as it stands \
                          (the ghost outline of an image the track has no sighting in). \
                          A patch is square, so a resize is one scale and not two. Nothing \
                          is pinned. A cluster-stage track is refused: use resize_bench_shape.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "half_length",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description":
                                "The patch's new half-length along both axes, in the \
                                 reconstruction's world units. Give this or an edge and a \
                                 pixel with its observation or camera_image, not both.",
                        }),
                    ),
                    ("moved_edge", edge_schema()),
                    ("observation", observation_schema()),
                    ("camera_image", pixel_view_schema()),
                    ("edge", edge_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "resize_bench_shape",
            description: "Resize ONE cluster-stage sighting's parallelogram by putting one edge \
                          of it under a pixel, with the opposite edge left where it is -- the \
                          edge drag on the Image Detail panel's bench layer at the cluster \
                          stage. There is no shared geometry there, so the arithmetic runs in \
                          that image's pixels: the shape is scaled by one scalar, which keeps \
                          whatever anisotropy the detector read, and the sighting moves by half \
                          the change along the dragged edge's own direction, which is what holds \
                          the far edge still. Only that observation is touched. A track-stage \
                          track is refused: its square is the patch's, so use resize_bench_patch.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                    ("edge", edge_schema()),
                    ("pixel", pixel_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "tilt_bench_patch",
            description: "Turn a bench track's patch to face a new outward normal -- the \
                          arrowhead drag in the 3D viewer and in Image Detail, and the other \
                          gesture no sighting can make. A keypoint says which ray the patch lies \
                          along and nothing \
                          about which way the surface under it faces, so this is where the \
                          orientation of a patch is settled. The turn is the least rotation onto \
                          the normal named, about the axis square to the old normal and the new \
                          one, so no spin about the normal comes with it -- that is \
                          spin_bench_patch's. The centre and the half-length do not move, and \
                          every observation keeps its own in-plane offset, its keypoint becoming \
                          the projection of that offset rebuilt on the turned axes; a sighting \
                          the turned patch no longer projects into is left with no keypoint. \
                          The turn stops 80 degrees from any observation's camera, which is \
                          where that photograph would be looking along the surface rather than \
                          at it, and the reply's sentence names the observation that stopped it. \
                          Nothing is pinned, and a track at infinity is refused: its normal is \
                          its own bearing.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "normal",
                        json!({
                            "type": "array",
                            "items": { "type": "number" },
                            "minItems": 3,
                            "maxItems": 3,
                            "description":
                                "[x, y, z] in the reconstruction's own coordinates: the outward \
                                 normal the patch should face. Any non-zero length -- only the \
                                 direction is read.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "spin_bench_patch",
            description: "Turn a bench track's patch about its own outward normal -- the corner \
                          drag on the Image Detail panel's bench layer at the track stage. The \
                          square turns in place, keeping its centre, its size and the face it \
                          shows, so what changes is only which way up it sits and no sighting \
                          moves at all. That is what makes it a different step from a tilt, \
                          which turns the normal itself. A cluster-stage track is refused: it \
                          has no patch to turn, only one affine shape per sighting, so use \
                          spin_bench_shape.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "degrees",
                        json!({
                            "type": "number",
                            "description":
                                "How far to turn, in degrees, positive about the patch's outward \
                                 normal.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "spin_bench_shape",
            description: "Turn ONE cluster-stage sighting's parallelogram in its own image's \
                          pixels -- the corner drag on the Image Detail panel's bench layer at \
                          the cluster stage. The shape spins about the sighting and keeps its \
                          size and its shear; the sighting itself does not move, and its \
                          refinement is dropped because that was an answer about the shape it \
                          was run at. A track-stage track is refused: there one patch turns \
                          about its own normal, so use spin_bench_patch.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    ("observation", observation_schema()),
                ],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "degrees",
                        json!({
                            "type": "number",
                            "description":
                                "How far to turn, in degrees, positive from +x toward +y of the \
                                 image raster.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "set_bench_track_verdict",
            description: "Rule on one observation of a bench track by hand: in, out, or back to \
                          candidate. A verdict set this way is pinned, which is what leaves it \
                          alone when the thresholds are applied. A track cannot see one image \
                          twice, so turning an observation in while another observation of the \
                          same image is in is refused. Setting the verdict an observation already \
                          has changes nothing and pushes no version.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                    (
                        "verdict",
                        json!({
                            "type": "string",
                            "enum": ["in", "out", "candidate"],
                            "description":
                                "in: the observation belongs to the track, and a commit writes \
                                 it. out: it was refused, and stays in the list so the refusal \
                                 is visible. candidate: proposed and not ruled on.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "apply_bench_track_thresholds",
            description: "Set a bench track's bars and turn them into verdicts in one step: \
                          every unpinned observation is judged against them and takes the verdict \
                          they propose, and an observation ruled on by hand is left alone. A bar \
                          the call does not name stays where the track has it. The reply's report \
                          says how many were turned in, turned out, left pinned and left \
                          unmeasured.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    (
                        "min_zncc",
                        threshold_schema(
                            "The ZNCC an observation has to reach: the achieved template ZNCC at \
                             the cluster stage, the leave-one-out ZNCC at the track stage.",
                        ),
                    ),
                    (
                        "max_shift_px",
                        threshold_schema(
                            "How far an observation may sit from its seed (cluster stage) or \
                             from the patch's projection (track stage), in source-image px.",
                        ),
                    ),
                    (
                        "max_keypoint_uncertainty",
                        threshold_schema(
                            "The largest tile localizability sigma_pos an observation may carry, \
                             in template-grid px.",
                        ),
                    ),
                    (
                        "min_relative_zncc",
                        threshold_schema(
                            "The fraction of the track's own self-agreement a candidate's ZNCC \
                             has to reach.",
                        ),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "split_bench_track",
            description: "Move the named observations off a bench track onto a second track \
                          beside it, whatever their verdicts were, and answer with the label the \
                          new item took. The half that comes off arrives at the cluster stage: a \
                          split questions the 3D hypothesis fitted to both halves, so carrying it \
                          onto the new half would state it as fact. Splitting off every \
                          observation, or none, is refused.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "observations",
                        json!({
                            "type": "array",
                            "items": { "type": "integer", "minimum": 0 },
                            "description":
                                "Which observations to take off, by their positions in \
                                 get_bench_track's list.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "commit_bench_track",
            description: "Write a bench track into the reconstruction as one version: its in \
                          observations become the track of a point, over the point it came from \
                          where it came from one and as a new point otherwise. Refused for a \
                          track still at the cluster stage, for one with fewer than two in \
                          observations, and on a reconstruction whose observations are .sift \
                          feature indexes rather than inline keypoints — each in the bench's own \
                          words. The track stays on the bench, seated on the point it wrote, and \
                          that point becomes the viewer's selection; the reply names it by index \
                          and by id. A commit onto a point that already holds exactly this track \
                          writes nothing: it pushes no version and answers changed: false with \
                          that point named as usual.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "evaluate_bench_track",
            description: "Read every observation of a bench track at the stage the track is in \
                          and MOVE NOTHING — the position, the frame and every keypoint come \
                          back as they were. It is the refinement against the template at the \
                          cluster stage, and at the track stage one round of the localizer at \
                          the pixels the observations already sit at: each gets its \
                          leave-one-out ZNCC, seed_shift_px (how far the correlation peak sits \
                          from the observation), projection_offset_px (how far the observation \
                          sits from the point's projection — the number that says the point is \
                          off, not the sighting), the reprojection error, the ray angle and its \
                          tile localizability. No gate drops a row: an observation that cannot \
                          be read carries a reason sentence instead of a score. Read the \
                          numbers back with get_bench_track. It runs on a worker thread, so a \
                          reading still going after 200 ms replies with running: true and an \
                          operation_id to poll with get_background_task instead of the version \
                          it pushed. Needs the photographs, which are decoded on demand.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    ("search_px", search_px_schema()),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "fit_bench_track",
            description: "Fit a bench track at the stage it is in — the step that MOVES it. At \
                          the track stage it localizes every sighting against the patch, \
                          refines each to sub-pixel, re-triangulates the in ones, re-centres \
                          the frame there and re-fuses the consensus bitmap; at the cluster \
                          stage it is the refinement, which is what a reading is too. Nothing \
                          is dropped by a gate: a sighting that does not belong is turned out \
                          with set_bench_track_verdict or by the thresholds, not deleted from \
                          the evidence. The fit ends by evaluating its own result, so the \
                          numbers it leaves behind are the ones evaluate_bench_track reports. \
                          Refused for a track stage with fewer than two in observations, which \
                          a reading permits. Answers as evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    ("search_px", search_px_schema()),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "set_bench_track_stage",
            description: "Move a bench track between its two representations. \"track\" \
                          triangulates the in observations, fits a patch to them and localizes \
                          each keypoint against it, which is what a commit needs; \"cluster\" \
                          drops the geometry and leaves the patches registering onto one \
                          template, which is what questioning a wrong position looks like. Runs \
                          on a worker thread and answers as evaluate_bench_track does. Setting \
                          the stage a track is already at changes nothing.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    (
                        "stage",
                        json!({
                            "type": "string",
                            "enum": ["cluster", "track"],
                            "description": "Which representation the track should be in.",
                        }),
                    ),
                ],
            ),
        },
        ToolSpec {
            name: "search_bench_track_descriptors",
            description: "Ask the node's SIFT index which OTHER photographs hold the patch \
                          around one observation of a bench track, and add each as a candidate. \
                          It is a constellation query, not a lookup of one descriptor: the \
                          detected keypoints within radius_px of the observation are looked up in \
                          the index, the hits are grouped by image, and an image whose hits agree \
                          on a single affine warp with at least min_inliers of them is found. That \
                          warp applied to the observation's own pixel and shape is the seed the \
                          new candidate takes, so it arrives where and at the size the warp says \
                          the patch is — evaluate_bench_track is what then scores it. An image the \
                          track already has an observation in is left alone whatever its verdict, \
                          and so is the searched image itself. Needs a CURRENT SIFT index: \
                          build_index_files or open_index_files first, and get_bench reports \
                          its state under index_files.sift_index. Runs on a worker thread and \
                          answers as evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[
                    ("track", bench_track_schema()),
                    (
                        "radius_px",
                        json!({
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "description":
                                "The constellation's radius around the observation, in the \
                                 searched image's own pixels. Omit for the radius that holds \
                                 about fifty of that image's keypoints, which is the size the \
                                 query is worth asking at: the affine is a first-order \
                                 approximation about the patch centre, so a wider patch buys \
                                 recall and loses the accuracy of the warp.",
                        }),
                    ),
                    (
                        "min_inliers",
                        json!({
                            "type": "integer",
                            "minimum": 3,
                            "description":
                                "Fewest agreeing correspondences an image needs to be found. \
                                 Omit for the query's own bar.",
                        }),
                    ),
                ],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "search_bench_track_geometry",
            description: "Ask the reconstruction's GEOMETRY which other photographs see the \
                          patch of a bench track, and add each as a candidate. This is the \
                          per-point form of the view expansion sfm embed-patches runs: the \
                          track's patch is projected into every camera of the node, the ones \
                          it does not face or that hold it behind them are dropped, and each \
                          survivor's rendered patch is scored against a reference fused from \
                          the named observation and the track's in observations. A view is \
                          admitted when that score clears the track's own min_relative_zncc \
                          bar, so apply_bench_track_thresholds moves what the next search \
                          admits. A candidate arrives at the patch's own projection, with the \
                          projected patch shape and sweep provenance, carrying no verdict and \
                          no measurement — evaluate_bench_track is what then scores it. An \
                          image the track already has an observation in is left alone whatever \
                          its verdict, so repeating the search changes nothing. Needs the TRACK \
                          stage and a fitted patch; a cluster-stage track has no geometry to \
                          project and is refused. It reads no SIFT index, unlike \
                          search_bench_track_descriptors. Runs on a worker thread and answers \
                          as evaluate_bench_track does.",
            kind: Write,
            schema: object(
                &[("track", bench_track_schema())],
                &[
                    ("reconstruction_label", edited_label_schema()),
                    ("observation", observation_schema()),
                ],
            ),
        },
        ToolSpec {
            name: "open_index_files",
            description: "Open one reconstruction's index files: its SIFT index (.kdf), \
                          which search_bench_track_descriptors queries, and its cluster patches \
                          (.matches), the SIFT index's features clustered into tracks and \
                          refined into patches. Each path omitted is the node's own, \
                          <stem>-sift-index.kdf or <stem>-cluster-patches.matches beside its \
                          .sfmr, opened when it is there and reported as state none when it is \
                          not; a path named has to open or the call is refused. A file that \
                          opens is adopted whether or not it fits: get_bench reports each \
                          file's state under index_files, current when it answers for this \
                          reconstruction as it stands and stale with a stale_reason otherwise. \
                          The cluster patches are current only when they are over this \
                          reconstruction's images, in its order, and were made from the SIFT \
                          index that is open, and that index is current. Only a current index \
                          answers a search. Nothing about the reconstruction or the bench \
                          moves, so this pushes no version and undo has nothing to take back. \
                          The viewer opens the node's own files when they are there; this is \
                          for files somewhere else, or for asking again about files that \
                          changed on disk.",
            kind: Write,
            schema: object(
                &[
                    (
                        "sift_index_path",
                        json!({
                            "type": "string",
                            "description":
                                "The .kdf to open. Omit for the node's own index path, which \
                                 get_bench reports under index_files.sift_index.",
                        }),
                    ),
                    (
                        "cluster_patches_path",
                        json!({
                            "type": "string",
                            "description":
                                "The cluster-patches .matches to open. Omit for the node's own, \
                                 which get_bench reports under index_files.cluster_patches.",
                        }),
                    ),
                ],
                &[("reconstruction_label", edited_label_schema())],
            ),
        },
        ToolSpec {
            name: "build_index_files",
            description: "Build one reconstruction's index files beside its .sfmr and open \
                          them: a SIFT index over every .sift file of the node \
                          (<stem>-sift-index.kdf), then from that index a cluster-patches file \
                          (<stem>-cluster-patches.matches) holding its features clustered and \
                          refined into patches, with the defaults of sfm match --cluster and \
                          sfm cluster-patches. The index carries one image-table row per image \
                          of the node, in the node's own order, and records each image's .sift \
                          content hash so a later re-extraction reads as stale; the cluster \
                          patches record the content hash of the index they were made from. A \
                          current SIFT index is kept when the cluster patches are the file that \
                          is missing or out of date; otherwise both are rebuilt, replacing what \
                          is there. Refused when the reconstruction has never been saved, and \
                          when no .sift file of the node can be found. It reads every \
                          descriptor and photograph of the capture, so it runs on a worker \
                          thread and answers as evaluate_bench_track does.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
        ToolSpec {
            name: "close_index_files",
            description: "Let go of the index files open beside one reconstruction, leaving \
                          the files where they are. get_bench then reports both as state none, \
                          and a search is refused until they are built or opened again. \
                          Refused when neither is open.",
            kind: Write,
            schema: object(&[], &[("reconstruction_label", edited_label_schema())]),
        },
    ]
}
