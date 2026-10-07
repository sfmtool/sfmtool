// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ── The bench ───────────────────────────────────────────────────────────
//
// One fixture again: a node whose keypoints are each point's exact projection,
// with a textured photograph cached for every image, so the two steps that read
// pixels have something to register against. What these assert is the boundary
// -- that each tool is the `AppState` call the panel makes, that an observation
// index survives the steps that follow it, and that a refusal is the bench's own
// sentence -- while what each step *does* to a track is asserted in
// `bench::tests` and in core's own tests over a plane whose numbers are known.

/// A scene holding one bench-capable node, `run_a`, selected, with a photograph
/// cached for every image.
///
/// The photograph is a pattern rather than a flat field, and its periods are a
/// few pixels and differ between the axes, for the reason `bench::tests` gives:
/// a field with nothing in it hands the correlation kernels no tile to register,
/// and the refusal would be the fixture's rather than the code's.
pub(super) fn benchable() -> (AppState, Viewer3D) {
    benchable_with(crate::state::edits::tests::projected_embedded_demo(12))
}

/// [`benchable`] over a reconstruction the caller built, so a test of the
/// finite/infinity boundary can hand it one holding a bearing.
pub(super) fn benchable_with(recon: sfmtool_core::SfmrReconstruction) -> (AppState, Viewer3D) {
    let mut state = AppState::new();
    state.append_node(SceneNode::from_path(
        std::path::Path::new("/runs/run_a.sfmr"),
        recon,
    ));
    let id = state.scene[0].id;
    state.select_recon(id);
    let camera = &state.scene[0].recon().image_table.cameras[0];
    let (w, h) = (camera.width, camera.height);
    for image in 0..state.scene[0].image_count() {
        let data: Vec<u8> = (0..(w * h * 3))
            .map(|i| {
                let p = i / 3;
                ((p % w) % 9 * 14 + (p / w) % 7 * 18) as u8
            })
            .collect();
        state.insert_photograph(
            id,
            image,
            sfmtool_core::camera::image::ImageU8::new(w, h, 3, data),
        );
    }
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];
    state.window = Some(FakeWindow::default().info());
    (state, viewer)
}

/// The point the bench tests put on the bench: it observes images 0, 1 and 2 at
/// their exact projections, which is what makes a triangulation of it well
/// conditioned.
pub(super) const BENCH_POINT: u32 = 2;

/// Put [`BENCH_POINT`] on the bench through the wire and give back its label.
#[track_caller]
pub(super) fn on_the_bench(state: &mut AppState, viewer: &mut Viewer3D) -> String {
    let reply = call(
        state,
        viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": BENCH_POINT }),
    );
    reply["item"]
        .as_str()
        .expect("a create names the item it made")
        .to_string()
}

/// Start a bench tool that goes to a worker, let it finish, and answer it the
/// way the readback phase does -- which is what [`adjusted`] does for the
/// adjustment, and for the same reason.
#[track_caller]
pub(super) fn worked(
    state: &mut AppState,
    viewer: &mut Viewer3D,
    name: &str,
    arguments: Value,
) -> Value {
    let map = arguments.as_object().cloned().expect("an object");
    let command = tools::parse(name, Some(&map)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let pending = match agent(state, viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("{name} must defer to a worker"),
    };
    state.finish_background_task();
    match super::super::edit::background_reply(state, &pending).expect("the operation finished") {
        Ok(ToolOutput::Json(value)) => value,
        Ok(ToolOutput::Png { .. }) => panic!("expected JSON, got an image"),
        Err(e) => panic!("expected success, got refusal: {e}"),
    }
}

/// The three pixel steps in order: a cluster started at a pixel, an observation
/// added at another, and a verdict on it -- each one version, and the verdict
/// visible under the index the add reported.
#[test]
fn a_cluster_an_observation_and_a_verdict_round_trip_through_get_bench_track() {
    let (mut state, mut viewer) = benchable();
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    assert_eq!(made["cursor"], made["serial"], "{made}");
    assert!(
        made["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Started {item}")),
        "{made}"
    );

    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 1,
            "pixel": [124.0, 93.0],
        }),
    );
    // Named no track, so it landed on the focused item, which is the cluster
    // the create just made.
    assert_eq!(added["item"], json!(item), "{added}");
    let observation = added["observation"].as_u64().expect("the index it took");
    assert_eq!(observation, 1, "{added}");

    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observation": observation,
            "verdict": "out",
        }),
    );

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["item"], json!(item), "{track}");
    assert_eq!(track["stage"], json!("cluster"), "{track}");
    assert_eq!(track["focused"], json!(true), "{track}");
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows.len(), 2, "{track}");
    assert_eq!(rows[0]["verdict"], json!("in"), "{track}");
    assert_eq!(rows[0]["provenance"]["kind"], json!("pixel"), "{track}");
    // The verdict is under the index the add reported, which is the claim: an
    // index an agent is holding goes on naming the observation it named.
    assert_eq!(rows[1]["observation"], json!(observation), "{track}");
    assert_eq!(rows[1]["verdict"], json!("out"), "{track}");
    assert_eq!(rows[1]["pinned"], json!(true), "{track}");
    assert_eq!(rows[1]["camera_image"], json!(1), "{track}");
    assert_eq!(
        rows[1]["cluster"]["seed_pixel"],
        json!([124.0, 93.0]),
        "{track}"
    );

    // And every step was a version of the node, walked back one at a time.
    assert_eq!(version_count(&state), 4);

    // unpin hands the verdict back to the thresholds: nothing has measured the
    // row, so it keeps its verdict and loses its pin, as one more version.
    let unpinned = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observation": observation,
            "verdict": "unpin",
        }),
    );
    assert_eq!(unpinned["verdict"], json!("out"), "{unpinned}");
    assert_eq!(unpinned["pinned"], json!(false), "{unpinned}");
    assert_eq!(version_count(&state), 5);
}

/// A copy commits as a **creation**: it carries no origin, so it writes a new
/// point rather than replacing the one the original came from.
#[test]
fn a_duplicate_carries_the_patch_and_commits_as_a_creation() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let made = call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    let copy = made["item"].as_str().expect("the copy").to_string();
    assert_eq!(copy, format!("{item} copy"), "{made}");
    assert_eq!(made["copy_of"], json!(item), "{made}");
    assert_eq!(
        made["label"].as_str().expect("a version label"),
        format!("Duplicated {item} as {copy}"),
        "{made}"
    );
    assert_eq!(version_count(&state), before + 1);

    // The copy is the focused item, so a call that names none acts on it, and
    // it carries the same sightings.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["item"], json!(copy), "{track}");
    assert_eq!(track["focused"], json!(true), "{track}");
    assert_eq!(
        track["origin"],
        json!(null),
        "a copy has no origin: {track}"
    );
    let original = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(
        track["observations"], original["observations"],
        "the copy's sightings differ from the original's"
    );
    assert!(original["origin"].is_object(), "{original}");

    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        committed["point"]["replaced"],
        json!(null),
        "a copy has to create: {committed}"
    );
    assert!(committed["point"]["index"].is_number(), "{committed}");
}

/// The patch tools are the panel's handles: the patch slid, one edge put under
/// a pixel with the far one held, and a turn in the patch's own plane. Each is
/// one version, and what they write is what the exactness claim says it is --
/// including that at the track stage a move of the centre carries **every**
/// sighting with it, because the patch is the thing they are all views of.
#[test]
fn the_patch_tools_slide_resize_and_turn_and_each_is_one_version() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    // The sighting of observation 0, where it stands.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let was: Vec<f64> = track["observations"][0]["pixel"]
        .as_array()
        .expect("every observation says where it sits")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    let to = [was[0] + 2.5, was[1] - 1.5];

    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "pixel": to,
        }),
    );
    assert_eq!(moved["item"], json!(item), "{moved}");
    assert_eq!(moved["observation"], json!(0), "{moved}");
    assert!(
        moved["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Moved {item} by ")),
        "{moved}"
    );
    assert_eq!(version_count(&state), before + 1);

    // The tool moved the patch, not the one observation it was aimed through:
    // every sighting moved, and each kept its own offset from where the centre
    // projects, which is what the tiles are cut on. The fixture's keypoints are
    // each point's exact projection, so every offset is zero and stays zero --
    // what is asserted is that the carry preserved them rather than that they
    // were reset.
    let sightings = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = sightings["observations"]
        .as_array()
        .expect("the observations");
    assert!(rows.len() > 1, "the fixture's track is a single sighting");
    for row in rows {
        // A point's rows arrive pinned, and a slide neither sets a pin nor
        // clears one.
        assert_eq!(row["pinned"], json!(true), "a slide is not a verdict");
    }
    {
        let track = state
            .bench_track(state.scene[0].id, &item)
            .expect("the track is on the bench");
        let frame = track
            .track()
            .and_then(|payload| payload.placement.clone())
            .expect("a frame");
        for observation in &track.observations {
            let (camera, pose) = crate::bench::geometry::view_of(
                &state.scene[0].edited().base.image_table,
                observation.image as usize,
            )
            .expect("the fixture's images have cameras");
            let expected = camera
                .project_homogeneous(&pose, frame.center.coords, frame.w)
                .expect("the demo's patch is in front of every camera");
            let site = observation.site().expect("a sighting");
            assert!(
                (site[0] - expected[0]).abs() < 1e-3 && (site[1] - expected[1]).abs() < 1e-3,
                "image {} should sight the centre at {expected:?}, it sights {site:?}",
                observation.image,
            );
        }
    }

    // The outline as it now stands at that sighting, so the resize can be
    // aimed at a place on it.
    let outline = |state: &AppState| {
        let track = state
            .bench_track(state.scene[0].id, &item)
            .expect("the track is on the bench");
        let sighting = &track.observations[0];
        let (camera, pose) = crate::bench::geometry::view_of(
            &state.scene[0].edited().base.image_table,
            sighting.image as usize,
        )
        .expect("the fixture's images have cameras");
        let frame = track
            .track()
            .and_then(|payload| payload.placement.clone())
            .expect("a track from a point carries the stored patch");
        let anchored = crate::bench::geometry::anchored_frame(&frame, &camera, &pose, sighting);
        (anchored, camera, pose)
    };
    let corner = |patch: &sfmtool_core::patch::cloud::OrientedPatch,
                  camera: &sfmtool_core::camera::CameraIntrinsics,
                  pose: &sfmtool_core::geometry::RigidTransform,
                  s: f64,
                  t: f64| {
        let (xyz, w) = patch.corner_homogeneous(s, t);
        camera
            .project_homogeneous(pose, xyz, w)
            .expect("in front of the camera")
    };

    let (before_patch, camera, pose) = outline(&state);
    let far_before = corner(&before_patch, &camera, &pose, -1.0, 0.0);
    let target = corner(&before_patch, &camera, &pose, 1.7, 0.0);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "edge": "+u",
            "pixel": target,
        }),
    );
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert!(
        resized["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Resized {item} to ")),
        "{resized}"
    );
    assert_eq!(version_count(&state), before + 2);

    // The claim, on the outline as it is **redrawn** -- the patch re-anchored
    // on the sighting whose edge was dragged, which is what a person sees:
    // that edge lands on the pixel the call named, the far edge has not moved,
    // and the frame is still square. A thousandth of a pixel, because the
    // redrawing goes through an `f32` keypoint slot.
    let (redrawn, _, _) = outline(&state);
    let frame = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a frame");
    assert_eq!(frame.half_extent[0], frame.half_extent[1]);
    let landed = corner(&redrawn, &camera, &pose, 1.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-3 && (landed[1] - target[1]).abs() < 1e-3,
        "the +u edge should land on {target:?}, it landed on {landed:?}",
    );
    let far_after = corner(&redrawn, &camera, &pose, -1.0, 0.0);
    assert!(
        (far_after[0] - far_before[0]).abs() < 1e-3 && (far_after[1] - far_before[1]).abs() < 1e-3,
        "the far edge moved from {far_before:?} to {far_after:?}",
    );

    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "degrees": 30.0 }),
    );
    assert_eq!(turned["degrees"], json!(30.0), "{turned}");
    assert_eq!(
        turned["label"].as_str().expect("a version label"),
        format!("Spun {item} by 30.0 degrees"),
        "{turned}"
    );
    assert_eq!(version_count(&state), before + 3);
    let after = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a frame");
    assert_eq!(after.center, frame.center, "a turn moves the patch nowhere");
    assert!((after.normal() - frame.normal()).norm() < 1e-12);

    // And the three are versions of one history, walked back one at a time.
    for _ in 0..3 {
        call(
            &mut state,
            &mut viewer,
            "undo",
            json!({ "reconstruction_label": "run_a" }),
        );
    }
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let back: Vec<f64> = track["observations"][0]["pixel"]
        .as_array()
        .expect("a place")
        .iter()
        .map(|n| n.as_f64().expect("a number"))
        .collect();
    assert!(
        (back[0] - was[0]).abs() < 1e-6 && (back[1] - was[1]).abs() < 1e-6,
        "three undos did not put the sighting back: {back:?} != {was:?}",
    );

    // And the tool that moves **one** sighting is still there, for the cluster
    // stage's dot and for a script that means one keypoint: it writes that
    // observation alone and pins it. The point's rows arrive pinned, so they
    // are handed to the bars first.
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": "all",
            "verdict": "unpin",
        }),
    );
    let one = call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "pixel": [was[0] + 2.0, was[1]],
        }),
    );
    assert!(
        one["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Moved observation 0 of {item} to (")),
        "{one}"
    );
    let rows = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(rows["observations"][0]["pinned"], json!(true), "{rows}");
    assert_eq!(rows["observations"][1]["pinned"], json!(false), "{rows}");
}

/// The two pixel forms also take a `camera_image` in place of an observation,
/// which reads the pixel against the patch **as it stands** in that image: the
/// ghost outline's square, in an image the track has no sighting in. The slide
/// lands the patch's own centre under the pixel, the resize puts the patch's
/// own edge there, each is one version, and naming both photographs at once is
/// refused.
#[test]
fn the_patch_tools_take_a_pixel_in_a_camera_image_the_track_has_no_sighting_in() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let id = state.scene[0].id;
    let track = state.bench_track(id, &item).expect("on the bench").clone();
    let unseen = (0..state.scene[0].image_count())
        .find(|i| track.observations.iter().all(|o| o.image as usize != *i))
        .expect("the fixture's track does not span every image");
    let (camera, pose) =
        crate::bench::geometry::view_of(&state.scene[0].edited().base.image_table, unseen)
            .expect("the fixture's images have cameras");
    let patch = |state: &AppState| {
        state
            .bench_track(id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a frame")
    };
    let corner = |patch: &sfmtool_core::patch::cloud::OrientedPatch, s: f64, t: f64| {
        let (xyz, w) = patch.corner_homogeneous(s, t);
        camera
            .project_homogeneous(&pose, xyz, w)
            .expect("in front of the camera")
    };
    let before = version_count(&state);

    let was = patch(&state);
    let target = corner(&was, 0.6, -0.3);
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": unseen,
            "pixel": target,
        }),
    );
    assert_eq!(moved["camera_image"], json!(unseen), "{moved}");
    assert!(moved.get("observation").is_none(), "{moved}");
    assert_eq!(version_count(&state), before + 1);
    let landed = corner(&patch(&state), 0.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-6 && (landed[1] - target[1]).abs() < 1e-6,
        "the patch's own centre should land on {target:?}, it landed on {landed:?}",
    );

    // By name, as every camera_image argument may be.
    let name = state.scene[0].recon().image_table.images[unseen]
        .name
        .clone();
    let was = patch(&state);
    let far = corner(&was, -1.0, 0.0);
    let target = corner(&was, 1.5, 0.0);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": name,
            "edge": "+u",
            "pixel": target,
        }),
    );
    assert_eq!(resized["camera_image"], json!(unseen), "{resized}");
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert_eq!(version_count(&state), before + 2);
    let grown = patch(&state);
    let landed = corner(&grown, 1.0, 0.0);
    let held = corner(&grown, -1.0, 0.0);
    assert!(
        (landed[0] - target[0]).abs() < 1e-6 && (landed[1] - target[1]).abs() < 1e-6,
        "the +u edge should land on {target:?}, it landed on {landed:?}",
    );
    assert!(
        (held[0] - far[0]).abs() < 1e-6 && (held[1] - far[1]).abs() < 1e-6,
        "the far edge moved from {far:?} to {held:?}",
    );

    for (tool, arguments) in [
        (
            "translate_bench_patch",
            json!({
                "reconstruction_label": "run_a", "observation": 0,
                "camera_image": unseen, "pixel": [10.0, 10.0],
            }),
        ),
        (
            "resize_bench_patch",
            json!({
                "reconstruction_label": "run_a", "observation": 0,
                "camera_image": unseen, "edge": "+u", "pixel": [10.0, 10.0],
            }),
        ),
    ] {
        let refused = refused_call(&mut state, &mut viewer, tool, arguments).to_string();
        assert!(
            refused.contains("observation") && refused.contains("camera_image"),
            "{tool}: {refused}"
        );
    }
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "camera_image": 999, "pixel": [10.0, 10.0] }),
    )
    .to_string();
    assert!(refused.contains("999"), "{refused}");
    assert_eq!(
        version_count(&state),
        before + 2,
        "a refusal pushes nothing"
    );
}

/// The translation's normal part, which no photograph can say: the depth of the
/// patch. The version's sentence carries the signed distance and the place the
/// centre reached.
#[test]
fn translating_a_bench_patch_along_its_normal_says_how_far_and_which_way() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);
    let frame = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a track from a point carries the stored patch")
    };
    let was = frame(&state);

    let distance = -0.042;
    let moved = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, distance] }),
    );
    assert_eq!(moved["item"], json!(item), "{moved}");
    assert_eq!(moved["by"], json!([0.0, 0.0, distance]), "{moved}");
    assert_eq!(moved["changed"], json!(true), "{moved}");
    assert_eq!(version_count(&state), before + 1);

    // The sentence a drag of the normal's segment writes, and the sign is in
    // it: an offset toward the cameras and one away from them are opposite
    // answers about how far off the patch is.
    let now = frame(&state);
    assert_eq!(
        moved["label"].as_str().expect("a version label"),
        format!(
            "Moved {item} by {distance:.3} units along its normal to ({:.3}, {:.3}, {:.3})",
            now.center.x, now.center.y, now.center.z
        ),
        "{moved}"
    );

    // And the patch really went that far along the normal, with its axes and
    // its size left alone.
    assert_eq!(now.half_extent, was.half_extent);
    assert!((now.normal() - was.normal()).norm() < 1e-12);
    assert!(
        ((now.center - was.center) - was.normal() * distance).norm() < 1e-12,
        "the patch went from {:?} to {:?}",
        was.center,
        now.center,
    );

    // A distance is relative where a pixel is absolute, so the same call again
    // moves it again rather than naming where it already stands.
    let again = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, distance] }),
    );
    assert_eq!(again["changed"], json!(true), "{again}");

    // A displacement inside the patch's own tolerance changes nothing, and it
    // says so in the normal's own words rather than in a slide's: the two
    // gestures move the patch in different directions and a reader of the Action
    // Log should be able to tell them apart without a version to read it off.
    let still = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "by": [0.0, 0.0, now.half_extent[0] * 1e-9],
        }),
    );
    assert_eq!(still["changed"], json!(false), "{still}");
    let report = still["report"].as_str().expect("a sentence");
    assert!(report.starts_with("Moved"), "{report}");
    assert!(report.contains("along its normal"), "{report}");
    assert!(report.contains("no effect"), "{report}");
    assert_eq!(version_count(&state), before + 2, "{still}");
}

/// A track at infinity has no normal standing off it -- a direction patch's
/// normal is its own bearing -- so the **normal part** of a translation is
/// refused in those words while a purely tangential one is carried; and a
/// displacement JSON cannot carry is turned away at the parse. The refusals push
/// nothing.
#[test]
fn a_bearing_refuses_a_translation_along_its_normal_and_takes_a_tangential_one() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.0, 0.0, 0.05] }),
    )
    .to_string();
    assert!(refused.contains("infinity"), "{refused}");
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": "a little" }),
    )
    .to_string();
    assert!(refused.contains("by"), "{refused}");
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    // Neither of the two ways to name where it goes, and both of them: the pair
    // is exclusive, and the sentence says which two spellings it is choosing
    // between.
    for arguments in [
        json!({ "reconstruction_label": "run_a" }),
        json!({
            "reconstruction_label": "run_a", "by": [0.01, 0.0, 0.0],
            "observation": 0, "pixel": [10.0, 10.0],
        }),
    ] {
        let refused =
            refused_call(&mut state, &mut viewer, "translate_bench_patch", arguments).to_string();
        assert!(refused.contains("by"), "{refused}");
    }
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    // And the tangential part of the very same step is carried, the moved
    // bearing renormalized onto the sphere.
    let slid = call(
        &mut state,
        &mut viewer,
        "translate_bench_patch",
        json!({ "reconstruction_label": "run_a", "by": [0.02, -0.01, 0.0] }),
    );
    assert_eq!(slid["changed"], json!(true), "{slid}");
    let patch = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a bearing carries its tangent patch");
    assert_eq!(patch.w, 0.0, "a bearing stays a bearing");
    assert!((patch.center.coords.norm() - 1.0).abs() < 1e-12);
}

/// The other patch tool that names no pixel: which way the patch faces, which
/// no photograph can say. The version's sentence carries the turn actually
/// made.
#[test]
fn tilting_a_bench_patch_turns_its_normal_and_says_how_far() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);
    let frame = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
            .expect("a track from a point carries the stored patch")
    };
    let was = frame(&state);

    // Ten degrees over toward the patch's own `+u`, which every observation can
    // still see it at, so the turn is made whole. Named at three times unit
    // length, because only the direction is read.
    let (sin, cos) = 10.0_f64.to_radians().sin_cos();
    let asked = (was.normal() * cos + was.u_axis * sin) * 3.0;
    let tilted = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({ "reconstruction_label": "run_a", "normal": [asked.x, asked.y, asked.z] }),
    );
    assert_eq!(tilted["item"], json!(item), "{tilted}");
    // The reply's normal is the unit one the patch took, not the length named.
    let replied = &tilted["normal"];
    let unit = asked.normalize();
    for (axis, want) in [unit.x, unit.y, unit.z].into_iter().enumerate() {
        let got = replied[axis].as_f64().expect("a number");
        assert!((got - want).abs() < 1e-9, "{tilted}");
    }
    assert_eq!(tilted["changed"], json!(true), "{tilted}");
    assert_eq!(version_count(&state), before + 1);
    assert_eq!(
        tilted["label"].as_str().expect("a version label"),
        format!("Tilted {item} by 10.0 degrees"),
        "{tilted}"
    );

    // The patch really faces there now, and its centre and size did not move:
    // a tilt is a turn about the centre and nothing else.
    let now = frame(&state);
    assert_eq!(now.center, was.center);
    assert_eq!(now.half_extent, was.half_extent);
    assert!(
        (now.normal() - asked.normalize()).norm() < 1e-9,
        "the patch faces {:?}, not {:?}",
        now.normal(),
        asked.normalize(),
    );

    // A normal is absolute where a distance is relative, so naming the one it
    // already faces is no turn, and it says so in its own words.
    let still = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "normal": [now.normal().x, now.normal().y, now.normal().z],
        }),
    );
    assert_eq!(still["changed"], json!(false), "{still}");
    let report = still["report"].as_str().expect("a sentence");
    assert!(report.starts_with(&format!("Tilted {item}:")), "{report}");
    assert!(report.contains("no effect"), "{report}");
    assert_eq!(version_count(&state), before + 1, "{still}");
}

/// The patch's placement is on the wire in one block, read the same way off a
/// point and off the bench track made from it, and its `normal` is the one
/// `tilt_bench_patch` takes: read, turned, sent back, read again, and committed
/// through to `get_point`.
#[test]
fn the_placement_reads_the_same_off_a_point_and_its_bench_track_and_round_trips_a_tilt() {
    let (mut state, mut viewer) = benchable();
    let vector = |value: &Value| {
        let n: Vec<f64> = value
            .as_array()
            .unwrap_or_else(|| panic!("a vector, not {value}"))
            .iter()
            .map(|c| c.as_f64().expect("a number"))
            .collect();
        nalgebra::Vector3::new(n[0], n[1], n[2])
    };
    let placement_of_track = |state: &mut AppState, viewer: &mut Viewer3D| {
        let track = call(
            state,
            viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        );
        track["stage_data"]["placement"].clone()
    };

    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": BENCH_POINT }),
    );
    let stored = point["placement"].clone();
    assert!(
        stored.is_object(),
        "an embedded point carries its patch: {point}"
    );
    on_the_bench(&mut state, &mut viewer);
    let placement = placement_of_track(&mut state, &mut viewer);
    assert_eq!(placement, stored, "the bench reads the point's own patch");

    // The normal is the outward one, u × v, and unit.
    let normal = vector(&placement["normal"]);
    let u = vector(&placement["u_axis"]);
    let v = vector(&placement["v_axis"]);
    assert!((normal - u.cross(&v)).norm() < 1e-9, "{placement}");
    assert!((normal.norm() - 1.0).abs() < 1e-9, "{placement}");

    // Turned ten degrees toward +u and sent back, it is what the next read says.
    let (sin, cos) = 10.0_f64.to_radians().sin_cos();
    let asked = normal * cos + u * sin;
    call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({ "reconstruction_label": "run_a", "normal": [asked.x, asked.y, asked.z] }),
    );
    let tilted = placement_of_track(&mut state, &mut viewer);
    assert!(
        (vector(&tilted["normal"]) - asked).norm() < 1e-9,
        "{tilted}"
    );

    // Committed, the point reads the tilted patch back, to the stored column's
    // single precision.
    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": committed["point"]["id"] }),
    );
    assert!(
        (vector(&point["placement"]["normal"]) - asked).norm() < 1e-5,
        "{point}"
    );
}

/// The cap is the whole point of the handle: the normal turns until a
/// photograph would be looking along the surface and no further, and the
/// sentence names the observation's image that stopped it.
#[test]
fn a_tilt_past_what_the_observations_can_see_stops_and_names_the_image() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let id = state.scene[0].id;
    let was = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a track from a point carries the stored patch");

    // Square to the normal the patch has, which is further over than any
    // photograph of it can still see.
    let across = was.u_axis;
    let tilted = call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({
            "reconstruction_label": "run_a",
            "normal": [across.x, across.y, across.z],
        }),
    );
    assert_eq!(tilted["changed"], json!(true), "{tilted}");
    let label = tilted["label"].as_str().expect("a version label");
    assert!(
        label.contains("stopped 80.0 degrees from "),
        "a capped tilt should say so: {label}",
    );
    // The image it names is one of the track's own, read back off the state
    // rather than written into the test.
    let names: Vec<String> = state
        .bench_track(id, &item)
        .expect("the track")
        .observations
        .iter()
        .map(|observation| {
            state.image_name(crate::scene::ImageRef::new(id, observation.image as usize))
        })
        .collect();
    assert!(
        names.iter().any(|name| label.ends_with(name.as_str())),
        "{label} names no image of {names:?}",
    );
    // And it stopped short of what was asked: the normal is still well clear of
    // the direction named.
    let now = state
        .bench_track(id, &item)
        .and_then(|track| track.track().and_then(|payload| payload.placement.clone()))
        .expect("a patch");
    assert!(
        now.normal().dot(&across) < 0.9,
        "the turn was not capped at all",
    );
    // The reply reports the normal the patch stopped at, not the one named.
    let replied: Vec<f64> = tilted["normal"]
        .as_array()
        .expect("a normal")
        .iter()
        .map(|c| c.as_f64().expect("a number"))
        .collect();
    let replied = nalgebra::Vector3::new(replied[0], replied[1], replied[2]);
    assert!(
        (replied - now.normal()).norm() < 1e-9,
        "the reply says {replied:?}, the patch faces {:?}",
        now.normal(),
    );
    assert!((replied - across).norm() > 0.1, "{tilted}");
}

/// A track at infinity has no normal standing off it to turn -- a direction
/// patch's normal is its own bearing -- and a direction that is not one is
/// turned away at the parse. Neither pushes a version.
#[test]
fn tilting_refuses_a_patch_at_infinity_and_a_normal_that_is_not_one() {
    let (mut state, mut viewer) = benchable_with(bearing_demo());
    on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "tilt_bench_patch",
        json!({ "reconstruction_label": "run_a", "normal": [0.0, 0.0, 1.0] }),
    )
    .to_string();
    assert!(refused.contains("infinity"), "{refused}");

    for normal in [json!([0.0, 0.0]), json!("up"), json!([0.0, "up", 1.0])] {
        let refused = refused_call(
            &mut state,
            &mut viewer,
            "tilt_bench_patch",
            json!({ "reconstruction_label": "run_a", "normal": normal }),
        )
        .to_string();
        assert!(refused.contains("normal"), "{normal} gave {refused}");
    }
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");
}

/// A turn at the cluster stage is one sighting's affine shape and a turn at the
/// track stage is the patch's own, so they are two tools, each refusing the
/// other's stage and naming it.
#[test]
fn the_two_spins_each_refuse_the_other_stage_and_name_it() {
    let (mut state, mut viewer) = benchable();
    // A track-stage track first, so the cluster made after it is the focused one
    // a call that names no track resolves to.
    let track_stage = on_the_bench(&mut state, &mut viewer);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    let before = version_count(&state);
    let was = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.observations[0].shape())
        .expect("a seeded sighting has a shape");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "spin_bench_patch",
        json!({ "reconstruction_label": "run_a", "degrees": 45.0 }),
    )
    .to_string();
    assert!(refused.contains("spin_bench_shape"), "{refused}");
    // And the other way about, on a track-stage track.
    let refused_there = refused_call(
        &mut state,
        &mut viewer,
        "spin_bench_shape",
        json!({
            "reconstruction_label": "run_a", "track": track_stage,
            "degrees": 45.0, "observation": 0,
        }),
    )
    .to_string();
    assert!(
        refused_there.contains("spin_bench_patch"),
        "{refused_there}"
    );
    assert_eq!(version_count(&state), before, "a refusal pushes nothing");

    let turned = call(
        &mut state,
        &mut viewer,
        "spin_bench_shape",
        json!({
            "reconstruction_label": "run_a",
            "track": item,
            "degrees": 90.0,
            "observation": 0,
        }),
    );
    assert_eq!(
        turned["label"].as_str().expect("a version label"),
        format!("Spun observation 0 of {item} by 90.0 degrees"),
        "{turned}"
    );
    assert_eq!(version_count(&state), before + 1);

    // A quarter turn in the raster's own sense: `R(90) * shape`, which sends
    // each column onto the perpendicular of the one it was.
    let shape = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.observations[0].shape())
        .expect("a seeded sighting has a shape");
    let expected = [[-was[1][0], -was[1][1]], [was[0][0], was[0][1]]];
    for (row, want) in shape.iter().zip(&expected) {
        for (got, want) in row.iter().zip(want) {
            assert!((got - want).abs() < 1e-9, "{shape:?} is not {expected:?}");
        }
    }
}

/// The cluster stage's own two shape tools: an edge that scales the
/// parallelogram and holds its far edge, and the 2x2 stated outright. Each is
/// named for the part it acts on, so each refuses a track-stage track and names
/// the tool that does belong to it.
#[test]
fn the_shape_tools_size_a_cluster_sighting_and_refuse_a_track_stage_track() {
    let (mut state, mut viewer) = benchable();
    let track_stage = on_the_bench(&mut state, &mut viewer);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let item = made["item"].as_str().expect("the new item").to_string();
    let before = version_count(&state);
    let shape_of = |state: &AppState| {
        state
            .bench_track(state.scene[0].id, &item)
            .and_then(|track| track.observations[0].shape())
            .expect("a seeded sighting has a shape")
    };
    let was = shape_of(&state);

    // The edge two px further out along `+u` than the parallelogram reaches,
    // with the far edge held: the new half-width is half way between.
    let radius = state
        .bench_track(state.scene[0].id, &item)
        .and_then(|track| track.cluster().map(|payload| payload.radius))
        .expect("a cluster carries its radius");
    let half_px = sfmtool_core::bench::half_width_px(was, radius);
    let resized = call(
        &mut state,
        &mut viewer,
        "resize_bench_shape",
        json!({
            "reconstruction_label": "run_a", "observation": 0, "edge": "+u",
            "pixel": [120.0 + half_px + 2.0, 90.0],
        }),
    );
    assert_eq!(resized["edge"], json!("+u"), "{resized}");
    assert_eq!(resized["changed"], json!(true), "{resized}");
    let grown = sfmtool_core::bench::half_width_px(shape_of(&state), radius);
    assert!(
        (grown - (half_px + 1.0)).abs() < 1e-6,
        "the far edge did not hold: {half_px} became {grown}",
    );

    // And the whole 2x2 stated outright, shear and all, which is the general
    // form the spin and the resize are two special cases of.
    let asked = [[9.0, 1.5], [0.0, 9.0]];
    let shaped = call(
        &mut state,
        &mut viewer,
        "shape_bench_observation",
        json!({
            "reconstruction_label": "run_a", "observation": 0, "shape": asked,
        }),
    );
    assert_eq!(shaped["shape"], json!(asked), "{shaped}");
    assert_eq!(shape_of(&state), asked);
    assert_eq!(version_count(&state), before + 2);

    // Both belong to the cluster stage, and both say which tool a track-stage
    // track wants instead.
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "resize_bench_shape",
        json!({
            "reconstruction_label": "run_a", "track": track_stage, "observation": 0,
            "edge": "+u", "pixel": [120.0, 90.0],
        }),
    )
    .to_string();
    assert!(refused.contains("resize_bench_patch"), "{refused}");
    let refused = refused_call(
        &mut state,
        &mut viewer,
        "shape_bench_observation",
        json!({
            "reconstruction_label": "run_a", "track": track_stage, "observation": 0,
            "shape": asked,
        }),
    )
    .to_string();
    assert!(refused.contains("cluster"), "{refused}");
    assert_eq!(
        version_count(&state),
        before + 2,
        "a refusal pushes nothing"
    );
}

/// Every observation says where it sits, whether or not anything has read it.
///
/// A candidate added to a track-stage track -- by the wire here, by a
/// descriptor search in the panel -- carries its pixel as its keypoint and
/// nothing measured until a reading is run, and `pixel` is that one answer: the
/// keypoint where there is one, the seed where there is not. So an agent can
/// look at a fresh candidate without first evaluating the track, and it is
/// looking at the place the panel's own mark and tile are drawn at
/// ([`crate::bench::observation_site`]).
#[test]
fn every_observation_reports_where_it_sits_read_or_not() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 3,
            "pixel": [130.5, 95.25],
        }),
    );
    let candidate = added["observation"].as_u64().expect("the index it took") as usize;

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(
        rows[candidate]["track"]["keypoint"],
        json!([130.5, 95.25]),
        "the added pixel is the candidate's keypoint: {track}"
    );
    assert_eq!(
        rows[candidate]["track"]["zncc"],
        Value::Null,
        "nothing has read the candidate yet: {track}"
    );
    assert_eq!(
        rows[candidate]["pixel"],
        json!([130.5, 95.25]),
        "a fresh candidate does not say where it sits: {track}"
    );
    // And an observation a reading has written reports that keypoint, which is
    // the other half of the one rule.
    assert_eq!(
        rows[0]["pixel"], rows[0]["track"]["keypoint"],
        "a read observation reports something other than its keypoint: {track}"
    );
}

/// Every track-stage row carries the Jacobian Track View's *Zoom* column reads
/// and the zoom it prints, and they are the patch's own geometry: each is
/// checked against the patch re-anchored on the row's pixel, the placement its
/// tile is rendered through, projected through the row's camera half a grid
/// px either side of its centre along each patch axis, at the patch resolution
/// `stage_data` reports, worked out here from the camera and the pose rather
/// than through the viewer's warp map.
#[test]
fn every_track_stage_row_reports_its_patch_jacobian_and_zoom() {
    use nalgebra::Point3;
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let id = state.scene[0].id;
    let bench_track = state.bench_track(id, &item).expect("on the bench");
    let sfmtool_core::bench::Stage::Track(payload) = &bench_track.stage else {
        panic!("a track made from a point is at the track stage: {track}");
    };
    let patch = payload.placement.clone().expect("a patch");
    let recon = state.scene[0].recon();
    // The reconstruction's patch resolution, which is not the 64 texels the
    // tiles are drawn at.
    let r = track["stage_data"]["patch_resolution"]
        .as_u64()
        .expect("a patch resolution") as u32;
    assert_eq!(r, crate::bench::patch_resolution(recon), "{track}");
    assert_ne!(r, 64, "this would prove less");
    let r = f64::from(r);
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows.len(), 3, "{track}");
    for row in rows {
        let index = row["camera_image"].as_u64().expect("an image") as usize;
        let image = &recon.image_table.images[index];
        let camera = &recon.image_table.cameras[image.camera_index as usize];
        let pose = crate::scene::cam_from_world(image);
        let pixel: [f64; 2] = serde_json::from_value(row["pixel"].clone()).expect("a pixel");
        let anchored = patch
            .anchored_at_keypoint(camera, &pose, pixel)
            .expect("the row's ray meets the patch");
        let project = |p: Point3<f64>| {
            let cam = image.quaternion_wxyz.to_rotation_matrix() * p.coords + image.translation_xyz;
            let (x, y) = camera
                .ray_to_pixel([cam.x, cam.y, cam.z])
                .expect("the camera sees the patch");
            [x, y]
        };
        // One grid px of the patch's `R` along each axis, in scene units: a
        // column steps along `u`, and a row along `-v`, since rows count
        // downward.
        let col_step = anchored.u_axis * (2.0 * anchored.half_extent[0] / r);
        let row_step = -anchored.v_axis * (2.0 * anchored.half_extent[1] / r);
        let across = |step: nalgebra::Vector3<f64>| {
            let (ahead, behind) = (
                project(anchored.center + step * 0.5),
                project(anchored.center - step * 0.5),
            );
            [ahead[0] - behind[0], ahead[1] - behind[1]]
        };
        let (by_col, by_row) = (across(col_step), across(row_step));
        let want = [[by_col[0], by_row[0]], [by_col[1], by_row[1]]];

        let got: [[f64; 2]; 2] =
            serde_json::from_value(row["patch_jacobian"].clone()).expect("a Jacobian");
        let scale = want.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "{row}");
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            assert!(
                (got[r][c] - want[r][c]).abs() < 1e-3 * scale,
                "entry [{r}][{c}] is {}, the projection gives {}: {row}",
                got[r][c],
                want[r][c]
            );
        }

        // The zoom is grid px per pixel: the reciprocals of the singular values,
        // least first, so their product is `1 / |det J|`.
        let [low, high]: [f64; 2] =
            serde_json::from_value(row["patch_zoom"].clone()).expect("a zoom");
        let det = (want[0][0] * want[1][1] - want[0][1] * want[1][0]).abs();
        assert!(low <= high, "{row}");
        assert!(
            (low * high * det - 1.0).abs() < 1e-2,
            "the zooms {low} and {high} do not multiply to 1 / {det}: {row}"
        );

        // The sampler is the rule's choice from the same Jacobian.
        let want_sampler = sfmtool_core::camera::sampler::SamplerChoice::per_view()
            .for_jacobian(Some(got))
            .name();
        assert_eq!(row["sampler"], json!(want_sampler), "{row}");
        let loss = row["sampler_minor_axis_loss"].as_f64().expect("a loss");
        let want_loss = sfmtool_core::camera::sampler::minor_axis_loss(
            sfmtool_core::camera::warp_map::singular_values_2x2(got),
        );
        assert!((loss - want_loss).abs() < 1e-12, "{row}");
    }
}

/// A cluster-stage row's tile is the refinement kernel's grid, not a warp of
/// a patch, so it reports no patch Jacobian and no zoom.
#[test]
fn a_cluster_stage_row_reports_no_patch_jacobian_or_zoom() {
    let (mut state, mut viewer) = benchable();
    call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["stage"], json!("cluster"), "{track}");
    let rows = track["observations"].as_array().expect("the observations");
    assert!(!rows.is_empty(), "{track}");
    for row in rows {
        assert_eq!(row["patch_jacobian"], Value::Null, "{row}");
        assert_eq!(row["patch_zoom"], Value::Null, "{row}");
        assert_eq!(row["sampler"], Value::Null, "{row}");
        assert_eq!(row["sampler_minor_axis_loss"], Value::Null, "{row}");
    }
}

/// A pixel added to a track-stage track is a keypoint a commit can write: turned
/// `in`, it commits with no fit in between, at the pixel the call named.
#[test]
fn an_added_observation_commits_at_its_pixel_without_a_fit() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 3,
            "pixel": [130.5, 95.25],
        }),
    );
    let candidate = added["observation"].as_u64().expect("the index it took");
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observation": candidate,
            "verdict": "in",
        }),
    );
    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let index = committed["point"]["index"]
        .as_u64()
        .expect("the commit names the point it wrote") as u32;

    let edited = state.scene[0].edited();
    let written = edited.point(index).expect("the committed point");
    let slot = written
        .observations()
        .iter()
        .position(|o| o.image_index == 3)
        .expect("the added image is written");
    assert_eq!(written.keypoint_xy(slot), Some([130.5, 95.25]));
}

/// A seed carrying an affine shape puts that shape on the observation, which is
/// what a caller holding a detector's keypoint frame has to be able to state.
#[test]
fn a_cluster_seeded_with_an_affine_keeps_the_shape_it_was_given() {
    let (mut state, mut viewer) = benchable();
    let shape = json!([[7.1, -0.4], [0.4, 7.1]]);
    call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "affine": shape,
        }),
    );
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        track["observations"][0]["cluster"]["seed_shape"], shape,
        "{track}"
    );
}

/// The three seed forms are alternatives, and a call that mixes them is refused
/// rather than having one of them silently win.
#[test]
fn the_seed_forms_are_exclusive() {
    let (mut state, mut viewer) = benchable();
    let both = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "feature": 4,
        }),
    );
    assert!(both.0.contains("carries its own position"), "{both}");

    let sized_twice = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
            "affine": [[7.1, 0.0], [0.0, 7.1]],
        }),
    );
    assert!(
        sized_twice.0.contains("how large the patch is"),
        "{sized_twice}"
    );

    let nowhere = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({ "reconstruction_label": "run_a", "camera_image": 0 }),
    );
    assert!(
        nowhere.0.contains("needs somewhere to seed from"),
        "{nowhere}"
    );
}

/// A feature seed is read out of the image's `.sift` file, so a node that has
/// none is refused naming the image rather than seeding from nothing.
#[test]
fn a_feature_seed_on_a_node_with_no_sift_file_is_refused() {
    let (mut state, mut viewer) = benchable();
    let error = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "feature": 4 }),
    );
    assert!(error.0.contains(".sift"), "{error}");
    assert_eq!(version_count(&state), 1, "a refusal pushed a version");
}

/// A point put on the bench is an item `get_bench` lists, focused, at the track
/// stage, seated on the point it came from.
#[test]
fn create_bench_track_lists_the_item_on_get_bench() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["focused_item"], json!(item), "{bench}");
    let items = bench["items"].as_array().expect("the items");
    assert_eq!(items.len(), 1, "{bench}");
    assert_eq!(items[0]["item"], json!(item), "{bench}");
    assert_eq!(items[0]["kind"], json!("track"), "{bench}");
    assert_eq!(items[0]["focused"], json!(true), "{bench}");
    assert_eq!(items[0]["stage"], json!("track"), "{bench}");
    assert_eq!(items[0]["origin"]["point"], json!(BENCH_POINT), "{bench}");
    assert_eq!(items[0]["counts"]["in"], json!(3), "{bench}");

    // And the track's own table says the same, with a row per observation.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let rows = track["observations"].as_array().expect("the observations");
    assert_eq!(rows.len(), 3, "{track}");
    assert_eq!(rows[0]["provenance"]["kind"], json!("origin"), "{track}");
    assert!(rows[0]["track"]["keypoint"].is_array(), "{track}");
    assert!(track["thresholds"]["min_zncc"].is_number(), "{track}");
}

/// A split answers with the label the half that came off took, and leaves two
/// items on the bench.
#[test]
fn split_answers_with_the_new_items_label() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let reply = call(
        &mut state,
        &mut viewer,
        "split_bench_track",
        json!({ "reconstruction_label": "run_a", "observations": [2] }),
    );
    let made = reply["item"].as_str().expect("the new item").to_string();
    assert_ne!(made, item, "{reply}");
    assert_eq!(reply["split_from"], json!(item), "{reply}");

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    let items = bench["items"].as_array().expect("the items");
    assert_eq!(items.len(), 2, "{bench}");
    let split = items
        .iter()
        .find(|entry| entry["item"] == json!(made))
        .expect("the half that came off");
    assert_eq!(split["counts"]["observations"], json!(1), "{bench}");
    // It comes off at the cluster stage: a split questions the position fitted
    // to both halves, so it is not carried onto the new one.
    assert_eq!(split["stage"], json!("cluster"), "{bench}");
}

/// The thresholds and the painting they produce are one step, and the bars a
/// call did not name stay where the track has them.
#[test]
fn apply_bench_track_thresholds_moves_the_bars_it_names() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let before = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    )["thresholds"]
        .clone();

    let reply = call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "min_zncc": 0.42 }),
    );
    assert_eq!(reply["thresholds"]["min_zncc"], json!(0.42), "{reply}");
    assert_eq!(
        reply["thresholds"]["max_shift_px"], before["max_shift_px"],
        "an unnamed bar moved: {reply}"
    );
    assert!(
        reply["report"]
            .as_str()
            .expect("a report")
            .contains("Applied the thresholds"),
        "{reply}"
    );
    let after = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(after["thresholds"]["min_zncc"], json!(0.42), "{after}");

    // The projection error bar is on the wire as the other bars are.
    assert_eq!(
        before["max_projection_error_px"],
        json!(sfmtool_core::bench::BENCH_MAX_PROJECTION_ERROR_PX),
        "{before}"
    );
    let reply = call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "max_projection_error_px": 1.5 }),
    );
    assert_eq!(
        reply["thresholds"]["max_projection_error_px"],
        json!(1.5),
        "{reply}"
    );
    assert_eq!(reply["thresholds"]["min_zncc"], json!(0.42), "{reply}");
}

/// A sighting the fit kept at its seed says on the wire where the walk would
/// have taken it and what it scored there, and `sight_bench_observation` with
/// that pixel accepts it.
#[test]
fn a_refused_walk_is_on_the_wire_and_sighting_its_pixel_accepts_it() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let fresh = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        fresh["thresholds"]["max_shift_px"],
        json!(sfmtool_core::bench::BENCH_MAX_SHIFT_PX),
        "{fresh}"
    );
    // A bar of zero: any move at all is a walk past it.
    call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "max_shift_px": 0.0 }),
    );
    worked(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let observations = track["observations"].as_array().expect("observations");
    let (index, walked) = observations
        .iter()
        .enumerate()
        .find(|(_, o)| !o["track"]["walked_to"].is_null())
        .unwrap_or_else(|| panic!("no walk was refused: {track}"));
    let measured = &walked["track"];
    let to = measured["walked_to"].as_array().expect("a pixel");
    assert_eq!(to.len(), 2, "{measured}");
    let to = [
        to[0].as_f64().expect("a number"),
        to[1].as_f64().expect("a number"),
    ];
    assert!(measured["walked_px"].is_number(), "{measured}");
    assert!(
        measured["walked_zncc"].is_number() || measured["walked_zncc"].is_null(),
        "{measured}"
    );

    call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({
            "reconstruction_label": "run_a",
            "observation": index,
            "pixel": to,
        }),
    );
    let after = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let row = &after["observations"][index];
    assert_eq!(row["pinned"], json!(true), "{row}");
    assert!(row["track"]["walked_to"].is_null(), "{row}");
    let keypoint = row["track"]["keypoint"].as_array().expect("a keypoint");
    for (axis, want) in to.iter().enumerate() {
        let got = keypoint[axis].as_f64().expect("a number");
        assert!((got - want).abs() < 1e-3, "{row}");
    }
}

/// A commit answers with the version it pushed and the sentence it recorded,
/// and an undo takes it back -- the bench steps being versions of the node like
/// any other, which is why there is no bench undo.
#[test]
fn commit_answers_with_a_version_an_undo_takes_back() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let points = state.scene[0].point_count();

    let reply = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert_eq!(reply["item"], json!(item), "{reply}");
    assert_eq!(reply["dirty"], json!(true), "{reply}");
    let report = reply["report"].as_str().expect("a report");
    assert!(report.starts_with("Committed track:"), "{report}");
    assert_eq!(version_count(&state), 3);
    assert_eq!(state.scene[0].point_count(), points, "{report}");
    // It is an `Edit` row and the agent's, as every other commit's is, and the
    // selection the commit moved onto the written point is the row after it.
    let mut rows = rows(&state);
    let selection = rows.pop().expect("the commit selected what it wrote");
    assert!(
        selection.2.starts_with("Selected point "),
        "{}",
        selection.2
    );
    let last = rows.pop().expect("one row per step");
    assert_eq!(last, (Actor::Mcp, false, report.to_string()));

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        state.scene[0]
            .edited()
            .point(BENCH_POINT)
            .is_some_and(|point| point.observations().len() == 3),
        "the undo did not restore the point the commit replaced"
    );
}

/// `unfocus_bench_item` is Track View's Edit box cleared: no version, the
/// item left on the bench, `get_bench` then reporting no focused item over a
/// bench that has items, and a track tool that names none refused with the
/// remedies. A second call, with nothing focused, is a no-effect reply.
#[test]
fn unfocus_leaves_the_item_and_nothing_focused() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let reply = call(&mut state, &mut viewer, "unfocus_bench_item", json!({}));
    assert_eq!(reply["changed"], json!(true), "{reply}");
    assert_eq!(reply["item"], json!(item), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");
    assert_eq!(version_count(&state), before, "an unfocus pushed a version");
    let last = state.action_log.entries().last().expect("a row");
    assert_eq!(last.kind, crate::action_log::Kind::Selection);
    assert_eq!(last.text, format!("Stopped editing {item}"));

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["focused_item"], Value::Null, "{bench}");
    assert_eq!(bench["items"].as_array().map(Vec::len), Some(1), "{bench}");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        refused.0,
        "No item on run_a's bench is focused. Name one with track, focus one with \
         focus_bench_item, or put one on with create_bench_track or create_bench_cluster."
    );

    let again = call(&mut state, &mut viewer, "unfocus_bench_item", json!({}));
    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(again["item"], Value::Null, "{again}");
    assert_eq!(
        version_count(&state),
        before,
        "a no-effect call pushed a version"
    );
    let last = state.action_log.entries().last().expect("a row");
    assert!(last.text.contains("no effect"), "{}", last.text);
}

/// `focus_bench_item` pushes no version, answers `changed` for whether the
/// focused item moved, makes `get_bench` report the item on its own node and
/// null on another, and is not refused while a task holds the node.
#[test]
fn focus_pushes_no_version_and_get_bench_reports_the_focused_item() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    call(&mut state, &mut viewer, "unfocus_bench_item", json!({}));
    let before = version_count(&state);
    let id = state.scene[0].id;
    state
        .start_background_task(
            crate::background::Operation::BENCH_FIT,
            id,
            Box::new(|_| crate::background::Finished::Cancelled),
        )
        .expect("nothing else is running");

    let reply = call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
    assert_eq!(reply["changed"], json!(true), "{reply}");
    assert_eq!(reply["item"], json!(item), "{reply}");
    assert_eq!(version_count(&state), before, "a focus pushed a version");
    let last = state.action_log.entries().last().expect("a row");
    assert_eq!(last.kind, crate::action_log::Kind::Selection);
    assert_eq!(last.text, format!("Editing {item}"));

    let again = call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    state.finish_background_task();

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["focused_item"], json!(item), "{bench}");

    // A second reconstruction's bench reports no focused item: there is one
    // for the viewer, and it is on run_a.
    state.append_node(crate::scene::SceneNode::demo(
        crate::state::edits::tests::projected_embedded_demo(12),
    ));
    let other = state.scene[1].label.clone();
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": other }),
    );
    assert_eq!(bench["focused_item"], Value::Null, "{bench}");
}

/// The selection rules on the wire: `select_point` of any point but the focused
/// item's origin unfocuses it and pushes no version, `select_point` of the
/// origin keeps it, `unfocus_bench_item` selects the origin, and
/// `focus_bench_item` selects the item's node and origin.
#[test]
fn select_point_unfocuses_and_focusing_and_unfocusing_select_the_origin() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let id = state.scene[0].id;
    let origin = crate::scene::PointRef::new(id, BENCH_POINT as usize);
    assert_eq!(
        state.selected_point,
        Some(origin),
        "the put selected no origin"
    );
    let before = version_count(&state);

    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": BENCH_POINT }),
    );
    assert!(state.focused_item().is_some(), "the origin unfocused it");

    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": 0 }),
    );
    assert!(state.focused_item().is_none(), "another point kept it");
    assert_eq!(
        state.selected_point,
        Some(crate::scene::PointRef::new(id, 0))
    );
    assert_eq!(
        version_count(&state),
        before,
        "a selection pushed a version"
    );
    let texts: Vec<&str> = state
        .action_log
        .entries()
        .rev()
        .take(2)
        .map(|entry| entry.text.as_str())
        .collect();
    assert_eq!(texts[1], format!("Stopped editing {item}"), "{texts:?}");
    assert!(texts[0].starts_with("Selected point "), "{texts:?}");

    call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
    assert_eq!(state.selected_point, Some(origin), "focusing kept point 0");

    state.deselect_point();
    assert!(
        state.focused_item().is_some(),
        "clearing the point unfocused"
    );
    call(&mut state, &mut viewer, "unfocus_bench_item", json!({}));
    assert_eq!(
        state.selected_point,
        Some(origin),
        "the unfocus left no origin"
    );
    assert_eq!(version_count(&state), before);
}

/// The tools the focus pair replaced are gone from the catalog and from the
/// dispatch, so a call under an old name is the ordinary unknown-tool error.
#[test]
fn the_old_activation_tools_are_unknown() {
    for name in ["activate_bench_item", "deactivate_bench_item"] {
        assert!(
            crate::mcp::tools::catalog()
                .iter()
                .all(|spec| spec.name != name),
            "{name} is still advertised"
        );
        let error = crate::mcp::tools::parse(name, None).expect_err("an unknown tool");
        assert!(error.0.contains(name), "{error}");
    }
}

/// The item calls: a rename hands back the new label, a focus moves which item
/// a call that names none acts on, and a discard empties the bench.
#[test]
fn rename_focus_and_discard_answer_with_the_item_they_acted_on() {
    let (mut state, mut viewer) = benchable();
    let first = on_the_bench(&mut state, &mut viewer);
    let second = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    )["item"]
        .as_str()
        .expect("the new item")
        .to_string();

    let renamed = call(
        &mut state,
        &mut viewer,
        "rename_bench_item",
        json!({ "reconstruction_label": "run_a", "item": second, "label": "bull-nose" }),
    );
    assert_eq!(renamed["item"], json!("bull-nose"), "{renamed}");
    let versions = version_count(&state);
    let control = refused_call(
        &mut state,
        &mut viewer,
        "rename_bench_item",
        json!({
            "reconstruction_label": "run_a",
            "item": "bull-nose",
            "label": "line1\nline2\ttab \u{0}nul",
        }),
    );
    assert!(control.0.contains("control character"), "{control}");
    assert_eq!(
        version_count(&state),
        versions,
        "a refusal pushed a version"
    );
    let gone = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": second }),
    );
    assert!(gone.0.contains("Nothing on the bench is called"), "{gone}");

    call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": first }),
    );
    let active = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(active["item"], json!(first), "{active}");

    for item in [first.as_str(), "bull-nose"] {
        call(
            &mut state,
            &mut viewer,
            "discard_bench_item",
            json!({ "reconstruction_label": "run_a", "item": item }),
        );
    }
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["items"].as_array().expect("an array").len(), 0);
    assert_eq!(bench["focused_item"], Value::Null, "{bench}");
}

/// A create call that names a label puts its item on the bench under it, in
/// the one version the create pushes; a label another item holds takes the
/// first free `" (n)"` suffix, and the reply names the label it took.
#[test]
fn a_create_names_its_item_and_a_taken_label_takes_a_suffix() {
    let (mut state, mut viewer) = benchable();
    let cluster = |label: &str| {
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
            "label": label,
        })
    };

    let before = version_count(&state);
    let first = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        cluster("nose"),
    );
    assert_eq!(first["item"], json!("nose"), "{first}");
    assert_eq!(
        version_count(&state),
        before + 1,
        "one version for the create"
    );

    let second = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        cluster("nose"),
    );
    assert_eq!(second["item"], json!("nose (2)"), "{second}");

    let track = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": BENCH_POINT, "label": "nose" }),
    );
    assert_eq!(track["item"], json!("nose (3)"), "{track}");

    // The point is on the bench already: a call naming another label is
    // refused naming the label the track has, rather than dropping "horn".
    let versions = version_count(&state);
    let renamed = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": BENCH_POINT, "label": "horn" }),
    );
    assert!(
        renamed.0.contains(&format!(
            "Point {BENCH_POINT} is on the bench already as nose (3)"
        )),
        "{renamed}"
    );
    assert!(
        renamed.0.contains("rename nose (3) to call it horn"),
        "{renamed}"
    );
    assert_eq!(
        version_count(&state),
        versions,
        "the refusal pushed a version"
    );

    // Without a label, or with the label it has, the call focuses that track,
    // and the reply's report says so rather than leaving only the label of
    // the version an earlier step pushed.
    for named in [json!(null), json!("nose (3)")] {
        let again = call(
            &mut state,
            &mut viewer,
            "create_bench_track",
            json!({ "reconstruction_label": "run_a", "point": BENCH_POINT, "label": named }),
        );
        assert_eq!(again["item"], json!("nose (3)"), "{again}");
        assert_eq!(again["changed"], json!(false), "{again}");
        assert_eq!(
            again["report"],
            json!(format!(
                "Put point {BENCH_POINT} on the bench: no effect, it is on the bench already as \
                 nose (3), now the focused item"
            )),
            "{again}"
        );
    }

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    let labels: Vec<&str> = bench["items"]
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item["item"].as_str().expect("a label"))
        .collect();
    assert_eq!(labels, ["nose", "nose (2)", "nose (3)"], "{bench}");

    let versions = version_count(&state);
    let blank = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        cluster("  "),
    );
    assert!(blank.0.contains("other than whitespace"), "{blank}");
    let control = refused_call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        cluster("line1\nline2"),
    );
    assert!(
        control.0.contains("without a control character"),
        "{control}"
    );
    assert_eq!(
        version_count(&state),
        versions,
        "a refusal pushed a version"
    );
}

/// Every refusal is the bench's own sentence and pushes no version.
#[test]
fn the_bench_refuses_in_its_own_words() {
    let (mut state, mut viewer) = benchable();

    // With nothing on the bench there is no focused item to act on, and the
    // refusal says how to get one.
    let empty = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(empty.0.contains("create_bench_track"), "{empty}");

    let item = on_the_bench(&mut state, &mut viewer);
    let unknown = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "track": "nothing-is-called-this",
            "observation": 0,
            "verdict": "out",
        }),
    );
    assert!(
        unknown.0.contains("Nothing on the bench is called"),
        "{unknown}"
    );

    let past_the_end = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 9, "verdict": "out" }),
    );
    assert!(
        past_the_end.0.contains("no observation 9"),
        "{past_the_end}"
    );

    // A cluster cannot be committed: there is no position to write.
    call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let cluster = refused_call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(cluster.0.contains("cluster"), "{cluster}");

    let before = version_count(&state);
    let discarded = refused_call(
        &mut state,
        &mut viewer,
        "discard_bench_item",
        json!({ "reconstruction_label": "run_a", "item": "nothing-is-called-this" }),
    );
    assert!(
        discarded.0.contains("nothing on the bench is called"),
        "{discarded}"
    );
    assert_eq!(version_count(&state), before, "a refusal pushed a version");
    // The item that is there is untouched by any of it.
    assert!(state.bench_track(state.scene[0].id, &item).is_some());
}

/// The two steps that read photographs go to a worker and report through the
/// same two-level reply `bundle_adjust` uses, and the evaluation that follows
/// each is on the wire under `evaluation`.
#[test]
fn fit_and_set_stage_run_as_background_tasks_and_the_evaluation_follows_them() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    assert!(
        staged["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Set {item} to the cluster stage")),
        "{staged}"
    );
    assert_eq!(
        call(
            &mut state,
            &mut viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        )["stage"],
        json!("cluster")
    );

    // The operation is the one an agent polls for, under the name the panel
    // shows it as.
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["running"], json!(false), "{task}");
    assert_eq!(task["operation"], json!("Set track stage"), "{task}");
    assert_eq!(task["reconstruction_label"], json!("run_a"), "{task}");

    // The stage change is a change of inputs, so the track reads as being
    // evaluated until the evaluation of the cluster lands.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["evaluation"]["state"], json!("evaluating"), "{track}");
    state.settle_bench_evaluation();

    // And then the measurements are on the wire, under the observation indexes
    // they were computed for.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["evaluation"]["state"], json!("current"), "{track}");
    assert_eq!(track["evaluation"]["running"], json!(false), "{track}");
    assert!(
        track["observations"][0]["cluster"]["zncc"].is_number(),
        "the evaluation measured nothing: {track}"
    );
    // The middle reading sits beside it, from the same samples.
    let middle = track["observations"][0]["cluster"]["zncc_middle"]
        .as_f64()
        .unwrap_or_else(|| panic!("no cluster zncc_middle on the wire: {track}"));
    assert!((-1.0..=1.0).contains(&middle), "{track}");
    // And the grids, as three rows of three.
    let cluster = &track["observations"][0]["cluster"];
    for key in ["zncc_grid", "zncc_self_similarity_radius_grid"] {
        let rows = cluster[key]
            .as_array()
            .unwrap_or_else(|| panic!("no cluster {key} on the wire: {track}"));
        assert_eq!(rows.len(), 3, "{track}");
        for row in rows {
            let row = row.as_array().expect("a row of cells");
            assert_eq!(row.len(), 3, "{track}");
            assert!(row.iter().all(|cell| cell.is_number() || cell.is_null()));
        }
    }
    for key in [
        "zncc_self_similarity_radius",
        "zncc_self_similarity_radius_middle",
    ] {
        let radius = cluster[key]
            .as_f64()
            .unwrap_or_else(|| panic!("no cluster {key} on the wire: {track}"));
        assert!((0.0..=3.0).contains(&radius), "{track}");
    }
    // Its ellipse: a cluster member has no patch, so none along the patch.
    let ellipse = &cluster["zncc_self_similarity_ellipse"];
    assert_eq!(
        ellipse["grid_px"]["axes"][0].as_f64(),
        cluster["zncc_self_similarity_radius"].as_f64(),
        "{track}"
    );
    assert!(ellipse["image_px"]["axes"][1].is_number(), "{track}");
    assert!(ellipse["patch"].is_null(), "{track}");
    // The tile's ZNCC against itself, seven rows of seven, 1 at the centre
    // and a number in the corners outside the disk.
    let surface = cluster["zncc_self_similarity_surface"]
        .as_array()
        .unwrap_or_else(|| panic!("no cluster zncc_self_similarity_surface on the wire: {track}"));
    assert_eq!(surface.len(), 7, "{track}");
    assert!(surface
        .iter()
        .all(|row| row.as_array().is_some_and(|row| row.len() == 7)));
    assert!(surface[0][0].is_number(), "{track}");
    assert!(
        surface[3][3] == json!(1.0) || surface[3][3].is_null(),
        "{track}"
    );
    // And per cell its ellipse, whose semi-major axis is the cell's radius.
    let ellipses = cluster["zncc_self_similarity_ellipse_grid"]
        .as_array()
        .unwrap_or_else(|| panic!("no cluster ellipse grid on the wire: {track}"));
    assert_eq!(ellipses.len(), 3, "{track}");
    for (r, row) in ellipses.iter().enumerate() {
        for (c, cell) in row.as_array().expect("a row of cells").iter().enumerate() {
            assert_eq!(
                cell["axes"][0].as_f64(),
                cluster["zncc_self_similarity_radius_grid"][r][c].as_f64(),
                "{track}"
            );
            assert_eq!(cell["axes_is_at_least"].as_array().map(Vec::len), Some(2));
            assert_eq!(cell["matrix"].as_array().map(Vec::len), Some(2));
        }
    }

    // The fit is its own step, under its own operation name, and it ends by
    // reading its result: the track stage's two distances are both on the wire
    // afterwards.
    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert!(staged["report"].is_string(), "{staged}");
    let fitted = worked(
        &mut state,
        &mut viewer,
        "fit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert!(
        fitted["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Fitted {item}")),
        "{fitted}"
    );
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["operation"], json!("Fit track"), "{task}");

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let measured = &track["observations"][0]["track"];
    for column in [
        "zncc",
        "zncc_middle",
        "seed_shift_px",
        "projection_offset_px",
        "zncc_self_similarity_radius",
        "zncc_self_similarity_radius_middle",
    ] {
        assert!(
            measured[column].is_number(),
            "{column} is not on the wire: {track}"
        );
    }
    for key in [
        "zncc_grid",
        "zncc_self_similarity_radius_grid",
        "zncc_self_similarity_ellipse_grid",
    ] {
        assert_eq!(
            measured[key].as_array().map(Vec::len),
            Some(3),
            "{key} is not on the wire: {track}"
        );
    }
    assert_eq!(
        measured["zncc_self_similarity_surface"]
            .as_array()
            .map(Vec::len),
        Some(7),
        "zncc_self_similarity_surface is not on the wire: {track}"
    );
    assert!(
        measured["zncc_self_similarity_tolerance"].is_number(),
        "zncc_self_similarity_tolerance is not on the wire: {track}"
    );
    // The ellipse of the matching shifts, in grid px, image px and along the
    // patch's axes, whole and middle.
    for key in [
        "zncc_self_similarity_ellipse",
        "zncc_self_similarity_ellipse_middle",
    ] {
        let ellipse = &measured[key];
        assert_eq!(
            ellipse["grid_px"]["axes"][0].as_f64(),
            measured[if key.ends_with("middle") {
                "zncc_self_similarity_radius_middle"
            } else {
                "zncc_self_similarity_radius"
            }]
            .as_f64(),
            "{key}: {track}"
        );
        assert!(
            ellipse["grid_px"]["axes_is_at_least"][0].is_boolean(),
            "{track}"
        );
        assert!(
            ellipse["grid_px"]["major_angle"].is_number()
                || ellipse["grid_px"]["major_angle"].is_null(),
            "{track}"
        );
        assert_eq!(
            ellipse["grid_px"]["matrix"].as_array().map(Vec::len),
            Some(2)
        );
        assert!(ellipse["image_px"]["axes"][0].is_number(), "{track}");
        // A finite patch reads lengths, in the reconstruction's own unit, or
        // with a null unit, scene units, where it names none.
        assert_eq!(ellipse["patch"]["kind"], json!("length"), "{track}");
        assert_eq!(
            ellipse["patch"]["unit"],
            json!(state.scene[0].recon().metadata.world_space_unit),
            "{track}"
        );
        assert!(
            ellipse["patch"]["ellipse"]["axes"][1].is_number(),
            "{track}"
        );
    }
    assert!(
        measured["reason"].is_null(),
        "a measured row carries no reason: {track}"
    );
    // The reference view's readings on every `in` row, and the one row the
    // rule picked, named in the stage data too.
    let rows = track["observations"].as_array().expect("rows");
    let mut picked = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let measured = &row["track"];
        for key in ["viewing_angle_deg", "coverage", "clipped_share"] {
            assert!(
                measured[key].is_number(),
                "{key} is not on the wire: {track}"
            );
        }
        // An `out` row is not considered by the rule.
        if row["verdict"] != json!("in") {
            assert!(measured["reference_view"].is_null(), "{track}");
            assert!(measured["pair_zncc"].is_null(), "{track}");
            continue;
        }
        assert!(measured["pair_zncc"].is_number(), "{track}");
        assert_eq!(
            measured["pair_zncc_grid"].as_array().map(Vec::len),
            Some(3),
            "{track}"
        );
        let standing = &measured["reference_view"];
        assert!(standing["fallback"].is_string(), "{track}");
        if standing["is_reference"] == json!(true) {
            assert!(standing["rejected_by"].is_null(), "{track}");
            picked.push(i);
        } else {
            assert!(standing["rejected_by"].is_string(), "{track}");
        }
    }
    assert_eq!(picked.len(), 1, "{track}");
    assert_eq!(
        track["stage_data"]["reference_observation"],
        json!(picked[0]),
        "{track}"
    );
}

/// An observation seeded a long way from the point's projection is **named** on
/// its row rather than searched for.
///
/// The gesture is one an agent makes by hand: a pixel typed into
/// `add_bench_track_observation` that is nowhere near where the point lands in
/// that photograph. The reading widens its window to reach the furthest seed
/// and each view's tile is the square of that width, so this is the call that
/// used to ask for hundreds of gigabytes; what it does now is say so on the row
/// and read everything else.
#[test]
fn an_observation_far_from_the_projection_is_named_rather_than_searched_for() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let added = call(
        &mut state,
        &mut viewer,
        "add_bench_track_observation",
        json!({
            "reconstruction_label": "run_a",
            "track": item,
            "camera_image": 5,
            "pixel": [24.0, 24.0],
        }),
    );
    let at = added["observation"]
        .as_u64()
        .expect("an add names the index it took") as usize;

    state.settle_bench_evaluation();

    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let row = &track["observations"][at]["track"];
    assert!(
        row["zncc"].is_null(),
        "nothing was searched for it: {track}"
    );
    let reason = row["reason"]
        .as_str()
        .expect("a row without a score says why");
    assert!(
        reason.contains("beyond the 64 px bound"),
        "the row names the bound it passed: {reason}"
    );
    // The rows that could be read were read: the bound takes one observation
    // out of the round and leaves the rest of the reading alone.
    let measured = track["observations"]
        .as_array()
        .expect("a list")
        .iter()
        .filter(|row| row["track"]["zncc"].is_number())
        .count();
    assert!(measured >= 2, "{track}");
}

/// Setting the stage a track is already at changes nothing: no task, no
/// version, the version the node stands at as the answer, and the step's own
/// no-effect sentence rather than the silence that would leave an agent reading
/// the previous step's label.
#[test]
fn setting_the_stage_a_track_is_already_at_starts_nothing() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let reply = call(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "track" }),
    );
    assert_eq!(version_count(&state), before, "{reply}");
    assert!(state.background_task().is_none());
    assert_eq!(reply["changed"], json!(false), "{reply}");
    let report = reply["report"].as_str().expect("the step's own sentence");
    assert!(report.contains("no effect"), "{reply}");
    assert!(report.contains("that stage already"), "{reply}");
    assert_ne!(
        report,
        reply["label"].as_str().expect("a version label"),
        "the reply echoed the previous step's label: {reply}"
    );
}

/// A commit that replaces a point hands the replacement an index one past the
/// end of the version it came from, and that index is a handle like any other:
/// what `get_scene` reports as the selection is what `get_point` and
/// `select_point` take back.
#[test]
fn a_committed_replacement_is_reachable_by_the_index_get_scene_reports() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": BENCH_POINT }),
    );
    call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    let scene = call(&mut state, &mut viewer, "get_scene", json!({}));
    let selected = &scene["selection"]["point"];
    let index = selected["index"].as_u64().expect("a selected point");
    assert!(
        index
            >= scene["scene"][0]["counts"]["points"]
                .as_u64()
                .expect("a count"),
        "the fixture no longer exercises a sparse index: {scene}"
    );

    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": index }),
    );
    assert_eq!(point["index"], json!(index), "{point}");
    assert_eq!(point["id"], selected["id"], "{point}");
    // And the selection tool takes it too, which is the other half of the
    // handle being a handle.
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": index }),
    );
}

/// `get_scene`'s observation count is the version's and not the file's: a
/// commit that replaces a thirty-observation track with a two-observation one
/// moves it by twenty-eight.
#[test]
fn the_observation_count_follows_the_version() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let observations = |state: &mut AppState, viewer: &mut Viewer3D| -> u64 {
        call(state, viewer, "get_scene", json!({}))["scene"][0]["counts"]["observations"]
            .as_u64()
            .expect("a count")
    };
    let before = observations(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let rows = track["observations"].as_array().expect("the rows").len();
    assert!(rows > 2, "the fixture needs a row to turn out: {track}");

    // One sighting refused, so the track commits with one fewer than the point
    // it replaces.
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": rows - 1, "verdict": "out" }),
    );
    call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );

    assert_eq!(
        observations(&mut state, &mut viewer),
        before - 1,
        "the count did not follow the commit"
    );
    // And an undo takes the observation back with the version.
    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(observations(&mut state, &mut viewer), before);
}

/// The stage change's sentence is written once. The report used to carry the
/// stage phrase twice -- the viewer's, then core's report printed whole behind
/// it -- which is the one thing an agent reads to find out what happened.
#[test]
fn the_stage_report_states_the_stage_once() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);

    let staged = worked(
        &mut state,
        &mut viewer,
        "set_bench_track_stage",
        json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
    );
    let report = staged["report"].as_str().expect("a report");
    assert_eq!(
        report.matches("stage").count(),
        1,
        "the stage is stated twice: {report}"
    );
    assert!(
        report.starts_with(&format!("Set {item} to the cluster stage")),
        "{report}"
    );
    // The Action Log row is that sentence, up to the serials.
    let last = rows(&state).pop().expect("one row per step");
    assert!(last.2.starts_with(report), "{last:?} against {report}");
}

/// The steps that read photographs decode **on the worker**: a node whose
/// photographs are neither decoded nor readable starts the task all the same,
/// and the reply is the handle the frame hands back once the window has passed.
#[test]
fn a_slow_fit_answers_with_a_handle_naming_it() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    // Nothing decoded: the fit has every photograph to read, which is the work
    // that must not happen before the deferral.
    state.photographs.clear();

    let arguments = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("fit_bench_track", Some(&arguments)).expect("a valid call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("a fit must defer to a worker"),
    };
    assert_eq!(pending.operation_name, "Fit track");

    // Past the window with the operation still the one running -- nothing has
    // drained its reports -- which is the handle case.
    let operation_id = pending.operation_id;
    let past = super::super::BackgroundReply {
        started: std::time::Instant::now() - super::super::REPLY_DIRECTLY_WITHIN,
        ..pending
    };
    let reply = match super::super::edit::background_reply(&state, &past).expect("past the window")
    {
        Ok(ToolOutput::Json(value)) => value,
        _ => panic!("a handle is JSON"),
    };
    assert_eq!(reply["running"], json!(true), "{reply}");
    assert_eq!(reply["operation"], json!("Fit track"), "{reply}");
    assert_eq!(reply["operation_id"], json!(operation_id), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");

    // The photographs the worker could not read are its refusal and not the
    // gesture's: the step began either way.
    state.finish_background_task();
    match super::super::edit::background_reply(&state, &past).expect("the operation finished") {
        Err(e) => assert!(e.0.contains(&format!("Cannot fit {item}")), "{e}"),
        Ok(_) => panic!("the fixture's photographs are not on disk"),
    }
}

// ── Live evaluation ─────────────────────────────────────────────────────

/// There is no call that evaluates a track: a track is evaluated as soon as it
/// is on the bench and again after every change, and both reads say whether the
/// numbers they return are the evaluation of the track as it stands.
#[test]
fn the_reads_say_whether_the_numbers_are_current_and_the_radius_is_an_input() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    assert!(
        tools::parse("evaluate_bench_track", None).is_err(),
        "the evaluate tool is still on the wire"
    );

    let read = |state: &mut AppState, viewer: &mut Viewer3D| {
        call(
            state,
            viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a", "track": item }),
        )["evaluation"]
            .clone()
    };
    let evaluation = read(&mut state, &mut viewer);
    assert_eq!(evaluation["state"], json!("evaluating"), "{evaluation}");
    assert!(evaluation["reason"].is_null(), "{evaluation}");

    state.settle_bench_evaluation();
    assert_eq!(read(&mut state, &mut viewer)["state"], json!("current"));
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        bench["items"][0]["evaluation"]["state"],
        json!("current"),
        "{bench}"
    );

    // The search radius is the track's shift bar: moving it is one version,
    // and the track reads again.
    assert!(
        tools::parse("set_bench_search_px", None).is_err(),
        "the search radius tool is still on the wire"
    );
    let before = version_count(&state);
    call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "track": item, "max_shift_px": 7.5 }),
    );
    assert_eq!(version_count(&state), before + 1);
    let evaluation = read(&mut state, &mut viewer);
    assert_eq!(evaluation["state"], json!("evaluating"), "{evaluation}");
    state.settle_bench_evaluation();
    assert_eq!(read(&mut state, &mut viewer)["state"], json!("current"));
}

// ── Create Track Here ───────────────────────────────────────────────────

/// `create_track_at_pixel` over the plane capture Create Track Here's own tests
/// use, as `run_a`: a grid of points on a textured plane, every photograph
/// cached.
fn plane_benchable() -> (AppState, Viewer3D) {
    let (mut state, viewer) = benchable_with(crate::bench::track_at_pixel::tests::plane_recon());
    let id = state.scene[0].id;
    crate::bench::track_at_pixel::tests::cache_photographs(&mut state, id);
    (state, viewer)
}

/// The tool is the panel's step: at a pixel on the textured plane it answers
/// with the version the commit pushed, the item the track stays on the bench
/// as, the member that built it and the point it wrote, which `get_point`
/// takes back.
#[test]
fn create_track_at_pixel_commits_a_point_and_names_it() {
    let (mut state, mut viewer) = plane_benchable();
    let pixel = crate::bench::track_at_pixel::tests::textured_pixel();
    let before = state.scene[0].edited().point_count();
    let reply = worked(
        &mut state,
        &mut viewer,
        "create_track_at_pixel",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": pixel }),
    );
    assert_eq!(reply["changed"], json!(true), "{reply}");
    assert_eq!(reply["member"], json!("transfer"), "{reply}");
    let item = reply["item"].as_str().expect("an item");
    assert!(item.starts_with("image_0@"), "{reply}");
    assert!(
        reply["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Created {item} at (")),
        "{reply}"
    );
    assert!(
        reply["label"]
            .as_str()
            .expect("the version's label")
            .starts_with("Committed"),
        "the cursor is on the commit's version: {reply}"
    );
    let index = reply["point"]["index"].as_u64().expect("an index");
    assert_eq!(reply["point"]["replaced"], Value::Null, "{reply}");
    assert_eq!(state.scene[0].edited().point_count(), before + 1);

    let id = reply["point"]["id"].as_str().expect("an id");
    let point = call(&mut state, &mut viewer, "get_point", json!({ "point": id }));
    assert_eq!(point["index"].as_u64(), Some(index), "{point}");

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["focused_item"], json!(item), "{bench}");
}

/// A label named in the call is the label the built track goes on the bench
/// under, and the commit's reply names it.
#[test]
fn create_track_at_pixel_puts_the_track_on_the_bench_under_the_label_it_is_given() {
    let (mut state, mut viewer) = plane_benchable();
    let pixel = crate::bench::track_at_pixel::tests::textured_pixel();
    let reply = worked(
        &mut state,
        &mut viewer,
        "create_track_at_pixel",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": pixel,
            "label": "plane corner",
        }),
    );
    assert_eq!(reply["item"], json!("plane corner"), "{reply}");
    assert_eq!(reply["changed"], json!(true), "{reply}");
}

/// Every member refusing is a tool error carrying the row's sentence and then
/// each member's stage and reason, in the order tried, with nothing pushed; a
/// pixel off the photograph is refused in the call, before any worker.
#[test]
fn create_track_at_pixel_refuses_with_every_members_stage() {
    let (mut state, mut viewer) = plane_benchable();
    let versions = state.scene[0].history.versions().len();
    let pixel = crate::bench::track_at_pixel::tests::lonely_pixel();
    let map = json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": pixel })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("create_track_at_pixel", Some(&map)).expect("parses");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        _ => panic!("create_track_at_pixel must defer to a worker"),
    };
    state.finish_background_task();
    let error = match super::super::edit::background_reply(&state, &pending)
        .expect("the operation finished")
    {
        Err(error) => error.to_string(),
        Ok(_) => panic!("expected every member to refuse"),
    };
    assert!(
        error.starts_with("Cannot create a track at (2.0, 125.0) in image_0.jpg: every member refused; the last, constellation, at constellation: "),
        "{error}"
    );
    assert!(error.contains("Build Index Files"), "{error}");
    let lines: Vec<&str> = error.lines().filter(|l| l.starts_with("- ")).collect();
    assert_eq!(lines.len(), 4, "{error}");
    for (line, member) in lines
        .iter()
        .zip(["clusters", "transfer", "sweep", "constellation"])
    {
        assert!(
            line.starts_with(&format!("- {member} refused at ")),
            "{error}"
        );
    }
    assert_eq!(state.scene[0].history.versions().len(), versions);

    let off = refused_call(
        &mut state,
        &mut viewer,
        "create_track_at_pixel",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": [-4.0, 10.0] }),
    );
    assert!(
        off.to_string().contains("not on the 128x128 photograph"),
        "{off}"
    );
    assert!(state.background_task().is_none());
}

// ── Find Nearby Tracks ──────────────────────────────────────────────────

/// The labels of the items on `run_a`'s bench, in order.
fn bench_items(state: &mut AppState, viewer: &mut Viewer3D) -> Vec<String> {
    let bench = call(
        state,
        viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    bench["items"]
        .as_array()
        .expect("an array")
        .iter()
        .map(|item| item["item"].as_str().expect("a label").to_string())
        .collect()
}

/// How many versions `get_history` lists for `run_a`.
fn history_len(state: &mut AppState, viewer: &mut Viewer3D) -> usize {
    let history = call(
        state,
        viewer,
        "get_history",
        json!({ "reconstruction_label": "run_a" }),
    );
    history["versions"]
        .as_array()
        .expect("a version list")
        .len()
}

/// `find_nearby_tracks` over the plane capture at the held-out point's pixel:
/// the eight points around it land on the bench under their labels as one
/// version, the reply names each with its point, source, layer, rank and
/// pixel, and one undo takes the bench back.
#[test]
fn find_nearby_tracks_lands_the_existing_points_under_their_labels() {
    let (mut state, mut viewer) = plane_benchable();
    let pixel = crate::bench::track_at_pixel::tests::textured_pixel();
    let versions = history_len(&mut state, &mut viewer);
    let items_before = bench_items(&mut state, &mut viewer);

    let reply = worked(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": pixel }),
    );
    assert_eq!(reply["changed"], json!(true), "{reply}");
    let group = reply["group_label"].as_str().expect("a group label");
    assert!(group.starts_with("image_0@"), "{reply}");
    assert_eq!(reply["layers"].as_array().map(Vec::len), Some(1), "{reply}");
    assert_eq!(reply["layers"][0]["rank"], json!(1), "{reply}");
    let tracks = reply["tracks"].as_array().expect("tracks");
    assert_eq!(tracks.len(), 8, "{reply}");
    for (track, letter) in tracks.iter().zip('a'..) {
        let index = track["point"]["index"].as_u64().expect("an existing point");
        assert_eq!(
            track["label"],
            json!(format!("{group} 1{letter} pt {index}")),
            "{track}"
        );
        assert_eq!(track["item"], track["label"], "{track}");
        assert_eq!(track["existing"], json!(true), "{track}");
        assert_eq!(track["committed"], json!(false), "{track}");
        assert_eq!(track["source"], json!("points"), "{track}");
        assert_eq!(track["layer"], json!(0), "{track}");
        assert_eq!(track["rank"], json!(1), "{track}");
        assert!(track["point"]["id"].is_string(), "{track}");
        assert_eq!(track["pixel"].as_array().map(Vec::len), Some(2), "{track}");
    }
    assert_eq!(reply["sources"][0]["source"], json!("points"), "{reply}");
    assert_eq!(reply["stopped_after"], json!("points"), "{reply}");
    assert_eq!(reply["duplicates"], json!(0), "{reply}");
    assert!(
        reply["report"]
            .as_str()
            .expect("a report")
            .starts_with("Found 8 nearby tracks in 1 layer at ("),
        "{reply}"
    );

    assert_eq!(history_len(&mut state, &mut viewer), versions + 1);
    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["focused_item"], tracks[0]["item"], "{bench}");

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench_items(&mut state, &mut viewer), items_before);
}

/// On the far plane the far-field sweep's track is committed as a new point
/// in the one version, which `get_point` takes back by its id; with `commit`
/// false the same find commits nothing and puts the track on the bench under
/// the caller's group label.
#[test]
fn find_nearby_tracks_commits_the_new_tracks_unless_told_not_to() {
    use crate::bench::track_at_pixel::tests::{cache_far_photographs, far_plane_recon, FAR_PIXEL};
    let far = || {
        let (mut state, viewer) = benchable_with(far_plane_recon());
        let id = state.scene[0].id;
        cache_far_photographs(&mut state, id);
        (state, viewer)
    };

    let (mut state, mut viewer) = far();
    let before = state.scene[0].edited().point_count();
    let versions = history_len(&mut state, &mut viewer);
    let reply = worked(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": FAR_PIXEL }),
    );
    let tracks = reply["tracks"].as_array().expect("tracks");
    let committed: Vec<&Value> = tracks
        .iter()
        .filter(|t| t["committed"] == json!(true))
        .collect();
    assert!(!committed.is_empty(), "{reply}");
    assert_eq!(committed[0]["source"], json!("far_field"), "{reply}");
    assert_eq!(reply["far_field"]["trigger"], json!("no_layer"), "{reply}");
    assert_eq!(
        state.scene[0].edited().point_count(),
        before + committed.len()
    );
    assert_eq!(history_len(&mut state, &mut viewer), versions + 1);
    let id = committed[0]["point"]["id"].as_str().expect("an id");
    let point = call(&mut state, &mut viewer, "get_point", json!({ "point": id }));
    assert_eq!(point["index"], committed[0]["point"]["index"], "{point}");

    let (mut state, mut viewer) = far();
    let reply = worked(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 0,
            "pixel": FAR_PIXEL,
            "commit": false,
            "label": "sky",
        }),
    );
    assert_eq!(reply["group_label"], json!("sky"), "{reply}");
    assert_eq!(reply["tracks"][0]["item"], json!("sky 1a"), "{reply}");
    assert_eq!(reply["tracks"][0]["committed"], json!(false), "{reply}");
    assert_eq!(reply["tracks"][0]["point"], Value::Null, "{reply}");
    assert_eq!(state.scene[0].edited().point_count(), before);
    assert_eq!(bench_items(&mut state, &mut viewer), ["sky 1a"]);
}

/// A find with nothing usable answers `changed: false` and no tracks, and
/// pushes no version; an image with no pose and a pixel off the photograph
/// are refused in the call, with no task.
#[test]
fn find_nearby_tracks_answers_an_empty_find_and_refuses_what_cannot_run() {
    let (mut state, mut viewer) = plane_benchable();
    let versions = history_len(&mut state, &mut viewer);
    let pixel = crate::bench::track_at_pixel::tests::lonely_pixel();
    let reply = worked(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": pixel }),
    );
    assert_eq!(reply["changed"], json!(false), "{reply}");
    assert_eq!(reply["tracks"], json!([]), "{reply}");
    assert!(
        reply["report"]
            .as_str()
            .expect("a report")
            .starts_with("Found no nearby tracks at (2.0, 125.0) in image_0.jpg"),
        "{reply}"
    );
    assert_eq!(history_len(&mut state, &mut viewer), versions);

    state.scene[0].recon_mut().image_table.images[1]
        .translation_xyz
        .x = f64::NAN;
    let unposed = refused_call(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({ "reconstruction_label": "run_a", "camera_image": 1, "pixel": [64.0, 64.0] }),
    );
    assert_eq!(
        unposed.to_string(),
        crate::bench::track_at_pixel::NOT_POSED,
        "{unposed}"
    );
    let off = refused_call(
        &mut state,
        &mut viewer,
        "find_nearby_tracks",
        json!({ "reconstruction_label": "run_a", "camera_image": 0, "pixel": [-4.0, 10.0] }),
    );
    assert!(
        off.to_string().contains("not on the 128x128 photograph"),
        "{off}"
    );
    assert!(state.background_task().is_none());
}

// ── The selected observations ───────────────────────────────────────────

/// The selected observations of the named track, as `get_bench_track` reports
/// them.
#[track_caller]
fn selected(state: &mut AppState, viewer: &mut Viewer3D, track: &str) -> Value {
    call(
        state,
        viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": track }),
    )["selected_observations"]
        .clone()
}

/// A selection is replaced whole, sorted and without repeats, reads back
/// through `get_bench_track`, writes a `Selection` row and pushes no version;
/// an empty list clears it.
#[test]
fn select_bench_observations_replaces_the_selection_and_reads_back() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([]));
    let serial = state.scene[0].history.current_version().serial;

    let reply = call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "observations": [2, 0, 2] }),
    );
    assert_eq!(reply["item"], json!(item), "{reply}");
    assert_eq!(reply["selected_observations"], json!([0, 2]), "{reply}");
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([0, 2]));
    assert_eq!(state.scene[0].history.current_version().serial, serial);
    let rows: Vec<&str> = state
        .action_log
        .entries()
        .filter(|entry| entry.kind == Kind::Selection)
        .map(|entry| entry.text.as_str())
        .collect();
    assert_eq!(rows, [format!("Selected observations 0, 2 of {item}")]);

    call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "observations": [] }),
    );
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([]));
}

/// A step on the focused item keeps the selection, an undo clears it, and a
/// rename keeps it under the new label.
#[test]
fn the_selection_survives_a_step_and_is_cleared_by_an_undo() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let select = |state: &mut AppState, viewer: &mut Viewer3D| {
        call(
            state,
            viewer,
            "select_bench_observations",
            json!({ "reconstruction_label": "run_a", "observations": [1] }),
        );
    };
    select(&mut state, &mut viewer);

    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 0, "verdict": "out" }),
    );
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([1]));

    call(
        &mut state,
        &mut viewer,
        "rename_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item, "label": "nose" }),
    );
    assert_eq!(selected(&mut state, &mut viewer, "nose"), json!([1]));

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([]));
}

/// Another item focused takes the selection with it, and coming back to the
/// first item does not bring it back.
#[test]
fn a_change_of_focused_item_clears_the_selection() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "observations": [0, 1] }),
    );
    let copy = call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    )["item"]
        .as_str()
        .expect("the copy")
        .to_string();
    assert_eq!(selected(&mut state, &mut viewer, &copy), json!([]));
    call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([]));
}

/// A split renumbers the observations left behind, so it clears the selection
/// it was made from.
#[test]
fn a_split_clears_the_selection() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "observations": [0] }),
    );
    call(
        &mut state,
        &mut viewer,
        "split_bench_track",
        json!({ "reconstruction_label": "run_a", "observations": [0] }),
    );
    assert!(state.bench_rows.is_none());
    call(
        &mut state,
        &mut viewer,
        "focus_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
    assert_eq!(selected(&mut state, &mut viewer, &item), json!([]));
}

/// Only the focused item has selected observations, and an index past the end
/// of the list names nothing; both are refused and change nothing.
#[test]
fn a_selection_on_another_track_or_past_the_end_is_refused() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let error = refused_call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "observations": [0, 9] }),
    );
    assert!(error.0.contains("there is no observation 9"), "{error}");

    call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    let error = refused_call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "track": item, "observations": [0] }),
    );
    assert!(error.0.contains("is not the focused item"), "{error}");
    assert!(state.bench_rows.is_none());
}

// ── set_view's bench_observation ────────────────────────────────────────

/// The 3D view looks through the observation's own camera image, as the Track
/// View row's double-click does.
#[test]
fn set_view_looks_through_a_bench_observation() {
    let (mut state, mut viewer) = benchable();
    on_the_bench(&mut state, &mut viewer);
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    let image = track["observations"][1]["camera_image"].clone();

    let out = call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "bench_observation": { "reconstruction_label": "run_a", "observation": 1 } }),
    );
    assert_eq!(out["view"]["looking_through"]["camera_image_index"], image);

    let error = refused_call(
        &mut state,
        &mut viewer,
        "set_view",
        json!({ "bench_observation": { "reconstruction_label": "run_a", "observation": 40 } }),
    );
    assert!(error.0.contains("there is no observation 40"), "{error}");
}

/// An unpin names one observation, a list of them, or `"all"`, and each is one
/// version; `in` and `out` name one observation only.
#[test]
fn an_unpin_names_a_list_of_observations_or_all_of_them() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let rows = |state: &mut AppState, viewer: &mut Viewer3D| {
        let track = call(
            state,
            viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        );
        track["observations"]
            .as_array()
            .expect("the observations")
            .iter()
            .map(|row| row["pinned"] == json!(true))
            .collect::<Vec<_>>()
    };
    let pinned = rows(&mut state, &mut viewer);
    assert!(pinned.len() > 2, "{pinned:?}");
    assert!(pinned.iter().all(|&p| p), "a point's rows arrive pinned");

    let before = version_count(&state);
    let listed = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": [0, 1],
            "verdict": "unpin",
        }),
    );
    assert_eq!(listed["item"], json!(item), "{listed}");
    assert_eq!(listed["changed"], json!(true), "{listed}");
    let named = listed["observations"]
        .as_array()
        .expect("one entry per row");
    assert_eq!(named.len(), 2, "{listed}");
    assert_eq!(named[0]["observation"], json!(0), "{listed}");
    assert_eq!(named[1]["pinned"], json!(false), "{listed}");
    assert!(
        listed["report"]
            .as_str()
            .expect("a sentence")
            .starts_with("Handed 2 verdicts back to the thresholds"),
        "{listed}"
    );
    assert_eq!(version_count(&state), before + 1);
    let now = rows(&mut state, &mut viewer);
    assert!(!now[0] && !now[1] && now[2..].iter().all(|&p| p), "{now:?}");

    let all = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": "all",
            "verdict": "unpin",
        }),
    );
    assert_eq!(all["changed"], json!(true), "{all}");
    assert_eq!(version_count(&state), before + 2);
    // The sentence says what moved and then gives the track's totals, so the
    // two numbers at its end are the verdicts the track holds now rather than
    // the changes.
    let track = state.scene[0]
        .history
        .current_bench()
        .track(&item)
        .expect("the item is on the bench")
        .clone();
    let total_in = track
        .observations
        .iter()
        .filter(|o| o.verdict == sfmtool_core::bench::Verdict::In)
        .count();
    let report = all["label"].as_str().expect("a sentence");
    assert!(
        report.ends_with(&format!(
            ", leaving {total_in} in, {} out",
            track.observations.len() - total_in
        )),
        "{all}"
    );
    assert!(
        report.contains("turned") || report.contains("none moved"),
        "{all}"
    );
    assert!(rows(&mut state, &mut viewer).iter().all(|&p| !p));

    // Nothing is pinned now, so another unpin of all is no version.
    let again = call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": "all",
            "verdict": "unpin",
        }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(version_count(&state), before + 2);

    // in and out rule on one observation at a time.
    let several = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": [0, 1],
            "verdict": "in",
        }),
    );
    assert!(several.0.contains("only to pin or unpin them"), "{several}");
    let both = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observation": 0,
            "observations": "all",
            "verdict": "unpin",
        }),
    );
    assert!(both.0.contains("not both"), "{both}");
    let past = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": [0, 99],
            "verdict": "unpin",
        }),
    );
    assert!(past.0.contains("no observation 99"), "{past}");
}

/// A pin names one observation, a list of them, or `"all"`, and pins each
/// named row at the verdict it has, as one version; a pin of rows all pinned
/// already is no version.
#[test]
fn a_pin_names_a_list_of_observations_or_all_of_them_and_keeps_their_verdicts() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let rows = |state: &mut AppState, viewer: &mut Viewer3D| {
        let track = call(
            state,
            viewer,
            "get_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        );
        track["observations"]
            .as_array()
            .expect("the observations")
            .iter()
            .map(|row| (row["verdict"].clone(), row["pinned"] == json!(true)))
            .collect::<Vec<_>>()
    };
    let verdict = |state: &mut AppState, viewer: &mut Viewer3D, body: Value| {
        let mut args = json!({ "reconstruction_label": "run_a" });
        for (key, value) in body.as_object().expect("an object") {
            args[key] = value.clone();
        }
        call(state, viewer, "set_bench_track_verdict", args)
    };
    verdict(
        &mut state,
        &mut viewer,
        json!({ "observations": "all", "verdict": "unpin" }),
    );
    let unpinned = rows(&mut state, &mut viewer);
    assert!(unpinned.iter().all(|(_, p)| !p), "{unpinned:?}");

    let before = version_count(&state);
    let one = verdict(
        &mut state,
        &mut viewer,
        json!({ "observation": 0, "verdict": "pin" }),
    );
    assert_eq!(one["item"], json!(item), "{one}");
    assert_eq!(one["changed"], json!(true), "{one}");
    assert_eq!(one["observation"], json!(0), "{one}");
    assert_eq!(one["pinned"], json!(true), "{one}");
    assert_eq!(one["verdict"], unpinned[0].0, "{one}");
    assert_eq!(version_count(&state), before + 1);

    let listed = verdict(
        &mut state,
        &mut viewer,
        json!({ "observations": [0, 1], "verdict": "pin" }),
    );
    assert_eq!(listed["changed"], json!(true), "{listed}");
    let named = listed["observations"]
        .as_array()
        .expect("one entry per row");
    assert_eq!(named.len(), 2, "{listed}");
    assert_eq!(named[1]["pinned"], json!(true), "{listed}");
    assert!(
        listed["report"]
            .as_str()
            .expect("a sentence")
            .starts_with("Pinned "),
        "{listed}"
    );
    assert_eq!(version_count(&state), before + 2);

    let all = verdict(
        &mut state,
        &mut viewer,
        json!({ "observations": "all", "verdict": "pin" }),
    );
    assert_eq!(all["changed"], json!(true), "{all}");
    assert_eq!(version_count(&state), before + 3);
    let now = rows(&mut state, &mut viewer);
    assert!(now.iter().all(|(_, p)| *p), "{now:?}");
    assert_eq!(
        now.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>(),
        unpinned.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>(),
        "every verdict stands as it was"
    );

    // Everything is pinned now, so another pin of all is no version.
    let again = verdict(
        &mut state,
        &mut viewer,
        json!({ "observations": "all", "verdict": "pin" }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(version_count(&state), before + 3);

    let past = refused_call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({
            "reconstruction_label": "run_a",
            "observations": [0, 99],
            "verdict": "pin",
        }),
    );
    assert!(past.0.contains("no observation 99"), "{past}");
}

// ── The viewed track on the wire ────────────────────────────────────────

/// `get_point` on the point Track View shows with Edit clear carries the viewed
/// track's evaluation, `evaluating` and then `current`, with the read-only bars
/// and each row's verdict by them; any other point carries none.
#[test]
fn get_point_reports_the_viewed_points_evaluation_and_no_other_points() {
    let (mut state, mut viewer) = benchable();
    let id = state.scene[0].id;
    state.select_point(PointRef::new(id, BENCH_POINT as usize));
    // What the dock does when it draws Track View.
    state.refresh_viewed_track();

    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": BENCH_POINT }),
    );
    let evaluation = &point["evaluation"];
    assert_eq!(evaluation["state"], json!("evaluating"), "{point}");
    assert_eq!(evaluation["item"], point["id"], "{point}");
    assert_eq!(
        evaluation["thresholds"]["min_zncc"],
        json!(sfmtool_core::bench::Thresholds::default().min_zncc)
    );
    let rows = evaluation["observations"].as_array().expect("rows");
    assert_eq!(rows.len(), point["track"].as_array().expect("track").len());
    assert!(rows.iter().all(|row| row["verdict"] == json!("in")
        && row["pinned"] == json!(true)
        && row.get("verdict_by_bars").is_some()));

    state.settle_bench_evaluation();
    state.set_viewed_thresholds(sfmtool_core::bench::Thresholds {
        min_zncc: 1.1,
        ..Default::default()
    });
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": BENCH_POINT }),
    );
    let evaluation = &point["evaluation"];
    assert_eq!(evaluation["state"], json!("current"), "{point}");
    assert_eq!(evaluation["thresholds"]["min_zncc"], json!(1.1));
    let rows = evaluation["observations"].as_array().expect("rows");
    assert!(
        rows.iter()
            .any(|row| row["verdict_by_bars"] == json!("out")),
        "a bar no reading reaches turns a measured row out: {point}"
    );
    assert!(rows.iter().all(|row| row["verdict"] == json!("in")));
    // The row the reference-view rule picked, which no bar moves.
    let reference = evaluation["reference_observation"]
        .as_u64()
        .expect("a reference row") as usize;
    assert_eq!(
        rows[reference]["track"]["reference_view"]["is_reference"],
        json!(true),
        "{point}"
    );

    let other = (0..state.scene[0].edited().point_count() as u32)
        .find(|&p| p != BENCH_POINT && state.scene[0].edited().point(p).is_some())
        .expect("another point");
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": other }),
    );
    assert!(point.get("evaluation").is_none(), "{point}");

    // A select_point earlier in the same batch moves the viewed point with it.
    call(
        &mut state,
        &mut viewer,
        "select_point",
        json!({ "point": other }),
    );
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": other }),
    );
    assert!(point["evaluation"].is_object(), "{point}");
}

/// `create_bench_track` naming the viewed point carries the moved read-only
/// bars onto the new track, and the reply's label and report name them; the
/// same call on another point starts at the defaults.
#[test]
fn create_bench_track_carries_the_read_only_bars_for_the_viewed_point_only() {
    let (mut state, mut viewer) = benchable();
    let id = state.scene[0].id;
    let other = (0..state.scene[0].edited().point_count() as u32)
        .find(|&p| p != BENCH_POINT && state.scene[0].edited().point(p).is_some())
        .expect("another point");
    state.select_point(PointRef::new(id, BENCH_POINT as usize));
    let bars = sfmtool_core::bench::Thresholds {
        max_shift_px: 4.0,
        ..Default::default()
    };
    state.set_viewed_thresholds(bars.clone());

    let elsewhere = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": other }),
    );
    assert!(
        !elsewhere["label"]
            .as_str()
            .expect("a label")
            .contains("with"),
        "{elsewhere}"
    );
    let item = elsewhere["item"].as_str().expect("an item");
    assert_eq!(
        state.bench_track(id, item).expect("on").thresholds,
        sfmtool_core::bench::Thresholds::default()
    );

    // The put focused that item; select the viewed point again.
    state.select_point(PointRef::new(id, BENCH_POINT as usize));
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": BENCH_POINT }),
    );
    assert!(
        made["label"]
            .as_str()
            .expect("a label")
            .ends_with(", with max shift 4.0 px"),
        "{made}"
    );
    assert!(
        made["report"]
            .as_str()
            .expect("a report")
            .contains(", with max shift 4.0 px"),
        "{made}"
    );
    let item = made["item"].as_str().expect("an item");
    assert_eq!(state.bench_track(id, item).expect("on").thresholds, bars);
}

// ── A view-only bench ───────────────────────────────────────────────────

/// A `sift_files` node's bench is view-only: a point goes on the bench, and
/// every tool that would edit it is refused up front with the sentence that
/// names Convert to Embedded Patches, pushing nothing. After the conversion,
/// putting the same point on the bench again rebuilds the frame-less item
/// with a patch frame under its own label, and the item takes edits.
#[test]
fn a_sift_files_bench_is_view_only_until_converted_and_rebenched() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state.select_recon(id);
    state.window = Some(FakeWindow::default().info());
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 0 }),
    );
    let item = made["item"].as_str().expect("an item").to_string();
    let frameless = |state: &AppState| match &state.bench_track(id, &item).expect("on").stage {
        sfmtool_core::bench::Stage::Track(payload) => payload.placement.is_none(),
        sfmtool_core::bench::Stage::Cluster(_) => panic!("a point goes on at the track stage"),
    };
    assert!(frameless(&state));
    let versions = version_count(&state);

    let refusals = [
        (
            "fit_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        ),
        (
            "fit_bench_track_normal",
            json!({ "reconstruction_label": "run_a", "method": "photometric" }),
        ),
        (
            "set_bench_track_stage",
            json!({ "reconstruction_label": "run_a", "stage": "cluster" }),
        ),
        // The stage the track is at already: refused rather than "no effect".
        (
            "set_bench_track_stage",
            json!({ "reconstruction_label": "run_a", "stage": "track" }),
        ),
        (
            "set_bench_track_verdict",
            json!({ "reconstruction_label": "run_a", "observation": 0, "verdict": "out" }),
        ),
        (
            "apply_bench_track_thresholds",
            json!({ "reconstruction_label": "run_a", "min_zncc": 0.8 }),
        ),
        (
            "translate_bench_patch",
            json!({ "reconstruction_label": "run_a", "by": [0.1, 0.0, 0.0] }),
        ),
        (
            "resize_bench_patch",
            json!({ "reconstruction_label": "run_a", "half_length": 0.5 }),
        ),
        (
            "sight_bench_observation",
            json!({ "reconstruction_label": "run_a", "observation": 0, "pixel": [10.0, 10.0] }),
        ),
        (
            "tilt_bench_patch",
            json!({ "reconstruction_label": "run_a", "normal": [0.0, 0.0, 1.0] }),
        ),
        (
            "spin_bench_patch",
            json!({ "reconstruction_label": "run_a", "degrees": 10.0 }),
        ),
        (
            "add_bench_track_observation",
            json!({ "reconstruction_label": "run_a", "camera_image": 3, "pixel": [10.0, 10.0] }),
        ),
        (
            "split_bench_track",
            json!({ "reconstruction_label": "run_a", "observations": [1] }),
        ),
        (
            "duplicate_bench_item",
            json!({ "reconstruction_label": "run_a" }),
        ),
        (
            "commit_bench_track",
            json!({ "reconstruction_label": "run_a" }),
        ),
        (
            "search_bench_track_geometry",
            json!({ "reconstruction_label": "run_a", "observation": 0 }),
        ),
        (
            "create_bench_cluster",
            json!({
                "reconstruction_label": "run_a",
                "camera_image": 0,
                "pixel": [10.0, 10.0],
                "radius_px": 7.5,
            }),
        ),
    ];
    for (tool, arguments) in refusals {
        let error = refused_call(&mut state, &mut viewer, tool, arguments.clone());
        assert!(
            error.0.starts_with("Bench editing needs embedded patches")
                && error.0.contains("Convert to Embedded Patches")
                && error.0.contains("convert_to_embedded_patches"),
            "{tool} {arguments}: {error}"
        );
    }
    assert_eq!(version_count(&state), versions, "a refusal pushes nothing");

    // The panel's side asks the same question and gets the same sentence.
    let why = state.bench_edit_refusal(id).expect("view-only");
    assert!(
        why.starts_with("Bench editing needs embedded patches"),
        "{why}"
    );
    assert_eq!(
        state.set_bench_verdict(id, &item, 0, sfmtool_core::bench::Verdict::Out),
        Err(why.clone())
    );

    // The remedy the sentence names is one that runs.
    assert!(state.convert_to_embedded_patches_refusal(id).is_none());
    state
        .start_convert_to_embedded_patches(id)
        .expect("well posed");
    state.finish_background_task();
    assert!(state.bench_edit_refusal(id).is_none());
    assert!(
        frameless(&state),
        "the conversion leaves the bench as it was"
    );

    // Putting the point on the bench again rebuilds the item, in one version.
    let versions = version_count(&state);
    let again = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 0 }),
    );
    assert_eq!(again["item"], json!(item), "{again}");
    assert_eq!(version_count(&state), versions + 1);
    assert!(!frameless(&state), "the rebuilt item carries a patch frame");
    // A second put of the point only focuses the rebuilt item.
    call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 0 }),
    );
    assert_eq!(version_count(&state), versions + 1);

    // And it takes an edit.
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 0, "verdict": "out" }),
    );
    assert_eq!(version_count(&state), versions + 2);
}

/// The bench is editable from the conversion on, so an item put on it before
/// the conversion can take edits before its point is put on the bench again.
/// The re-bench gives the item the point's frame and keeps those edits: the
/// verdict, the thresholds and a sighting all survive it. A duplicate made
/// before the re-bench would never get a frame, so it is refused until then.
#[test]
fn edits_made_between_the_conversion_and_the_rebench_survive_it() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state.select_recon(id);
    state.window = Some(FakeWindow::default().info());
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 0 }),
    );
    let item = made["item"].as_str().expect("an item").to_string();
    state
        .start_convert_to_embedded_patches(id)
        .expect("well posed");
    state.finish_background_task();
    assert!(state.bench_edit_refusal(id).is_none());

    // A copy now would drop the origin and never get a frame.
    let versions = version_count(&state);
    let error = refused_call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        error.0.contains("has no patch frame yet") && error.0.contains("Put point 0"),
        "{error}"
    );
    assert_eq!(version_count(&state), versions, "a refusal pushes nothing");

    // Edits the frame-less item takes before the re-bench.
    call(
        &mut state,
        &mut viewer,
        "set_bench_track_verdict",
        json!({ "reconstruction_label": "run_a", "observation": 0, "verdict": "out" }),
    );
    call(
        &mut state,
        &mut viewer,
        "apply_bench_track_thresholds",
        json!({ "reconstruction_label": "run_a", "min_zncc": 0.8 }),
    );
    call(
        &mut state,
        &mut viewer,
        "sight_bench_observation",
        json!({ "reconstruction_label": "run_a", "observation": 1, "pixel": [12.0, 34.0] }),
    );
    let edited = (**state.bench_track(id, &item).expect("on")).clone();
    let versions = version_count(&state);

    call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 0 }),
    );
    assert_eq!(version_count(&state), versions + 1);
    let rebuilt = (**state.bench_track(id, &item).expect("on")).clone();
    let sfmtool_core::bench::Stage::Track(payload) = &rebuilt.stage else {
        panic!("still a track");
    };
    assert!(
        payload.placement.is_some(),
        "the item has the point's frame"
    );
    assert_eq!(rebuilt.observations.len(), edited.observations.len());
    assert_eq!(
        rebuilt.observations[0].verdict,
        sfmtool_core::bench::Verdict::Out
    );
    assert!(rebuilt.observations[0].pinned);
    assert_eq!(rebuilt.thresholds, edited.thresholds);
    assert_eq!(
        rebuilt.observations[1], edited.observations[1],
        "the sighted observation is kept as it was"
    );

    // With the frame in place, a copy keeps it.
    call(
        &mut state,
        &mut viewer,
        "duplicate_bench_item",
        json!({ "reconstruction_label": "run_a", "item": item }),
    );
}

/// An item nobody edited comes out of the re-bench exactly as a fresh put of
/// the converted point would build it, under its own label and origin.
#[test]
fn an_unedited_item_rebuilt_on_the_rebench_is_the_fresh_track() {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let (mut state, id) = crate::state::edits::tests::convertible_state(dir.path());
    state.select_recon(id);
    state.window = Some(FakeWindow::default().info());
    let mut viewer = Viewer3D::new();
    viewer.panel_size = [1280, 720];

    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    let item = made["item"].as_str().expect("an item").to_string();
    state
        .start_convert_to_embedded_patches(id)
        .expect("well posed");
    state.finish_background_task();
    let serial = state.scene[0].history.current_version().serial.as_u64();
    let (fresh, report) = sfmtool_core::bench::create_track(
        &sfmtool_core::bench::Bench::new(),
        state.scene[0].edited(),
        3,
        &sfmtool_core::bench::CreateTrackOptions {
            version: serial,
            label: Some(item.clone()),
        },
    )
    .expect("the point goes on");
    let fresh = (**fresh.track(&report.label).expect("on")).clone();

    call(
        &mut state,
        &mut viewer,
        "create_bench_track",
        json!({ "reconstruction_label": "run_a", "point": 3 }),
    );
    assert_eq!(**state.bench_track(id, &item).expect("on"), fresh);
}

// ── Deleting an image under the bench ───────────────────────────────────

/// The image names a bench track's observations are in, in observation order.
fn bench_observation_names(state: &mut AppState, viewer: &mut Viewer3D, item: &str) -> Vec<Value> {
    let track = call(
        state,
        viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    track["observations"]
        .as_array()
        .expect("a track lists its observations")
        .iter()
        .map(|observation| observation["camera_image_name"].clone())
        .collect()
}

/// Deleting an image renumbers every later one, and the bench moves with the
/// image table: the observation in the deleted image goes, the one after it
/// still names the photograph it was sighted in, and committing the track
/// writes a point whose observations are those photographs at their own
/// pixels. The selected observations follow the renumbering, and an undo
/// brings the bench back as it was. Bug bash 2026-09-29, finding 1.
#[test]
fn deleting_an_image_moves_the_bench_observations_with_the_image_table() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = bench_observation_names(&mut state, &mut viewer, &item);
    assert_eq!(before.len(), 3, "the bench point observes images 0, 1, 2");
    call(
        &mut state,
        &mut viewer,
        "select_bench_observations",
        json!({ "reconstruction_label": "run_a", "track": item, "observations": [1, 2] }),
    );

    let deleted = call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );

    // The track keeps the observations in images 0 and 2, under the indexes
    // those photographs hold now.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let images: Vec<Value> = track["observations"]
        .as_array()
        .expect("observations")
        .iter()
        .map(|observation| observation["camera_image"].clone())
        .collect();
    assert_eq!(images, vec![json!(0), json!(1)], "{track}");
    assert_eq!(
        bench_observation_names(&mut state, &mut viewer, &item),
        vec![before[0].clone(), before[2].clone()],
        "the bench observations name other photographs after the delete"
    );
    // Row 2 is row 1 now, and row 1 was in the deleted image.
    assert_eq!(track["selected_observations"], json!([1]), "{track}");

    // The commit writes those photographs at the pixels they were sighted at,
    // so the point reprojects onto every one of them.
    let committed = call(
        &mut state,
        &mut viewer,
        "commit_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    let point = call(
        &mut state,
        &mut viewer,
        "get_point",
        json!({ "point": committed["point"]["id"] }),
    );
    let written = point["track"].as_array().expect("a track");
    let names: Vec<Value> = written.iter().map(|o| o["name"].clone()).collect();
    assert_eq!(names, vec![before[0].clone(), before[2].clone()], "{point}");
    for observation in written {
        let error = observation["reproj_error"]
            .as_f64()
            .expect("every observation projects");
        assert!(
            error < 0.01,
            "a committed observation is off by {error} px: {point}"
        );
    }

    // The delete's row says what it did to the bench.
    let label = deleted["label"].as_str().expect("a label");
    assert!(
        label.ends_with("; dropped 1 bench observation in it"),
        "{label}"
    );

    // Undoing the commit and the delete puts the bench back with the image.
    for _ in 0..2 {
        call(
            &mut state,
            &mut viewer,
            "undo",
            json!({ "reconstruction_label": "run_a" }),
        );
    }
    assert_eq!(
        bench_observation_names(&mut state, &mut viewer, &item),
        before
    );
}

/// A track whose observations were all in the deleted image is discarded with
/// it, in the same version, and unfocused; the delete's row names it. An undo
/// puts it back.
#[test]
fn deleting_the_only_image_a_bench_track_observes_discards_the_track() {
    let (mut state, mut viewer) = benchable();
    let id = state.scene[0].id;
    let kept = on_the_bench(&mut state, &mut viewer);
    let made = call(
        &mut state,
        &mut viewer,
        "create_bench_cluster",
        json!({
            "reconstruction_label": "run_a",
            "camera_image": 1,
            "pixel": [120.0, 90.0],
            "radius_px": 6.0,
        }),
    );
    let gone = made["item"].as_str().expect("the new item").to_string();
    assert_eq!(state.focused_item_label(id), Some(gone.as_str()));
    let versions = version_count(&state);

    let deleted = call(
        &mut state,
        &mut viewer,
        "delete_camera_image",
        json!({ "reconstruction_label": "run_a", "camera_image": 1 }),
    );
    assert_eq!(version_count(&state), versions + 1, "one version for both");
    let label = deleted["label"].as_str().expect("a label");
    assert!(
        label.ends_with(&format!(
            "; dropped 2 bench observations in it and discarded {gone}, which had no other \
             observations"
        )),
        "{label}"
    );
    let bench = state.bench(id).expect("loaded");
    assert!(
        bench.track(&gone).is_none(),
        "the cluster is still on the bench"
    );
    assert!(bench.track(&kept).is_some(), "the other track went too");
    assert_eq!(
        state.focused_item_label(id),
        None,
        "a discarded item stays focused"
    );

    call(
        &mut state,
        &mut viewer,
        "undo",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(state.bench(id).expect("loaded").track(&gone).is_some());
}
