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
        state.full_res_cache.insert(
            crate::scene::ImageRef::new(id, image),
            Some(std::sync::Arc::new(
                sfmtool_core::camera::remap::ImageU8Pyramid::from_image(
                    sfmtool_core::camera::remap::ImageU8::new(w, h, 3, data),
                    crate::state::PYRAMID_LEVELS,
                ),
            )),
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
    // Named no track, so it landed on the active one, which is the cluster the
    // create just made.
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
    assert_eq!(track["active"], json!(true), "{track}");
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

    // The copy is the active track, so a call that names none acts on it, and
    // it carries the same sightings.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(track["item"], json!(copy), "{track}");
    assert_eq!(track["active"], json!(true), "{track}");
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
        assert_eq!(row["pinned"], json!(false), "a slide is not a verdict");
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
            let expected =
                crate::bench::geometry::project(&camera, &pose, frame.center.coords, frame.w)
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
        crate::bench::geometry::project(camera, pose, xyz, w).expect("in front of the camera")
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
    // observation alone and pins it.
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
        crate::bench::geometry::project(&camera, &pose, xyz, w).expect("in front of the camera")
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
    assert_eq!(
        tilted["normal"],
        json!([asked.x, asked.y, asked.z]),
        "{tilted}"
    );
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
    // A track-stage track first, so the cluster made after it is the active one
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
/// descriptor search in the panel -- carries a seed and no keypoint until a
/// reading is run, and `pixel` is that one answer: the keypoint where there is
/// one, the seed where there is not. So an agent can look at a fresh candidate
/// without first evaluating the track, and it is looking at the place the
/// panel's own mark and tile are drawn at
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
    assert_eq!(rows[candidate]["track"], Value::Null, "{track}");
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

/// A point put on the bench is an item `get_bench` lists, active, at the track
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
    assert_eq!(bench["active"]["track"], json!(item), "{bench}");
    let items = bench["items"].as_array().expect("the items");
    assert_eq!(items.len(), 1, "{bench}");
    assert_eq!(items[0]["item"], json!(item), "{bench}");
    assert_eq!(items[0]["kind"], json!("track"), "{bench}");
    assert_eq!(items[0]["active"], json!(true), "{bench}");
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

/// `deactivate_bench_item` is Track View's Edit box cleared: one version that
/// leaves the item on the bench, `get_bench` then reporting no active track
/// over a bench that has items, and a track tool that names none refused with
/// the remedies. A second call, with nothing active, is a no-effect reply.
#[test]
fn deactivate_leaves_the_item_and_nothing_active() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    let before = version_count(&state);

    let reply = call(
        &mut state,
        &mut viewer,
        "deactivate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(reply["changed"], json!(true), "{reply}");
    assert_eq!(version_count(&state), before + 1, "{reply}");
    assert_eq!(
        reply["label"],
        json!(format!("Stopped editing {item}; it stays on the bench")),
        "{reply}"
    );

    let bench = call(
        &mut state,
        &mut viewer,
        "get_bench",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(bench["active"]["track"], Value::Null, "{bench}");
    assert_eq!(bench["items"].as_array().map(Vec::len), Some(1), "{bench}");

    let refused = refused_call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(
        refused.0,
        "No track is active on run_a's bench. Name one with track, activate one with \
         activate_bench_item, or put one on with create_bench_track or create_bench_cluster."
    );

    let again = call(
        &mut state,
        &mut viewer,
        "deactivate_bench_item",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert_eq!(again["changed"], json!(false), "{again}");
    assert_eq!(
        version_count(&state),
        before + 1,
        "a no-effect call pushed a version"
    );
    let report = again["report"].as_str().expect("the step's own sentence");
    assert!(report.contains("no effect"), "{again}");
}

/// The three item steps: a rename hands back the new label, an activation moves
/// which item a call that names none acts on, and a discard empties the bench.
#[test]
fn rename_activate_and_discard_answer_with_the_item_they_acted_on() {
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
        "activate_bench_item",
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
    assert_eq!(bench["active"]["track"], Value::Null, "{bench}");
}

/// Every refusal is the bench's own sentence and pushes no version.
#[test]
fn the_bench_refuses_in_its_own_words() {
    let (mut state, mut viewer) = benchable();

    // With nothing on the bench there is no active track to act on, and the
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

/// The three steps that read photographs go to a worker and report through the
/// same two-level reply `bundle_adjust` uses.
#[test]
fn evaluate_fit_and_set_stage_run_as_background_tasks() {
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

    let evaluated = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert!(
        evaluated["report"]
            .as_str()
            .expect("a report")
            .starts_with(&format!("Evaluated {item}")),
        "{evaluated}"
    );

    // The operation is the one an agent polls for, under the name the panel
    // shows it as.
    let task = call(&mut state, &mut viewer, "get_background_task", json!({}));
    assert_eq!(task["running"], json!(false), "{task}");
    assert_eq!(task["operation"], json!("Evaluate track"), "{task}");
    assert_eq!(task["reconstruction_label"], json!("run_a"), "{task}");

    // And the measurements are on the wire, under the observation indexes they
    // were computed for.
    let track = call(
        &mut state,
        &mut viewer,
        "get_bench_track",
        json!({ "reconstruction_label": "run_a" }),
    );
    assert!(
        track["observations"][0]["cluster"]["zncc"].is_number(),
        "the evaluation measured nothing: {track}"
    );

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
        json!({ "reconstruction_label": "run_a", "track": item, "search_px": 8.0 }),
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
    for column in ["seed_shift_px", "projection_offset_px"] {
        assert!(
            measured[column].is_number(),
            "{column} is not on the wire: {track}"
        );
    }
    assert!(
        measured["reason"].is_null(),
        "a measured row carries no reason: {track}"
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

    let evaluated = worked(
        &mut state,
        &mut viewer,
        "evaluate_bench_track",
        json!({ "reconstruction_label": "run_a", "track": item }),
    );
    assert!(evaluated["report"].is_string(), "{evaluated}");

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

/// The two steps that read photographs decode **on the worker**: a node whose
/// photographs are neither decoded nor readable starts the task all the same,
/// and the reply is the handle the frame hands back once the window has passed.
#[test]
fn a_slow_evaluate_answers_with_a_handle_naming_it() {
    let (mut state, mut viewer) = benchable();
    let item = on_the_bench(&mut state, &mut viewer);
    // Nothing decoded: the evaluation has every photograph to read, which is
    // the work that must not happen before the deferral.
    state.full_res_cache.clear();

    let arguments = json!({ "reconstruction_label": "run_a" })
        .as_object()
        .cloned()
        .expect("an object");
    let command = tools::parse("evaluate_bench_track", Some(&arguments)).expect("a valid call");
    let pending = match agent(&mut state, &mut viewer, command) {
        Outcome::Deferred(super::super::Deferred::Background(pending)) => pending,
        Outcome::Done(Err(e)) => panic!("expected a deferral, got refusal: {e}"),
        _ => panic!("evaluate must defer to a worker"),
    };
    assert_eq!(pending.operation_name, "Evaluate track");

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
    assert_eq!(reply["operation"], json!("Evaluate track"), "{reply}");
    assert_eq!(reply["operation_id"], json!(operation_id), "{reply}");
    assert_eq!(reply["reconstruction_label"], json!("run_a"), "{reply}");

    // The photographs the worker could not read are its refusal and not the
    // gesture's: the step began either way.
    state.finish_background_task();
    match super::super::edit::background_reply(&state, &past).expect("the operation finished") {
        Err(e) => assert!(e.0.contains(&format!("Cannot evaluate {item}")), "{e}"),
        Ok(_) => panic!("the fixture's photographs are not on disk"),
    }
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
    assert_eq!(bench["active"]["track"], json!(item), "{bench}");
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
