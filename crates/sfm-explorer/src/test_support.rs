// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Helpers shared by the headless UI tests.

use crate::action_log::ActionLog;
use crate::progress::Detail;
use crate::window::{MonitorInfo, WindowChange, WindowError, WindowHost, WindowInfo, WindowState};

/// A [`WindowHost`] with no window behind it: it records the primitives it was
/// asked for, in order, and reports what it then is.
///
/// Lives here rather than in one test module because two need it — `mcp::tests`
/// for the tools, and `layout::tests` because applying a document and drawing
/// the Panels menu both take a host. Recording the *primitives* is the point:
/// the application order is [`WindowHost::apply`]'s provided body, so a test
/// asserting the sequence is asserting the real rule rather than a fake's
/// re-implementation of it.
///
/// The three flags are kept apart rather than collapsed, because a window can
/// be minimized *and* maximized at once and the collapse into one
/// [`WindowState`] is exactly what is under test.
#[derive(Debug, Clone)]
pub(crate) struct FakeWindow {
    pub(crate) minimized: bool,
    pub(crate) maximized: bool,
    pub(crate) fullscreen: bool,
    pub(crate) focused: bool,
    /// `None` stands in for a platform that will not say where a window is,
    /// and refuses a position for the same reason.
    pub(crate) position: Option<[i32; 2]>,
    pub(crate) inner_size: [u32; 2],
    /// A size it will not go below, standing in for whatever a real platform
    /// does to a request it will not honour: clamped, not refused, so that the
    /// read-back has something to differ from the request about.
    pub(crate) minimum: [u32; 2],
    /// Whether it lets an application take focus.
    pub(crate) focusable: bool,
    /// The monitors it reports, the current one first.
    pub(crate) monitors: Vec<MonitorInfo>,
    /// What it was asked, in the order it was asked.
    pub(crate) applied: Vec<String>,
}

impl Default for FakeWindow {
    fn default() -> Self {
        FakeWindow {
            minimized: false,
            maximized: false,
            fullscreen: false,
            focused: true,
            position: Some([120, 64]),
            inner_size: [1920, 1080],
            minimum: [1600, 1200],
            focusable: true,
            monitors: vec![
                MonitorInfo {
                    name: Some("DISPLAY1".to_string()),
                    position: [0, 0],
                    size: [3840, 2160],
                    scale_factor: 1.5,
                },
                MonitorInfo {
                    name: Some("DISPLAY2".to_string()),
                    position: [3840, 0],
                    size: [1920, 1080],
                    scale_factor: 1.0,
                },
            ],
            applied: Vec::new(),
        }
    }
}

impl FakeWindow {
    /// A fake showing one of the four states, everything else as default.
    pub(crate) fn in_state(state: WindowState) -> Self {
        FakeWindow {
            minimized: state == WindowState::Minimized,
            maximized: state == WindowState::Maximized,
            fullscreen: state == WindowState::Fullscreen,
            ..FakeWindow::default()
        }
    }

    pub(crate) fn info(&self) -> WindowInfo {
        WindowInfo {
            state: WindowState::of(self.minimized, self.fullscreen, self.maximized),
            focused: self.focused,
            scale_factor: 1.5,
            outer_position: self.position,
            // Inside a frame 8 px wide at the sides and bottom and 31 px at the
            // top, which is what the outer size below adds.
            inner_position: self.position.map(|[x, y]| [x + 8, y + 31]),
            outer_size: [self.inner_size[0] + 16, self.inner_size[1] + 39],
            inner_size: self.inner_size,
            monitor: self.monitors.first().cloned(),
        }
    }
}

impl WindowHost for FakeWindow {
    fn set_state(&mut self, state: WindowState) {
        self.applied.push(format!("state {}", state.wire_name()));
        match state {
            WindowState::Normal => {
                self.minimized = false;
                self.fullscreen = false;
                self.maximized = false;
            }
            WindowState::Maximized => {
                self.minimized = false;
                self.fullscreen = false;
                self.maximized = true;
            }
            WindowState::Minimized => self.minimized = true,
            WindowState::Fullscreen => {
                self.minimized = false;
                self.fullscreen = true;
            }
        }
    }

    fn set_outer_position(&mut self, [x, y]: [i32; 2]) -> Result<(), WindowError> {
        self.applied.push(format!("position {x},{y}"));
        if self.position.is_none() {
            return Err(WindowError(
                "This platform does not let an application position its own window.".to_string(),
            ));
        }
        self.position = Some([x, y]);
        Ok(())
    }

    fn set_inner_size(&mut self, [width, height]: [u32; 2]) {
        self.applied.push(format!("size {width}x{height}"));
        self.inner_size = [width.max(self.minimum[0]), height.max(self.minimum[1])];
    }

    fn focus(&mut self) {
        self.applied.push("focus".to_string());
        self.focused = self.focusable;
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        Some((self.info(), self.monitors.clone()))
    }
}

/// The host for the windowless case: there is no window here.
///
/// It refuses rather than pretending — a caller that forgot to pass a real host
/// fails loudly rather than silently exercising nothing. Every primitive is
/// unreachable, because [`WindowHost::apply`] stops at the observation.
pub(crate) struct NoWindow;

impl WindowHost for NoWindow {
    fn set_state(&mut self, _state: WindowState) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn set_outer_position(&mut self, _position: [i32; 2]) -> Result<(), WindowError> {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn set_inner_size(&mut self, _size: [u32; 2]) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn focus(&mut self) {
        unreachable!("apply stops at the observation where there is no window")
    }

    fn observe(&self) -> Option<(WindowInfo, Vec<MonitorInfo>)> {
        None
    }
}

/// A change that carries nothing but geometry, for the fits and the orders.
#[allow(dead_code)] // Used by `layout::tests`; `mcp::tests` builds its own.
pub(crate) fn geometry(position: Option<[i32; 2]>, size: Option<[u32; 2]>) -> WindowChange {
    WindowChange {
        outer_position: position,
        inner_size: size,
        ..WindowChange::default()
    }
}

/// Run one egui frame headlessly and discard its output.
///
/// The wrapper exists for the discarding: since epaint 0.36 a
/// [`TexturesDelta`](egui::TexturesDelta) panics on drop if it still holds
/// unapplied deltas, on the grounds that a real integration losing them is a
/// bug ("texture has not been allocated yet" on a later partial update — the
/// same hazard `app.rs` documents around its `update_texture` loop). These
/// frames have no painter at all, so the deltas are dropped deliberately, which
/// is exactly the case epaint asks be spelled out with an explicit `clear`.
pub(crate) fn run_frame_headless(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
}

/// Every string painted in one headless frame, in paint order.
///
/// A panel that elides its own text has to be asked what it actually drew, and
/// egui's frame output carries the galleys: laying out needs no GPU, so the
/// strings are real even with no painter behind them. Nested `Shape::Vec`s are
/// walked, since a widget's shapes arrive grouped.
pub(crate) fn painted_texts(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) -> Vec<String> {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
    output_texts(&output)
}

/// Every string a frame already run painted, for a test that needs the rest
/// of its output as well.
pub(crate) fn output_texts(output: &egui::FullOutput) -> Vec<String> {
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        collect_texts(&clipped.shape, &mut texts);
    }
    texts
}

/// Every string painted in one headless frame, with where it was painted and
/// the rectangle it was clipped to.
///
/// The clip rectangle is the point of it. A `ScrollArea` clips its contents to
/// the viewport it scrolls them through, so a table header drawn *outside* one
/// carries a different clip rectangle from the rows, and the two rectangles say
/// which side of the boundary each string was drawn on. Two panels draw such a
/// table, which is why the helper is here rather than in one of them.
pub(crate) fn painted_text_rects(
    ctx: &egui::Context,
    input: egui::RawInput,
    run_ui: impl FnMut(&mut egui::Ui),
) -> Vec<PaintedText> {
    let mut output = ctx.run_ui(input, run_ui);
    output.textures_delta.clear();
    output_text_rects(&output)
}

/// Every string a frame already run painted, with where it was painted and
/// the rectangle it was clipped to, for a test that needs the rest of its
/// output as well.
pub(crate) fn output_text_rects(output: &egui::FullOutput) -> Vec<PaintedText> {
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        collect_text_rects(&clipped.shape, clipped.clip_rect, &mut texts);
    }
    texts
}

/// One string a headless frame painted.
#[derive(Debug, Clone)]
pub(crate) struct PaintedText {
    pub(crate) text: String,
    /// Where the glyphs landed, in screen coordinates.
    pub(crate) rect: egui::Rect,
    /// What the painter was clipped to when it drew them.
    pub(crate) clip: egui::Rect,
}

fn collect_text_rects(shape: &egui::Shape, clip: egui::Rect, out: &mut Vec<PaintedText>) {
    match shape {
        egui::Shape::Text(text) => out.push(PaintedText {
            text: text.galley.text().to_owned(),
            rect: egui::Rect::from_min_size(text.pos, text.galley.size()),
            clip,
        }),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_text_rects(shape, clip, out);
            }
        }
        _ => {}
    }
}

fn collect_texts(shape: &egui::Shape, out: &mut Vec<String>) {
    match shape {
        egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_texts(shape, out);
            }
        }
        _ => {}
    }
}

// -- What an operation reported ------------------------------------------

/// The phase rows of one Action Log entry's breakdown: each stage's name, the
/// depth it sits at, and how many runs of it folded into the row.
///
/// Here rather than in one test module because three of them ask the same
/// question of three different operations, and a second copy would be a second
/// answer to keep in step.
pub(crate) fn phase_rows(detail: &[Detail]) -> Vec<(&'static str, u8, u32)> {
    detail
        .iter()
        .filter_map(|row| match row {
            Detail::Phase {
                name, depth, runs, ..
            } => Some((*name, *depth, *runs)),
            Detail::Message { .. } => None,
        })
        .collect()
}

/// What the stage called `name` said it did, or `None` when it said nothing or
/// never ran.
pub(crate) fn phase_note(detail: &[Detail], name: &str) -> Option<String> {
    detail.iter().find_map(|row| match row {
        Detail::Phase {
            name: phase, note, ..
        } if *phase == name => note.clone(),
        _ => None,
    })
}

/// Stamp the log as a frame would, then assert that the newest entry's cost
/// holds the stages it named.
///
/// The inequality is what says the row was timed from the work rather than
/// from the write: every phase ran inside the operation, so an entry whose
/// clock started only once the operation was over could not contain them.
pub(crate) fn assert_timed_from_the_work(log: &mut ActionLog) {
    log.settle(std::time::Instant::now(), Vec::new());
    let entry = log.entries().next_back().expect("an entry to settle");
    let took = entry.took.expect("the settle to have stamped it");
    let named: std::time::Duration = entry
        .detail
        .iter()
        .filter_map(|row| match row {
            Detail::Phase { depth: 0, took, .. } => Some(*took),
            _ => None,
        })
        .sum();
    assert!(
        took >= named,
        "the entry reports {took:?}, less than the {named:?} its own stages cost: \
         it was timed from the write rather than from the work",
    );
}

/// A non-zero intrinsic adjustment of the size a phone camera's images show:
/// a focal one percent longer, and a principal point moved by two percent of
/// the short side right and one percent up.
pub(crate) const ADJUSTMENT: sfmtool_core::IntrinsicAdjustment =
    sfmtool_core::IntrinsicAdjustment {
        focal: 0.01,
        principal_point: [0.02, -0.01],
        radial_bspline: [0.0; 2],
        rolling_shutter: [0.0; 3],
    };

/// `recon` with `adjustment` on image `image`, and the image's camera listing
/// both parameters as varying per image.
pub(crate) fn with_adjustment(
    mut recon: sfmtool_core::SfmrReconstruction,
    image: usize,
    adjustment: sfmtool_core::IntrinsicAdjustment,
) -> sfmtool_core::SfmrReconstruction {
    let image_set = &mut recon.image_set;
    let camera = image_set.images[image].camera_index as usize;
    image_set.images[image].intrinsic_adjustment = adjustment;
    image_set
        .per_image_adjustments
        .resize(image_set.cameras.len(), Default::default());
    image_set.per_image_adjustments[camera] = sfmtool_core::reconstruction::PerImageAdjustments {
        focal: true,
        principal_point: true,
        radial_bspline: false,
        rolling_shutter: false,
        rolling_shutter_scanlines: None,
    };
    recon
}

/// Per-image lens corrections about ten times the size the one measured phone
/// capture shows, so that what they move is well clear of rounding: a radial
/// correction of two percent of the normalized radius at the spline's domain
/// end, and rolling-shutter rates of tens of milliradians per short side.
pub(crate) const LENS_CORRECTIONS: sfmtool_core::IntrinsicAdjustment =
    sfmtool_core::IntrinsicAdjustment {
        focal: 0.0,
        principal_point: [0.0; 2],
        radial_bspline: [0.01, 0.02],
        rolling_shutter: [0.02, -0.01, 0.005],
    };

/// `recon` with every camera made `SFMTOOL_PINHOLE`, the camera's focal and
/// principal point kept and a spline of four zero coefficients on `[0, 1.2]`,
/// so it projects as the pinhole did but can take a radial correction and
/// rolling-shutter rates.
pub(crate) fn as_sfmtool_pinhole(
    mut recon: sfmtool_core::SfmrReconstruction,
) -> sfmtool_core::SfmrReconstruction {
    for camera in &mut recon.image_set.cameras {
        let (focal_length, _) = camera.focal_lengths();
        let (principal_point_x, principal_point_y) = camera.principal_point();
        camera.model = sfmtool_core::CameraModel::SfmtoolPinhole {
            focal_length,
            principal_point_x,
            principal_point_y,
            bspline_rho_max: 1.2,
            bspline: vec![0.0; 4],
        };
    }
    recon
}

/// `recon` with `adjustment` on image `image`, the image's camera listing all
/// four parameters as varying per image, with the scanlines setting
/// `scanlines`.
pub(crate) fn with_lens_corrections(
    recon: sfmtool_core::SfmrReconstruction,
    image: usize,
    adjustment: sfmtool_core::IntrinsicAdjustment,
    scanlines: sfmtool_core::camera::RollingShutterScanlines,
) -> sfmtool_core::SfmrReconstruction {
    let mut recon = with_adjustment(recon, image, adjustment);
    let camera = recon.image_set.images[image].camera_index as usize;
    let varies = &mut recon.image_set.per_image_adjustments[camera];
    varies.radial_bspline = true;
    varies.rolling_shutter = true;
    varies.rolling_shutter_scanlines = Some(scanlines);
    recon
        .image_set
        .check_image_intrinsics()
        .expect("the corrections apply to the camera");
    recon
}

/// `recon` with image `image` moved onto a camera entry of its own that holds
/// the image's intrinsics, its adjustment applied, and with no adjustment.
///
/// What an adjusted image has to draw exactly as: the same photograph through
/// the same intrinsics, said the old way.
pub(crate) fn image_on_own_camera(
    recon: &sfmtool_core::SfmrReconstruction,
    image: usize,
) -> sfmtool_core::SfmrReconstruction {
    let mut out = recon.clone();
    let intrinsics = recon.image_set.image_intrinsics(image).into_owned();
    let image_set = &mut out.image_set;
    image_set.cameras.push(intrinsics);
    if !image_set.per_image_adjustments.is_empty() {
        image_set
            .per_image_adjustments
            .resize(image_set.cameras.len(), Default::default());
    }
    image_set.images[image].camera_index = (image_set.cameras.len() - 1) as u32;
    image_set.images[image].intrinsic_adjustment = Default::default();
    out.metadata.camera_count = image_set.cameras.len() as u32;
    out
}

/// Put in `photographs`, for each of `paths`, a pyramid of the photograph's
/// size that holds only an 8 px corner of each level, as one a caller made
/// with `ImageU8Pyramid::from_regions` and inserted can be, so a read of any
/// other pixel of it fails. The size is each file's own, read from its
/// header, so the photograph passes a size check.
pub(crate) fn insert_corners_only(
    photographs: &sfmtool_core::camera::PhotographCache,
    paths: &[std::path::PathBuf],
) {
    use sfmtool_core::camera::image::{ImageU8, ImageU8Pyramid, PyramidLevel};
    for path in paths {
        let (width, height) = image::image_dimensions(path).expect("a photograph on disk");
        let levels = ImageU8Pyramid::full_levels(width, height);
        let (mut w, mut h) = (width, height);
        let mut corners = Vec::with_capacity(levels);
        for _ in 0..levels {
            let (cw, ch) = (w.min(8), h.min(8));
            corners.push(Some(PyramidLevel {
                image: ImageU8::new(cw, ch, 3, vec![128; (cw * ch * 3) as usize]),
                origin: [0, 0],
                full_size: [w, h],
            }));
            (w, h) = (w / 2, h / 2);
        }
        let pyramid = ImageU8Pyramid::from_regions(corners).expect("a valid region pyramid");
        photographs.insert(path, std::sync::Arc::new(pyramid));
    }
}

/// `recon` without its inline keypoints: the value an open gives for a version
/// 16 `sift_files` file written without `tracks/keypoints_xy` whose `.sift`
/// files cannot be read, which states no pixel for any observation.
pub(crate) fn without_inline_keypoints(
    mut recon: sfmtool_core::SfmrReconstruction,
) -> sfmtool_core::SfmrReconstruction {
    if let sfmtool_core::ObservationSource::SiftFiles { .. } = &recon.point_set.observations {
        recon.point_set.keypoints_xy = None;
    }
    recon
}

/// Rewrite the version 17 `sift_files` `.sfmr` at `path` in place as a version
/// 16 file written without `tracks/keypoints_xy`, which version 16 allowed:
/// the entry left out, `has_keypoints_xy` false and `metadata.json` at version
/// 16. The readers do not check the content hashes, so they are left as they
/// are.
pub(crate) fn rewrite_as_version_16_without_keypoints(path: &std::path::Path) {
    use std::io::{Read, Write};

    let mut archive = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    let mut entries = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let name = entry.name().unwrap().to_string();
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();
        entries.push((name, compressed));
    }
    drop(archive);
    let edit_json = |compressed: &[u8], key: &str, value: serde_json::Value| {
        let raw = zstd::stream::decode_all(compressed).unwrap();
        let mut json: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        json[key] = value;
        zstd::bulk::compress(&serde_json::to_vec(&json).unwrap(), 3).unwrap()
    };
    let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, compressed) in entries {
        let bytes = match name.as_str() {
            n if n.starts_with("tracks/keypoints_xy.") => continue,
            "metadata.json.zst" => edit_json(&compressed, "version", serde_json::json!(16)),
            "tracks/metadata.json.zst" => {
                edit_json(&compressed, "has_keypoints_xy", serde_json::json!(false))
            }
            _ => compressed,
        };
        zip.start_file(&name, stored).unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.finish().unwrap();
}

/// A photograph held in a photograph cache of its own, as a panel test hands
/// it to Image Detail: the cache answers its layout and its tiles at once,
/// cut from `pyramid`, so a frame draws it as the panel draws a photograph
/// the session's cache holds, and starts no task.
pub(crate) struct HeldPhotograph {
    cache: std::sync::Arc<sfmtool_core::camera::PhotographCache>,
    requests: crate::state::PhotographRequests,
    path: std::path::PathBuf,
}

impl HeldPhotograph {
    pub(crate) fn new(pyramid: sfmtool_core::camera::image::ImageU8Pyramid) -> Self {
        let cache = sfmtool_core::camera::PhotographCache::new(
            sfmtool_core::camera::PhotographCacheOptions {
                budget_bytes: 1 << 30,
                ..Default::default()
            },
        );
        let path = std::path::PathBuf::from("/held/photograph.jpg");
        cache.insert(&path, std::sync::Arc::new(pyramid));
        Self {
            cache: std::sync::Arc::new(cache),
            requests: crate::state::PhotographRequests::default(),
            path,
        }
    }

    /// A photograph the cache does not hold and knows nothing of, whose reads
    /// wait in a queue that is never run: what the panel sees of a
    /// photograph the cache has dropped, until it is read again.
    pub(crate) fn dropped() -> Self {
        let cache = sfmtool_core::camera::PhotographCache::new(
            sfmtool_core::camera::PhotographCacheOptions {
                budget_bytes: 1 << 30,
                ..Default::default()
            },
        );
        Self {
            cache: std::sync::Arc::new(cache),
            requests: crate::state::PhotographRequests::queued().0,
            path: std::path::PathBuf::from("/held/photograph.jpg"),
        }
    }

    /// The photograph as the panel asks for it.
    pub(crate) fn source(&self) -> crate::state::PhotographSource<'_> {
        crate::state::PhotographSource {
            cache: &self.cache,
            requests: &self.requests,
            path: &self.path,
        }
    }
}

/// The demo scene at its default options, rendered once for the whole test
/// binary, and the exact reconstruction of it: five 640 x 480 photographs of
/// a textured ground and a cube, and an `embedded_patches` value whose every
/// keypoint is its point's projection and every normal its surface's
/// (`specs/drafts/demo-scene.md`).
///
/// For a test whose assertion needs photographs that agree with the value:
/// a photometric step whose answer can be checked against the stored truth.
/// A test that only wants a cheap value with no pixels keeps
/// `SfmrReconstruction::demo` or `projected_embedded_demo`.
pub(crate) struct DemoSceneFixture {
    pub(crate) scene: sfmtool_core::reconstruction::DemoScene,
    pub(crate) recon: sfmtool_core::SfmrReconstruction,
}

/// The one [`DemoSceneFixture`] of the test binary. The render and the value
/// take about a tenth of a second, paid by the first test that asks.
pub(crate) fn demo_scene() -> &'static DemoSceneFixture {
    static SCENE: std::sync::OnceLock<DemoSceneFixture> = std::sync::OnceLock::new();
    SCENE.get_or_init(|| {
        use sfmtool_core::progress::Progress;
        use sfmtool_core::reconstruction::{DemoScene, DemoSceneOptions};
        let scene = DemoScene::render(&DemoSceneOptions::default(), &Progress::none())
            .expect("the default scene renders");
        let recon = scene
            .reconstruction(&Progress::none())
            .expect("the default scene's value builds");
        DemoSceneFixture { scene, recon }
    })
}

/// A state holding `recon`, a value of the demo scene's images, as its one
/// node, with each of the scene's photographs in the photograph cache as the
/// photograph of its image, so a step that reads pixels finds them without a
/// file on disk.
pub(crate) fn demo_scene_state_of(
    recon: sfmtool_core::SfmrReconstruction,
) -> (crate::state::AppState, crate::scene::ReconId) {
    let mut state = crate::state::AppState::new();
    state.append_node(crate::scene::SceneNode::demo(recon));
    let id = state.selected_recon.expect("a selected reconstruction");
    for (index, photograph) in demo_scene().scene.photographs().iter().enumerate() {
        let copy = sfmtool_core::camera::image::ImageU8::new(
            photograph.width(),
            photograph.height(),
            photograph.channels(),
            photograph.data().to_vec(),
        );
        state.insert_photograph(id, index, copy);
    }
    (state, id)
}

/// The index of the first point of the demo scene's value whose stored normal
/// is `normal` and which `views` cameras observe.
pub(crate) fn demo_scene_point(normal: [f32; 3], views: usize) -> usize {
    let recon = &demo_scene().recon;
    (0..recon.point_count())
        .find(|&p| {
            let n = recon.point_set.points[p].normal;
            (n - nalgebra::Vector3::from(normal)).norm() < 1e-6
                && recon.point_set.track_image_indices(p).len() == views
        })
        .unwrap_or_else(|| panic!("the scene has a point facing {normal:?} seen {views} times"))
}
