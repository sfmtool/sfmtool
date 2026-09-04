# Image Animation & Playback

The image browser can play a reconstruction's images in sequence, like video.
A transport in the browser's navigation minibar and a small set of keyboard
shortcuts start, stop, step and re-speed the playback; each frame it advances
becomes the window's selected image, so the 3D viewport and the detail panels
follow along. Reconstructions built from video are the common case, and
watching the sequence run is how continuity problems — a jump, a drift, a
stretch of bad tracking — announce themselves.

## Problem

The GUI image browser shows a static strip of thumbnails. For video-sequence SfM reconstructions
(the primary use case), there is no way to play through frames sequentially to visually inspect
the reconstruction quality over time. Users must click through images one by one.

Animation playback is particularly useful for:
- Verifying camera trajectory continuity (spotting jumps or drift)
- Inspecting feature tracking quality across frames
- Reviewing reconstruction coverage as the camera moves
- Comparing camera view mode backgrounds across the sequence

## Design

`AnimationState` and `PlayDirection`, the keyboard handling, and the minibar
transport — the play/pause button and the FPS label — all live with the browser
panel in
[image_browser.rs](../../crates/sfm-explorer/src/image_browser.rs). What
[dock.rs](../../crates/sfm-explorer/src/dock.rs) owns is the other end: it reads
`request_camera_switch` off the browser's response and performs the camera
switch.

### Playback State

The image browser holds an `AnimationState` that tracks:
- Whether playback is active (`playing: bool`)
- Playback direction (`direction: PlayDirection` — Forward / Backward)
- Frames per second (`fps: f32`, default 10.0, range 1-60)
- Accumulated time since last frame advance (`accumulated_time: f64`)
- Whether to loop at sequence boundaries (`looping: bool`, default true)

### Controls

**Keyboard shortcuts** — handled inside `ImageBrowser::show()` via `ui.input(|i| i.key_pressed(...))`
following the existing pattern used by `viewer_3d/input.rs`. These keys do not conflict with any
existing shortcuts (Space, Arrow keys, and brackets are unused):

- `Space` — Toggle play/pause
- `Left Arrow` — Step one frame backward (also pauses if playing)
- `Right Arrow` — Step one frame forward (also pauses if playing)
- `[` — Decrease playback speed (halve fps, minimum 1)
- `]` — Increase playback speed (double fps, maximum 60)

**Minibar play controls** — Small transport controls rendered as an overlay on the left edge of
the navigation minibar (no width reduction of the barcode):
- Play/Pause button (triangle / double-bar icon drawn with egui painter)
- FPS indicator label (e.g. "10 fps")

### Playback Behavior

1. **Frame advance**: On each frame, if playing, accumulate `dt` from `ui.input(|i| i.time)`
   (wall-clock time tracking, not `stable_dt` which is smoothed and may drift). When
   `accumulated_time >= 1.0 / fps`, emit a selection change and subtract the frame interval.
   This ensures consistent timing regardless of render framerate.

2. **Auto-selection**: When no image is selected and play is pressed, start from image 0
   (forward) or the last image (backward).

3. **Boundary behavior**:
   - Looping: wrap from last to first (forward) or first to last (backward)
   - Non-looping: pause when reaching the boundary

4. **Camera view sync**: When in camera view mode and playback advances the selected image,
   use `switch_camera_view()` (instant, not animated) for all fps values. Instant switching
   produces a smooth flipbook effect since the sequential camera positions already provide
   visual continuity. Animated transitions would overlap and fight each other at typical
   playback speeds.

5. **Interaction during playback**: Any manual thumbnail click, minibar click/drag, or keyboard
   frame step pauses playback. The user resumes with Space.

6. **Auto-scroll**: The existing auto-scroll logic (center selected thumbnail in view) already
   handles external selection changes, so animation will auto-scroll the strip.

7. **Reconstruction change**: Playback must pause when the reconstruction changes (cache
   invalidation resets `cached_image_count`, which is already detected). Reset animation state
   on invalidation.

8. **Edge cases**: Playback is a no-op when there are fewer than 2 images.

### Integration Points

- `ImageBrowser` carries the `animation: AnimationState` field
- Animation advances are reported through the existing `selection_changed: Option<Option<usize>>`
  field on `ImageBrowserResponse` (no separate field — it reuses the same data path as click
  selection, which already triggers auto-scroll and camera view updates)
- `ImageBrowserResponse` also carries `request_camera_switch: Option<usize>` — when playing in
  camera view mode it signals the instant, non-animated `switch_camera_view()`
- Keyboard handling lives inside `ImageBrowser::show()`, matching the existing pattern where
  each panel reads from egui's global input queue in its own `show()` method
- When playing, `ui.ctx().request_repaint()` is called to ensure continuous rendering

### Camera View Fly-Through

When camera view mode is active and animation is playing, each frame advance triggers an instant
camera switch via `switch_camera_view()` (the non-animated path used by `,`/`.` key navigation).
This produces a flipbook effect with smooth visual continuity from sequential camera positions.

The `request_camera_switch` response field — distinct from `request_camera_view`, which triggers
animated entry into camera view mode — is what selects the instant switch path.

## Non-Goals

- Video export / screen recording
- Audio synchronization
- Variable-speed playback curves (ease in/out)
- Playback of a sub-range of frames (possible future extension)
