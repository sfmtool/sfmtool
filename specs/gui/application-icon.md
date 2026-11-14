# Application Icon

SfM Explorer has one application icon, which the operating system shows for
the viewer in its window's title bar, the taskbar or Dock, the app switcher,
and, on Windows, beside `sfm-explorer.exe` in Explorer. It is the same whether
the viewer runs as the `sfm-explorer` binary or through `sfm explorer`, whose
process is Python's, so on every platform where the viewer can set an icon at
run time, it does.

## What it shows

A dark teal tile with rounded corners, on a transparent square. On the tile are
the 3D points of the `dino_dog_toy` test dataset's ground-truth
reconstruction, each drawn as a small square patch in the point's colour. One
larger patch is outlined in gold, and gold rays run from it down to three teal
camera frustums: a selected point and the cameras that see it.

## Files

The icon lives in
[`crates/sfm-explorer/assets/icon/`](../../crates/sfm-explorer/assets/icon/):

| File | What it is |
|------|------------|
| `sfm-explorer-icon.svg` | The source drawing: a 64-unit view box, the tile inset 2 units from each edge, the corners transparent. Every other file is made from it. |
| `sfm-explorer-icon-32.png`, `-64.png`, `-256.png`, `-512.png` | RGBA renderings at those sizes, the ones the viewer embeds. |
| `sfm-explorer.ico` | The Windows icon: 16, 20, 24, 32, 40, 48 and 64 px as 32-bit BGRA DIB entries with an AND mask, and 256 px as a PNG entry. |

[`scripts/render_explorer_icon.py`](../../scripts/render_explorer_icon.py)
makes the PNG files and the ICO from the SVG, and is run again after any change
to the drawing (`pixi run python scripts/render_explorer_icon.py`). It renders
each size with a headless Chromium-family browser, Microsoft Edge, Google
Chrome or Chromium, whichever it finds first, by putting the SVG inline at
N × N px on a transparent page and taking an N × N screenshot. It packs the ICO
with the Python standard library alone, decoding the browser's PNG output for
the DIB entries and copying the 256 px PNG as it is.

## Where each platform gets it

[`app_icon.rs`](../../crates/sfm-explorer/src/app_icon.rs) embeds the PNG
files a platform uses with `include_bytes!`, so the icon needs no file beside
the binary or the extension. `App::open_window` in
[`lib.rs`](../../crates/sfm-explorer/src/lib.rs) hands them over when it
creates the main window.

- **Windows.** The window's small icon (`ICON_SMALL`, the title bar) is the
  32 px PNG and its big icon (`ICON_BIG`, the taskbar and Alt-Tab) the 64 px
  one, set through winit's `with_window_icon` and `with_taskbar_icon`. A window
  icon set this way is what the taskbar shows for `sfm explorer` too, where
  the executable is `python.exe`. Separately,
  [`build.rs`](../../crates/sfm-explorer/build.rs) compiles
  `sfm-explorer.ico` into the crate's resources with `winres`, beside the
  manifest, as icon resource 1, which is the icon Explorer and shortcuts show
  for `sfm-explorer.exe`.
- **X11.** The window icon is the 256 px PNG, which winit sets as
  `_NET_WM_ICON`.
- **Wayland.** winit sets no window icon there; a Wayland compositor takes an
  application's icon from its desktop entry, and the viewer installs none.
- **macOS.** winit ignores a window icon. `platform::macos::set_dock_icon`
  ([`macos.rs`](../../crates/sfm-explorer/src/platform/macos.rs)) makes the
  512 px PNG the application's icon (`NSApplication`'s
  `setApplicationIconImage:` with an `NSImage` read from the PNG data) once the
  window exists, which the Dock and the app switcher then show.

The embedded PNGs are compile-time assets, but decoding one still does not
panic: a PNG that does not decode is a logged warning and no icon. A unit test
decodes every embedded size on every platform, checks it is a square RGBA
image of its size with transparent corners and an opaque centre, and builds a
winit icon from it, so a broken asset fails the tests rather than reaching a
user.
