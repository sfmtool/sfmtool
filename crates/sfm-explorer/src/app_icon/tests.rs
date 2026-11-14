// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use winit::window::Icon;

/// Every PNG the viewer embeds on some platform, with its side in pixels.
const EMBEDDED_PNGS: [(u32, &[u8]); 4] = [
    (
        32,
        include_bytes!("../../assets/icon/sfm-explorer-icon-32.png"),
    ),
    (
        64,
        include_bytes!("../../assets/icon/sfm-explorer-icon-64.png"),
    ),
    (
        256,
        include_bytes!("../../assets/icon/sfm-explorer-icon-256.png"),
    ),
    (
        512,
        include_bytes!("../../assets/icon/sfm-explorer-icon-512.png"),
    ),
];

/// Each embedded PNG decodes to a square RGBA image of its size, with the
/// tile's rounded corners transparent and its centre opaque, and makes a winit
/// icon. Checked on every platform, so a broken asset fails CI everywhere and
/// not only where it is embedded.
#[test]
fn every_embedded_icon_png_decodes_to_a_square_rgba_icon() {
    for (size, png) in EMBEDDED_PNGS {
        let rgba = image::load_from_memory_with_format(png, image::ImageFormat::Png)
            .unwrap_or_else(|e| panic!("the {size} px icon does not decode: {e}"))
            .into_rgba8();
        assert_eq!(rgba.dimensions(), (size, size), "the {size} px icon");
        assert_eq!(rgba.get_pixel(0, 0)[3], 0, "the {size} px icon's corner");
        assert_eq!(
            rgba.get_pixel(size / 2, size / 2)[3],
            255,
            "the {size} px icon's centre"
        );
        Icon::from_rgba(rgba.into_raw(), size, size)
            .unwrap_or_else(|e| panic!("the {size} px icon is not a winit icon: {e}"));
    }
}

/// The icons the viewer sets on its window are built from the embedded PNGs.
#[cfg(not(target_os = "macos"))]
#[test]
fn the_window_icons_are_built() {
    assert!(super::window_icon().is_some());
    #[cfg(target_os = "windows")]
    assert!(super::taskbar_icon().is_some());
}

/// A PNG that does not decode gives no icon rather than a panic.
#[cfg(not(target_os = "macos"))]
#[test]
fn a_png_that_does_not_decode_gives_no_icon() {
    assert!(super::icon_from_png(b"not a png").is_none());
}
