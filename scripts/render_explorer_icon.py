# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Render SfM Explorer's application icon from its SVG source.

Reads `crates/sfm-explorer/assets/icon/sfm-explorer-icon.svg` and writes, in
the same directory, the PNG sizes the viewer embeds and `sfm-explorer.ico`,
the Windows icon `build.rs` compiles into `sfm-explorer.exe`. See
`specs/gui/application-icon.md`.

Each size is rendered from the SVG by a headless Chromium-family browser
(Microsoft Edge, Google Chrome or Chromium, whichever is found first, or the
one `--browser` names): the SVG is put inline at N x N px on a transparent page
and the browser takes an N x N screenshot of it. The ICO is packed with the
Python standard library: 32-bit BGRA DIB entries for the sizes up to 64 px and
the 256 px PNG as it is.

Run:
    pixi run python scripts/render_explorer_icon.py
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ICON_DIR = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "sfm-explorer"
    / "assets"
    / "icon"
)
SVG_NAME = "sfm-explorer-icon.svg"
ICO_NAME = "sfm-explorer.ico"

# The sizes the viewer embeds as PNG files (`app_icon.rs`).
PNG_SIZES = (32, 64, 256, 512)
# The sizes the Windows icon holds. Those up to 64 px are DIB entries; the
# 256 px one is a PNG entry.
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 256)
ICO_PNG_SIZES = (256,)

BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "microsoft-edge",
    "google-chrome",
    "chromium",
    "chromium-browser",
)


def find_browser() -> str:
    """Return the path of the first Chromium-family browser found."""
    for candidate in BROWSER_CANDIDATES:
        if os.path.isabs(candidate):
            if os.path.isfile(candidate):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    sys.exit("No Chromium-family browser found; name one with --browser.")


def render_png(browser: str, svg: str, size: int, work: Path) -> bytes:
    """Render `svg` at `size` x `size` px with a transparent background."""
    page = work / f"r{size}.html"
    page.write_text(
        "<!doctype html><html><head><style>"
        "html,body{margin:0;background:transparent}"
        f"svg{{display:block;width:{size}px;height:{size}px}}"
        f"</style></head><body>{svg}</body></html>",
        encoding="utf-8",
    )
    out = work / f"icon-{size}.png"
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--default-background-color=00000000",
            f"--user-data-dir={work / 'profile'}",
            f"--window-size={size},{size}",
            f"--screenshot={out}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    data = out.read_bytes()
    width, height, _ = decode_png_rgba(data)
    if (width, height) != (size, size):
        sys.exit(f"The browser rendered {width}x{height} px for size {size}.")
    return data


def png_chunks(data: bytes):
    """Yield the (type, body) chunks of a PNG file."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        yield kind, data[pos + 8 : pos + 8 + length]
        pos += 12 + length


def decode_png_rgba(data: bytes) -> tuple[int, int, bytes]:
    """Decode an 8-bit RGB or RGBA non-interlaced PNG to RGBA rows."""
    idat = b""
    for kind, body in png_chunks(data):
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(
                ">IIBBBBB", body
            )
        elif kind == b"IDAT":
            idat += body
    if depth != 8 or color not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported PNG: depth {depth}, colour type {color}")
    channels = 4 if color == 6 else 3
    stride = width * channels
    raw = zlib.decompress(idat)
    rows = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        kind = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        for i in range(stride):
            a = row[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if kind == 1:
                row[i] = (row[i] + a) & 0xFF
            elif kind == 2:
                row[i] = (row[i] + b) & 0xFF
            elif kind == 3:
                row[i] = (row[i] + (a + b) // 2) & 0xFF
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                row[i] = (row[i] + pred) & 0xFF
        rows.append(bytes(row))
        prev = row
    if channels == 3:
        rows = [
            b"".join(r[i : i + 3] + b"\xff" for i in range(0, len(r), 3)) for r in rows
        ]
    return width, height, b"".join(rows)


def dib_entry(png: bytes) -> bytes:
    """An ICO image entry as a 32-bit BGRA DIB with its AND mask."""
    width, height, rgba = decode_png_rgba(png)
    header = struct.pack("<IiiHHIIiiII", 40, width, 2 * height, 1, 32, 0, 0, 0, 0, 0, 0)
    pixels = bytearray()
    # The mask has a set bit where a pixel is fully transparent, each row
    # padded to four bytes.
    mask_stride = ((width + 31) // 32) * 4
    mask = bytearray()
    for y in reversed(range(height)):
        row = rgba[y * width * 4 : (y + 1) * width * 4]
        mask_row = bytearray(mask_stride)
        for x in range(width):
            r, g, b, a = row[4 * x : 4 * x + 4]
            pixels += bytes((b, g, r, a))
            if a == 0:
                mask_row[x // 8] |= 0x80 >> (x % 8)
        mask += mask_row
    return header + bytes(pixels) + bytes(mask)


def pack_ico(pngs: dict[int, bytes]) -> bytes:
    """Pack the ICO from the rendered PNGs, one entry per size in ICO_SIZES."""
    images = [
        (size, pngs[size] if size in ICO_PNG_SIZES else dib_entry(pngs[size]))
        for size in ICO_SIZES
    ]
    out = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    for size, image in images:
        dim = 0 if size == 256 else size
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(image), offset)
        offset += len(image)
    return out + b"".join(image for _, image in images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browser", help="path of a Chromium-family browser")
    args = parser.parse_args()
    browser = args.browser or find_browser()
    svg = (ICON_DIR / SVG_NAME).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pngs = {
            size: render_png(browser, svg, size, work)
            for size in sorted(set(PNG_SIZES) | set(ICO_SIZES))
        }
    for size in PNG_SIZES:
        (ICON_DIR / f"sfm-explorer-icon-{size}.png").write_bytes(pngs[size])
    (ICON_DIR / ICO_NAME).write_bytes(pack_ico(pngs))
    print(f"Wrote {len(PNG_SIZES)} PNG files and {ICO_NAME} to {ICON_DIR}")


if __name__ == "__main__":
    main()
