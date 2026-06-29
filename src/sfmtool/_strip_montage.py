# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Lay rendered patch strips out into the labeled side-by-side montage image.

Given fully-rendered rows (each a left/right strip plus its text labels), this
handles only pixels: per-row label panels, group dividers, the title/legend/
column header bars, and writing the stacked PNG. All the ranking and scoring
decisions live in ``_compare_strips``.
"""

from __future__ import annotations

import cv2
import numpy as np

# One montage row: (group_label_or_None, head, line_ref, line_tgt,
#                   ref_strip_or_None, tgt_strip_or_None).
MontageRow = tuple[
    "str | None", str, str, str, "np.ndarray | None", "np.ndarray | None"
]

_LABEL_W = 150


def _label_panel(
    height: int, text_lines: list[str], width: int = _LABEL_W
) -> np.ndarray:
    panel = np.full((height, width, 3), 30, np.uint8)
    y = 18
    for line in text_lines:
        cv2.putText(
            panel,
            line,
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return panel


def _text_bar(
    width: int,
    segments: list[tuple[int, str]],
    height: int = 26,
    bg: int = 60,
    scale: float = 0.5,
) -> np.ndarray:
    bar = np.full((height, width, 3), bg, np.uint8)
    for x, text in segments:
        cv2.putText(
            bar,
            text,
            (x + 6, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return bar


def _pad_to(strip: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), np.uint8)
    h, w = strip.shape[:2]
    out[:h, :w] = strip[:height, :width]
    return out


def _short(name: str, limit: int = 36) -> str:
    return name if len(name) <= limit else ".." + name[-(limit - 2) :]


def _center_in(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Place ``img`` centered on a black ``height``x``width`` canvas (cropping if
    larger). Used for the reference tile, which may be smaller than the row when a
    context padding grows the observation tiles."""
    out = np.zeros((height, width, 3), np.uint8)
    h, w = min(img.shape[0], height), min(img.shape[1], width)
    y0, x0 = (height - h) // 2, (width - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = img[:h, :w]
    return out


def assemble_montage(
    out_path,
    rows: list[MontageRow],
    *,
    title: str,
    legend: str,
    left_label: str,
    right_label: str,
    left_name: str,
    right_name: str,
    tile: int,
    disp: int,
) -> tuple[int, int]:
    """Stack ``rows`` into the montage, write it to ``out_path``, and return its
    ``(width, height)`` in pixels.

    Rows render in order; a row whose ``group`` differs from the previous one is
    preceded by a labeled divider. The left/right strips are padded to the widest
    strip in their column, and a blank block stands in for a missing side.
    """
    w1 = max((r[4].shape[1] for r in rows if r[4] is not None), default=disp)
    w2 = max((r[5].shape[1] for r in rows if r[5] is not None), default=disp)
    row_width = _LABEL_W + 3 + w1 + 6 + w2
    div = np.full((tile, 6, 3), 110, np.uint8)
    sep = np.full((tile, 3, 3), 70, np.uint8)
    blank1 = np.zeros((tile, w1, 3), np.uint8)
    blank2 = np.zeros((tile, w2, 3), np.uint8)

    body_rows: list[np.ndarray] = []
    prev_group = "\0"
    for group, head, line_r, line_t, s1, s2 in rows:
        if group is not None and group != prev_group:
            body_rows.append(
                _text_bar(row_width, [(4, group)], height=18, bg=90, scale=0.42)
            )
        prev_group = group
        label = _label_panel(tile, [head, line_r, line_t])
        left = _pad_to(s1, w1, tile) if s1 is not None else blank1
        right = _pad_to(s2, w2, tile) if s2 is not None else blank2
        row = np.hstack([label, sep, left, div, right])
        body_rows.append(row)
        body_rows.append(np.full((2, row.shape[1], 3), 40, np.uint8))
    body = np.vstack(body_rows[:-1])
    width = body.shape[1]

    # Three stacked, full-width header rows so nothing overlaps: title, the field
    # legend, then the per-column reconstruction names over their blocks.
    title_bar = _text_bar(width, [(0, title)], height=22, bg=45, scale=0.45)
    legend_bar = _text_bar(width, [(0, legend)], height=20, bg=55, scale=0.42)
    cols_bar = _text_bar(
        width,
        [
            (_LABEL_W + 3, f"{left_label.upper()}: {_short(left_name)}"),
            (_LABEL_W + 3 + w1 + 6, f"{right_label.upper()}: {_short(right_name)}"),
        ],
        height=22,
        scale=0.45,
    )
    montage = np.vstack([title_bar, legend_bar, cols_bar, body])
    cv2.imwrite(str(out_path), montage)
    return montage.shape[1], montage.shape[0]


# One point-strips row: (label_lines, reference_tile_or_None, strip_or_None).
PointRow = tuple[list[str], "np.ndarray | None", "np.ndarray | None"]


def assemble_point_strips(
    out_path,
    rows: list[PointRow],
    *,
    title: str,
    legend: str | list[str],
    tile: int,
) -> tuple[int, int]:
    """Stack single-reconstruction point rows into the montage, write it to
    ``out_path``, and return its ``(width, height)`` in pixels.

    Each row is laid out as ``labels | reference patch | observation strip``. The
    reference tile is square at ``tile`` px; strips are padded to the widest strip
    across all rows. ``legend`` may be a single string or a list of strings, one
    stacked bar per line (so a long legend isn't clipped on narrow montages). This
    backs ``sfm inspect --strips``; the two-column ``assemble_montage`` backs
    ``sfm compare --strips``.
    """
    strip_w = max((r[2].shape[1] for r in rows if r[2] is not None), default=tile)
    # The reference column is as wide as the widest reference tile — a stored
    # bitmap renders RGB plus its alpha grayscale side by side, so it is wider
    # than the square consensus tile.
    ref_w = max((r[1].shape[1] for r in rows if r[1] is not None), default=tile)
    div = np.full((tile, 6, 3), 110, np.uint8)
    sep = np.full((tile, 3, 3), 70, np.uint8)
    blank_ref = np.zeros((tile, ref_w, 3), np.uint8)
    blank_strip = np.zeros((tile, strip_w, 3), np.uint8)

    body_rows: list[np.ndarray] = []
    for labels, ref, strip in rows:
        label = _label_panel(tile, labels)
        ref_slot = _center_in(ref, ref_w, tile) if ref is not None else blank_ref
        right = _pad_to(strip, strip_w, tile) if strip is not None else blank_strip
        row = np.hstack([label, sep, ref_slot, div, right])
        body_rows.append(row)
        body_rows.append(np.full((2, row.shape[1], 3), 40, np.uint8))
    body = np.vstack(body_rows[:-1])
    width = body.shape[1]

    title_bar = _text_bar(width, [(0, title)], height=22, bg=45, scale=0.45)
    legend_lines = [legend] if isinstance(legend, str) else list(legend)
    legend_bars = [
        _text_bar(width, [(0, line)], height=18, bg=55, scale=0.42)
        for line in legend_lines
    ]
    cols_bar = _text_bar(
        width,
        [(_LABEL_W + 3, "REF"), (_LABEL_W + 3 + ref_w + 6, "OBSERVATIONS")],
        height=22,
        scale=0.45,
    )
    montage = np.vstack([title_bar, *legend_bars, cols_bar, body])
    cv2.imwrite(str(out_path), montage)
    return montage.shape[1], montage.shape[0]
