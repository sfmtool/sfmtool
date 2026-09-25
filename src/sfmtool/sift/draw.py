# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw SIFT feature geometry on images."""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_sift_features(
    image_path: str | Path,
    output_path: str | Path,
    max_features: int | None = None,
    feature_indices: "np.ndarray | None" = None,
    feature_tool: str | None = None,
    feature_options: dict | None = None,
) -> None:
    """Draw SIFT features with affine shape ellipses on an image.

    Args:
        image_path: Path to the input image file
        output_path: Path where the output image should be saved
        max_features: Optional limit on number of features to draw (draws largest first)
        feature_indices: Optional array of feature indices to draw.
        feature_tool: Feature extraction tool name
        feature_options: Optional dict with feature tool options.

    Raises:
        FileNotFoundError: If image or SIFT file doesn't exist
    """
    import cv2

    from .file import SiftReader, get_sift_path_for_image

    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(
        str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    sift_path = get_sift_path_for_image(
        image_path,
        feature_tool=feature_tool,
        feature_options=feature_options,
    )
    try:
        with SiftReader(sift_path) as reader:
            positions, affine_shapes = reader.read_positions_and_shapes(
                count=max_features
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"SIFT file not found for image {image_path}. "
            f"Expected at: {sift_path}. "
            f"Run 'sfm sift --extract' first to generate SIFT features."
        )

    if feature_indices is not None:
        positions = positions[feature_indices]
        affine_shapes = affine_shapes[feature_indices]

    for pos, affine_matrix in zip(positions, affine_shapes):
        center_x, center_y = float(pos[0]), float(pos[1])
        center = (int(round(center_x)), int(round(center_y)))

        # The affine applied to the unit circle, drawn as it is. Axis lengths
        # plus one angle would misplace the major axis of a sheared or
        # anisotropic shape: it lies along the left singular vector, which is
        # the first column's direction only for a similarity.
        t = np.linspace(0.0, 2.0 * np.pi, 65)
        circle = np.stack([np.cos(t), np.sin(t)])
        ring = np.asarray(affine_matrix, dtype=float) @ circle
        pts = np.rint(ring.T + [center_x, center_y]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, (0, 255, 0), 1)  # green in BGR
        cv2.circle(image, center, 2, (0, 0, 255), -1)  # red center

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Drew {len(positions)} features on {image_path.name}")
    print(f"  Saved to: {output_path}")
