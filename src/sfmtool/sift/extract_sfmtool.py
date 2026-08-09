# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""sfmtool SIFT extraction backend (the toolkit's own Rust implementation).

Wraps the ``sfmtool._sfmtool.sift.extract_sift`` PyO3 binding so it plugs into the
same extraction pipeline as the COLMAP and OpenCV backends. The Rust core
parallelizes within each image (rayon), but on small images that per-image
parallelism cannot saturate a many-core host on its own, and each image carries
a serial floor (octave-0 build, setup) that leaves cores idle. The backend
therefore decodes and extracts several images concurrently (``cv2.imread`` and
the Rust extract both release the GIL; rayon's shared global pool caps total CPU
threads so this never oversubscribes), which overlaps one image's serial floor
with another's parallel work. Results are still yielded one image at a time in
input order so the caller can stream ``.sift`` writes instead of buffering a
whole chunk in memory. See ``_extract_workers`` for the concurrency knob.
"""

import collections
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sfmtool.sift.file import SiftExtractionError, xxh128_of_file

__all__ = [
    "get_default_sfmtool_feature_options",
    "extract_sift_with_sfmtool",
]


def get_default_sfmtool_feature_options(max_num_features: int | None = None) -> dict:
    """Get default options for the sfmtool SIFT backend.

    The keys mirror the output-defining fields of ``SiftParams::default()`` in
    ``sfmtool-core`` and are passed straight through to the
    ``sfmtool._sfmtool.sift.extract_sift`` binding. They also feed the feature-cache
    hash, so changing any of them yields a distinct cache directory.

    Hardware/performance-only knobs (e.g. thread count) are intentionally
    excluded — they do not change the feature output.

    Args:
        max_num_features: Cap on the number of features kept per image (the
            strongest are retained). ``None`` uses the sfmtool default of 8192.

    Returns:
        Dict of sfmtool SIFT options.
    """
    return {
        "octave_layers": 3,
        "sigma": 1.6,
        "blur_radius_factor": 2.25,
        "input_sigma": 0.5,
        "double_image": True,
        "contrast_threshold": 0.0067,
        "edge_threshold": 10.0,
        "max_num_features": 8192 if max_num_features is None else max_num_features,
        "orientation_bins": 36,
        "peak_ratio": 0.8,
        "descriptor_width": 4,
        "descriptor_bins": 8,
        "descriptor_magnification": 3.0,
        "descriptor_clamp": 0.2,
        # BT.709 luma (matches COLMAP). The grayscale conversion is output-defining,
        # so it is recorded here to pin it in the feature-tool hash even though the
        # version 1 .sift layout does not require an image_to_gray field.
        "gray_formula": "0.2126*R + 0.7152*G + 0.0722*B",
    }


def _decode_image(image_path: Path):
    """Read, decode, and thumbnail one image.

    Called inline at the head of each ``decode_and_extract`` task (fused with
    that image's extract), so it runs on one of the concurrent extract-worker
    threads; ``cv2.imread``/``cv2.cvtColor``/``cv2.resize`` release the GIL, so
    several images decode in parallel. Returns ``(image_path, rgb, thumbnail)``;
    the BGR image is dropped here once the RGB and thumbnail are derived. A decode
    failure raises ``SiftExtractionError``, which the caller's FIFO surfaces in
    input order (see ``_stream_sift_with_sfmtool``).
    """
    image = cv2.imread(
        str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if image is None:
        raise SiftExtractionError(f"Failed to load image: {image_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    thumbnail = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    return image_path, rgb, thumbnail


def _extract_one(
    image_path, rgb, thumbnail, params, feature_options, rust_extract_sift
):
    """Run the Rust extract on a decoded image and assemble its result tuple."""
    height, width = rgb.shape[:2]

    # The binding already returns C-contiguous float32/float32/uint8 arrays,
    # and keypoints already sorted by descending feature size (the .sift
    # ordering every consumer expects) — so no re-contiguize and no re-sort
    # here (matching the COLMAP/OpenCV backends).
    positions, affine_shapes, descriptors = rust_extract_sift(rgb, params)

    file_size = image_path.stat().st_size
    file_xxh128 = xxh128_of_file(image_path)

    feature_tool_metadata = {
        "feature_tool": "sfmtool",
        "feature_type": "sift",
        "feature_options": feature_options,
    }

    metadata = {
        "version": 1,
        "image_name": image_path.name,
        "image_file_xxh128": file_xxh128,
        "image_file_size": file_size,
        "image_width": width,
        "image_height": height,
        "feature_count": len(positions),
    }

    return (
        feature_tool_metadata,
        metadata,
        positions,
        affine_shapes,
        descriptors,
        thumbnail,
    )


def extract_sift_with_sfmtool(
    image_filename_list: list[str | Path],
    feature_options: dict,
    num_threads: int = -1,
):
    """Extract SIFT features from image files using the sfmtool Rust backend.

    Yields one result per image, in input order, decoding and extracting several
    images concurrently (see the module docstring and ``_extract_workers``).
    Yielding incrementally lets the caller stream ``.sift`` writes rather than
    buffering a whole chunk in memory.

    Args:
        image_filename_list: List of absolute paths to image files
        feature_options: Dict of sfmtool SIFT options (see
            ``get_default_sfmtool_feature_options``). Empty/None uses the
            Rust defaults.
        num_threads: Accepted for interface compatibility. The Rust core
            parallelizes within each image via rayon (using all cores for
            ``-1``); cross-image concurrency is controlled by
            ``SFMTOOL_SIFT_EXTRACT_WORKERS`` (see ``_extract_workers``), not this.

    Returns:
        Iterator of tuples (feature_tool_metadata, metadata, positions,
        affine_shapes, descriptors, thumbnail) for each image in order

    Raises:
        SiftExtractionError: If image loading fails or feature extraction fails
    """
    from sfmtool._sfmtool.sift import extract_sift as _rust_extract_sift

    image_filename_list = [
        Path(os.path.normpath(os.path.abspath(p))) for p in image_filename_list
    ]

    # The binding rejects unknown keys, and only the output-defining keys belong
    # here; pass None when there are no overrides so the Rust defaults apply.
    params = feature_options or None

    print(
        f"Extracting features from {len(image_filename_list)} image(s) "
        f"using the sfmtool backend"
    )

    return _stream_sift_with_sfmtool(
        image_filename_list, params, feature_options, _rust_extract_sift
    )


# Rough peak heap one in-flight image adds while it is decoded + extracted: the
# decoded RGB frame plus the f32 gaussian scale-space pyramid the Rust extract
# holds through description. With image doubling the octave-0 base is 4x the
# source pixels and the pyramid keeps s+3 gaussian levels per octave; summed over
# the octave chain that is ~130 bytes per *source* pixel, and the decoded RGB
# adds a few more. Rounded up for safety -- overestimating only forgoes a few
# workers, while underestimating risks OOM on a many-core host with large images.
_EXTRACT_BYTES_PER_SOURCE_PIXEL = 192


def _extract_mem_budget_bytes() -> int:
    """Heap budget shared across concurrently in-flight images (~half of RAM).

    Falls back to a conservative fixed budget when the platform cannot report
    physical memory (e.g. ``os.sysconf`` is absent on Windows).
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        total = 4 * 1024**3  # assume ~4 GiB when the platform won't report it
    return max(1, total // 2)


def _extract_workers(image_pixels: int | None = None) -> int:
    """Number of images to decode+extract concurrently.

    The Rust extract releases the GIL and its internal rayon work funnels
    through one shared global pool (sized to the core count), so running several
    images at once never oversubscribes the CPU -- it just keeps that pool fed,
    overlapping each image's serial floor (octave-0 build, setup) with another
    image's parallel work. On a tiny image the per-image rayon cannot saturate a
    high core count on its own, so this cross-image overlap is where the batch
    speedup comes from -- and to fill a many-core host it must scale *with* the
    core count, not sit at a small constant.

    So the default is ``os.cpu_count()`` images in flight, bounded only by
    memory: each carries a decoded frame + scale-space pyramid
    (``_EXTRACT_BYTES_PER_SOURCE_PIXEL`` per source pixel), so the count is capped
    to keep the in-flight set within ``_extract_mem_budget_bytes()``.
    ``image_pixels`` (source width x height) drives that estimate; when it is
    unknown the default stays at the historical conservative ``min(cores, 4)``
    rather than guess memory blind. ``SFMTOOL_SIFT_EXTRACT_WORKERS`` overrides
    everything (set 1 to disable concurrency).
    """
    override = os.environ.get("SFMTOOL_SIFT_EXTRACT_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            warnings.warn(
                f"Ignoring non-integer SFMTOOL_SIFT_EXTRACT_WORKERS={override!r}; "
                "using the default concurrency.",
                stacklevel=2,
            )
    cores = os.cpu_count() or 1
    if not image_pixels:
        return max(1, min(cores, 4))
    footprint = image_pixels * _EXTRACT_BYTES_PER_SOURCE_PIXEL
    by_memory = max(1, _extract_mem_budget_bytes() // footprint)
    return max(1, min(cores, by_memory))


def _stream_sift_with_sfmtool(
    image_filename_list, params, feature_options, rust_extract_sift
):
    """Generator backing ``extract_sift_with_sfmtool``.

    Decodes and extracts up to ``_extract_workers()`` images concurrently while
    still yielding one result per image in input order (a FIFO over the in-flight
    futures), so the caller can stream ``.sift`` writes. The bounded look-ahead
    caps memory to the worker count, and the ``ThreadPoolExecutor`` context
    manager joins the workers on exit -- including when the consumer stops early
    or a decode/extract raises (re-raised in input order by the FIFO).

    Image 0 is decoded on the calling thread so its dimensions can size the pool
    (peak memory scales with in-flight image count x pyramid size); that decode
    is reused as image 0's input, not repeated.
    """
    paths = iter(image_filename_list)
    first_path = next(paths, None)
    if first_path is None:
        return

    # Decode image 0 up front to learn the source dimensions, then size the pool
    # from them (a batch is like-sized -- one workspace's images). A decode error
    # on image 0 surfaces here, still at the first ``next()``, i.e. in input order.
    first_decoded = _decode_image(first_path)
    try:
        height, width = first_decoded[1].shape[:2]
        image_pixels = height * width
    except (AttributeError, TypeError, ValueError):
        image_pixels = None  # non-array frame (e.g. a test fake): size blind
    workers = _extract_workers(image_pixels)

    def extract_decoded(decoded):
        return _extract_one(*decoded, params, feature_options, rust_extract_sift)

    def decode_and_extract(image_path):
        return extract_decoded(_decode_image(image_path))

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="sift-extract"
    ) as executor:
        pending = collections.deque()

        # Prime the pipeline with up to `workers` images in flight: image 0 reuses
        # the decode above (extract only), the rest decode+extract in workers.
        pending.append(executor.submit(extract_decoded, first_decoded))
        for _ in range(workers - 1):
            image_path = next(paths, None)
            if image_path is None:
                break
            pending.append(executor.submit(decode_and_extract, image_path))

        while pending:
            # Take the oldest in-flight image (preserving input order and
            # re-raising its decode/extract error in order), then top the
            # pipeline back up before yielding so a worker starts the next image
            # while the caller consumes this result.
            result = pending.popleft()
            image_path = next(paths, None)
            if image_path is not None:
                pending.append(executor.submit(decode_and_extract, image_path))
            yield result.result()
