# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""A reconstruction as static web files, for ``sfm web-export``.

The data files (``scene.json`` and the JPEG atlas pages) are written by the
Rust core through :func:`sfmtool._sfmtool.io.write_web_export`. This package
holds the other two files, ``index.html`` and ``web-export.js``, which are the
same for every scene, and copies them in beside the data. ``--single-file``
folds all of it into one ``index.html``.

See ``specs/cli/visualization/web-export-command.md``.
"""

import base64
import json
import re
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

#: The viewer files copied into every export, from this package's directory.
STATIC_FILES = ("index.html", "web-export.js")

#: The largest page a claude.ai artifact publishes, and so the largest
#: ``--single-file`` page this writes.
SINGLE_FILE_LIMIT = 16 * 1024 * 1024

#: Names of the files an export writes, which ``overwrite`` replaces.
_WRITTEN = re.compile(
    r"^(scene\.json|index\.html|web-export\.js|(patches|thumbs)-\d+\.jpg)$"
)

#: Where a single-file page's inline scene and viewer go in ``index.html``.
_INLINE_MARKER = "<!-- wx-inline -->"


class WebExportError(Exception):
    """An export that cannot be written as asked."""


def _generator() -> str:
    try:
        return f"sfmtool {version('sfmtool')}"
    except PackageNotFoundError:
        return "sfmtool"


def _static(name: str) -> Path:
    return Path(__file__).parent / name


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    """Make ``out_dir`` ready to write into.

    A missing directory is created. An existing one must be empty, unless
    ``overwrite``, in which case the files an export writes are removed from it
    (so pages left from a larger earlier export do not linger) and every other
    file is kept.
    """
    if out_dir.exists() and not out_dir.is_dir():
        raise WebExportError(f"{out_dir} exists and is not a directory")
    if out_dir.is_dir():
        present = list(out_dir.iterdir())
        if present and not overwrite:
            raise WebExportError(
                f"{out_dir} is not empty; pass --overwrite to replace the files "
                "a web export writes there"
            )
        for path in present:
            if path.is_file() and _WRITTEN.match(path.name):
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def _script_safe(text: str) -> str:
    """``text`` made safe inside a ``<script>`` element: no ``</script``."""
    return re.sub(r"</(script)", r"<\\/\1", text, flags=re.IGNORECASE)


def single_file_page(data_dir: Path) -> str:
    """One self-contained ``index.html`` for the export in ``data_dir``.

    The scene goes in a JSON script element with each atlas page as a data URI,
    and the viewer module goes in a text script element that ``index.html``
    imports through a blob URL.
    """
    scene = json.loads((data_dir / "scene.json").read_text(encoding="utf-8"))
    for atlas in (scene.get("atlases") or {}).values():
        for page in (atlas or {}).get("pages", []):
            data = base64.b64encode((data_dir / page["file"]).read_bytes()).decode(
                "ascii"
            )
            page["file"] = f"data:image/jpeg;base64,{data}"
    scene_text = json.dumps(scene, separators=(",", ":")).replace("</", "<\\/")
    module = _script_safe(_static("web-export.js").read_text(encoding="utf-8"))
    inline = (
        f'<script type="application/json" id="wx-scene">{scene_text}</script>\n'
        f'<script type="text/plain" id="wx-module">{module}</script>'
    )
    page = _static("index.html").read_text(encoding="utf-8")
    if _INLINE_MARKER not in page:
        raise WebExportError("index.html has lost its inline marker")
    return page.replace(_INLINE_MARKER, inline, 1)


def export_reconstruction(
    recon,
    out_dir: Path,
    *,
    source_name: str | None = None,
    overwrite: bool = False,
    single_file: bool = False,
    patches: bool = True,
    thumbnails: bool = True,
    patch_size: int | None = None,
    jpeg_quality: int = 85,
    max_points: int | None = None,
    start_image: str | None = None,
) -> dict:
    """Write ``recon`` as a web export into ``out_dir``.

    Returns the report of :func:`sfmtool._sfmtool.io.write_web_export`, with
    ``files`` listing what is in ``out_dir`` now, by name and size in bytes.

    Raises:
        WebExportError: ``out_dir`` is not empty and ``overwrite`` is off, or a
            single-file page would pass :data:`SINGLE_FILE_LIMIT`, or an option
            does not fit the reconstruction.
    """
    from .._sfmtool.io import write_web_export

    out_dir = Path(out_dir)
    options = dict(
        patches=patches,
        thumbnails=thumbnails,
        patch_size=patch_size,
        jpeg_quality=jpeg_quality,
        max_points=max_points,
        start_image=start_image,
        source_name=source_name,
        generator=_generator(),
    )
    # Refuse what can be refused before anything in out_dir is touched.
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise WebExportError(
            f"{out_dir} is not empty; pass --overwrite to replace the files "
            "a web export writes there"
        )
    if start_image is not None and start_image not in recon.image_names:
        raise WebExportError(
            f"--start-image {start_image!r} names no image of the reconstruction"
        )

    if single_file:
        with tempfile.TemporaryDirectory(prefix="sfm-web-export-") as tmp:
            try:
                report = write_web_export(recon, tmp, **options)
            except ValueError as e:
                raise WebExportError(str(e)) from e
            page = single_file_page(Path(tmp)).encode("utf-8")
        if len(page) > SINGLE_FILE_LIMIT:
            raise WebExportError(
                f"the single-file page would be {len(page) / 2**20:.1f} MB, over the "
                f"{SINGLE_FILE_LIMIT // 2**20} MB a page may be; write a directory "
                "instead, or shrink it with --patch-size, --max-points or --no-thumbnails"
            )
        prepare_output(out_dir, overwrite)
        (out_dir / "index.html").write_bytes(page)
        report["files"] = [("index.html", len(page))]
        return report

    prepare_output(out_dir, overwrite)
    try:
        report = write_web_export(recon, out_dir, **options)
    except ValueError as e:
        raise WebExportError(str(e)) from e
    files = list(report["files"])
    for name in STATIC_FILES:
        shutil.copyfile(_static(name), out_dir / name)
        files.append((name, (out_dir / name).stat().st_size))
    report["files"] = files
    return report
