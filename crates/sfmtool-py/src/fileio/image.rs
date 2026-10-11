// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for reading and writing image files: the header-only reads
//! `image_dimensions` and `image_has_alpha`, the full decodes
//! `read_image_rgb` and `read_image_rgba`, and the encoders `write_image_rgb`
//! and `write_image_rgba`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use sfmtool_core::camera::image::{ImageU8, DEFAULT_JPEG_QUALITY};

/// Read an image file's `(width, height)` from its header alone.
///
/// Decodes only the format header — JPEG `SOF`, PNG `IHDR`, and so on — never
/// the pixel data, so it stays cheap on large images. The dimensions are the
/// ones stored in the file; EXIF orientation is not applied. The contents, not
/// the extension, choose the decoder, as in `read_image_rgb`.
#[pyfunction]
pub fn image_dimensions(path: PathBuf) -> PyResult<(u32, u32)> {
    sfmtool_core::camera::image::image_dimensions(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Whether the image file at `path` stores an alpha channel, from its header
/// alone.
///
/// The contents, not the extension, choose the decoder, as in
/// `read_image_rgb`. A caller that writes the image back out reads it with
/// `read_image_rgba` when this is true, so the alpha survives, and with
/// `read_image_rgb` otherwise. The errors are `read_image_rgb`'s.
#[pyfunction]
pub fn image_has_alpha(path: PathBuf) -> PyResult<bool> {
    sfmtool_core::camera::image::image_has_alpha(&path).map_err(|e| read_error(&path, e))
}

/// Decode the image file at `path` to a `y_x_rgb` `uint8` array, `(H, W, 3)`.
///
/// The layout is the one a `.sfmr` file stores its thumbnails in
/// (`images/thumbnails_y_x_rgb`): row, column, then the channels in RGB order,
/// C-contiguous. This is the decoder the Rust code reads photographs with
/// (`ImageU8::read_rgb`: the `jpeg-decoder` crate for a JPEG, the `image`
/// crate otherwise), so the pixels equal the ones the viewer, the bench and
/// the photograph cache read, bit for bit. The EXIF
/// orientation is ignored: the array has the width and height stored in the
/// file, as the SIFT extractors and the camera intrinsics do. A grey image has
/// its value repeated in the three channels, an alpha channel is dropped, and
/// a 16-bit image is scaled to 8 bits. The GIL is released while decoding.
///
/// Raises `FileNotFoundError` when the file does not exist and `OSError`
/// when it cannot be read or decoded; both messages name the path.
#[pyfunction]
pub fn read_image_rgb<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = py.detach(|| ImageU8::read_rgb(&path));
    to_array(py, &path, image)
}

/// Decode the image file at `path` to a `y_x_rgba` `uint8` array, `(H, W, 4)`.
///
/// The layout and alpha rule of a `.sfmr` file's patch bitmaps
/// (`points3d/patch_bitmaps_y_x_rgba`): the colour channels equal
/// `read_image_rgb`'s, and the alpha is the file's own where the image has one
/// and 255 where it has none, so `alpha > 0` marks pixels with data. Decoded
/// by `ImageU8::read_rgba`, with the EXIF orientation ignored. The GIL is
/// released while decoding, and the errors are `read_image_rgb`'s.
#[pyfunction]
pub fn read_image_rgba<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = py.detach(|| ImageU8::read_rgba(&path));
    to_array(py, &path, image)
}

/// The `(H, W, C)` array of a decoded image, or the Python exception for a
/// failed decode, naming `path`.
fn to_array<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    decoded: Result<ImageU8, image::ImageError>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let image = decoded.map_err(|e| read_error(path, e))?;
    let shape = (
        image.height() as usize,
        image.width() as usize,
        image.channels() as usize,
    );
    let array = ndarray::Array3::from_shape_vec(shape, image.into_data())
        .expect("an ImageU8 holds height * width * channels bytes");
    Ok(array.into_pyarray(py))
}

/// The Python exception for an image file that could not be read, naming
/// `path`: `FileNotFoundError` for a missing file, `OSError` otherwise.
fn read_error(path: &std::path::Path, e: image::ImageError) -> PyErr {
    let message = format!("could not read image {}: {e}", path.display());
    match &e {
        image::ImageError::IoError(io) if io.kind() == std::io::ErrorKind::NotFound => {
            pyo3::exceptions::PyFileNotFoundError::new_err(message)
        }
        _ => pyo3::exceptions::PyOSError::new_err(message),
    }
}

/// Encode a `y_x_rgb` `uint8` array, `(H, W, 3)`, to the image file at
/// `path`, in the format the path's extension names.
///
/// The inverse of `read_image_rgb`: the array is indexed row, column, channel,
/// with the channels in RGB order. An array in any other memory layout, such
/// as a slice of a wider image or a Fortran-ordered array, is copied to row
/// order before it is encoded. `.jpg` / `.jpeg` write a 4:2:0 JPEG at
/// `jpeg_quality` (the default 95 is OpenCV's), `.png` a lossless PNG at fast
/// compression, and any other extension the `image` crate encodes 8-bit pixels
/// to (`.tif`, `.bmp`, `.webp`, ...) is written with that crate's defaults.
/// `jpeg_quality` must be 1 to 100 whatever the format, though only a JPEG
/// uses it. Every format is written at 8 bits per channel. The image is
/// encoded by `ImageU8::write` with the GIL released, in memory first, and
/// then written to a temporary file beside `path` that is renamed into place,
/// so a failed write leaves neither a partial file nor a changed old one.
/// Where another process holds `path` open and the rename is refused, as on
/// Windows, the file is written in place instead.
///
/// Raises `TypeError` when `pixels` is not a `uint8` array; `ValueError` when
/// it is not `(H, W, 3)` with `H` and `W` at least 1, when `jpeg_quality` is
/// outside 1 to 100, when the extension names no format the writer encodes,
/// or when the format's encoder refuses the image, such as a side longer than
/// the format allows (65535 for JPEG, 16384 for WebP);
/// `FileNotFoundError` when the parent directory does not exist; and `OSError`
/// when the file cannot be written otherwise. Each message names the path.
#[pyfunction]
#[pyo3(signature = (path, pixels, *, jpeg_quality = DEFAULT_JPEG_QUALITY as i64))]
pub fn write_image_rgb(
    py: Python<'_>,
    path: PathBuf,
    pixels: &Bound<'_, PyAny>,
    jpeg_quality: i64,
) -> PyResult<()> {
    if !(1..=100).contains(&jpeg_quality) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "jpeg_quality must be 1 to 100, got {jpeg_quality}"
        )));
    }
    let image = to_image(pixels, 3, "write_image_rgb", "(H, W, 3) RGB")?;
    let written = py.detach(|| image.write(&path, jpeg_quality as u8));
    written.map_err(|e| write_error(&path, e))
}

/// Encode a `y_x_rgba` `uint8` array, `(H, W, 4)`, to the image file at
/// `path`, in the format the path's extension names.
///
/// The inverse of `read_image_rgba`, with `write_image_rgb`'s rules for the
/// layout, the formats and the errors. The format must hold an alpha channel:
/// writing to `.jpg` / `.jpeg` raises `ValueError` rather than drop the alpha,
/// so a caller with an opaque image writes its colour channels with
/// `write_image_rgb`.
#[pyfunction]
pub fn write_image_rgba(py: Python<'_>, path: PathBuf, pixels: &Bound<'_, PyAny>) -> PyResult<()> {
    let image = to_image(pixels, 4, "write_image_rgba", "(H, W, 4) RGBA")?;
    let written = py.detach(|| image.write(&path, DEFAULT_JPEG_QUALITY));
    written.map_err(|e| write_error(&path, e))
}

/// The `ImageU8` holding `pixels`, which must be a `uint8` array of shape
/// `(H, W, channels)` with `H` and `W` at least 1. It is copied in row order,
/// so an array in any memory layout is accepted.
fn to_image(
    pixels: &Bound<'_, PyAny>,
    channels: usize,
    function: &str,
    layout: &str,
) -> PyResult<ImageU8> {
    let array = pixels
        .extract::<PyReadonlyArrayDyn<'_, u8>>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{function} takes a uint8 numpy array, {layout}"
            ))
        })?;
    let (height, width) = match array.shape() {
        [h, w, c] if *c == channels && *h > 0 && *w > 0 => (*h, *w),
        shape => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{function} takes a non-empty {layout} array, got shape {shape:?}"
            )))
        }
    };
    let (Ok(width), Ok(height)) = (u32::try_from(width), u32::try_from(height)) else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{function}: a {width} x {height} image is too large to encode"
        )));
    };
    // `as_slice` alone would also accept a Fortran-ordered array and hand back
    // its column-major bytes; the macro takes the slice only in row order.
    let data = to_contiguous!(array).into_owned();
    Ok(ImageU8::new(width, height, channels as u32, data))
}

/// The Python exception for an image file that could not be written, naming
/// `path`: `ValueError` for a format, channel count, size or quality the
/// writer does not take, `FileNotFoundError` for a missing parent directory,
/// and `OSError` otherwise. An encoding error is a `ValueError`: the image is
/// encoded in memory, so the encoder fails only on an image it refuses, never
/// on the file.
fn write_error(path: &std::path::Path, e: image::ImageError) -> PyErr {
    let message = format!("could not write image {}: {e}", path.display());
    match &e {
        image::ImageError::Unsupported(_)
        | image::ImageError::Parameter(_)
        | image::ImageError::Encoding(_)
        | image::ImageError::Limits(_) => pyo3::exceptions::PyValueError::new_err(message),
        image::ImageError::IoError(io) if io.kind() == std::io::ErrorKind::NotFound => {
            pyo3::exceptions::PyFileNotFoundError::new_err(message)
        }
        _ => pyo3::exceptions::PyOSError::new_err(message),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(image_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(image_has_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(write_image_rgba, m)?)?;
    Ok(())
}
