// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for reading image files: the header-only reads
//! `image_dimensions` and `image_has_alpha`, and the full decodes
//! `read_image_rgb` and `read_image_rgba`.

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
use sfmtool_core::camera::image::ImageU8;

/// Read an image file's `(width, height)` from its header alone.
///
/// Decodes only the format header — JPEG `SOF`, PNG `IHDR`, and so on — never
/// the pixel data, so it stays cheap on large images. The dimensions are the
/// ones stored in the file; EXIF orientation is not applied.
#[pyfunction]
pub fn image_dimensions(path: PathBuf) -> PyResult<(u32, u32)> {
    image::image_dimensions(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
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
/// (`ImageU8::read_rgb`, the `image` crate), so the pixels equal the ones the
/// viewer, the bench and the photograph cache read, bit for bit. The EXIF
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(image_dimensions, m)?)?;
    m.add_function(wrap_pyfunction!(image_has_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(read_image_rgba, m)?)?;
    Ok(())
}
