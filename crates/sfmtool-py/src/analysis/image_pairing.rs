// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Binding for pairing the images of two reconstructions by file hash, then by
//! name (``sfmtool.analysis.pair_images``; see
//! ``specs/core/analysis/reconstruction-alignment.md``, "Pairing the images").

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use sfmtool_core::analysis::alignment::{pair_images as core_pair_images, ImageIdentity};

/// One side's identities, after checking that `hashes` has one entry per name
/// and that each hash given is 16 bytes.
fn identities<'a>(
    side: &str,
    names: &'a [String],
    hashes: &[Option<Bound<'_, PyBytes>>],
) -> PyResult<Vec<ImageIdentity<'a>>> {
    if names.len() != hashes.len() {
        return Err(PyValueError::new_err(format!(
            "{side}_names has {} entries and {side}_file_hashes has {}; give one hash \
             (or None) per image",
            names.len(),
            hashes.len()
        )));
    }
    names
        .iter()
        .zip(hashes)
        .enumerate()
        .map(|(i, (name, hash))| {
            let file_hash = match hash {
                None => None,
                Some(bytes) => Some(<[u8; 16]>::try_from(bytes.as_bytes()).map_err(|_| {
                    PyValueError::new_err(format!(
                        "{side}_file_hashes[{i}] has {} bytes; a file hash has 16",
                        bytes.as_bytes().len()
                    ))
                })?),
            };
            Ok(ImageIdentity {
                name: name.as_str(),
                file_hash,
            })
        })
        .collect()
}

/// `(k, 2)` ``uint32`` array of index pairs.
fn pairs_array<'py>(py: Python<'py>, pairs: &[(u32, u32)]) -> Bound<'py, PyArray2<u32>> {
    let flat: Vec<u32> = pairs.iter().flat_map(|&(s, t)| [s, t]).collect();
    numpy::ndarray::Array2::from_shape_vec((pairs.len(), 2), flat)
        .map(|a| PyArray2::from_owned_array(py, a))
        .expect("two entries per pair")
}

/// Pair the images of a source reconstruction with those of a target one.
///
/// Images are paired by the hash of the image file first, then by name:
///
/// 1. Images whose file hashes are known and equal pair, whatever their
///    names. A hash held by exactly one image on each side pairs those two; a
///    hash held by several images on a side (the same file under several
///    names) pairs only images of the same name among them.
/// 2. Each image still unpaired pairs with the still-unpaired image of the
///    same name on the other side, whether or not their hashes are known or
///    equal, so the same photographs at another resolution still pair. A
///    repeated name pairs once, its first unpaired image on each side.
///
/// ``SfmrReconstruction.find_image_file_hashes`` gives a reconstruction's
/// hashes. This is the pairing SfM Explorer's alignment uses.
///
/// Args:
///     source_names: The source's image names, one per image.
///     source_file_hashes: One 16-byte hash or ``None`` per source image.
///     target_names: The target's image names, one per image.
///     target_file_hashes: One 16-byte hash or ``None`` per target image.
///
/// Returns:
///     A dict: ``pairs``, a ``(k, 2)`` ``uint32`` array of
///     ``(source index, target index)`` ascending by source index;
///     ``by_file_hash`` and ``by_name``, how many pairs each step made;
///     ``by_name_different_files``, a ``(m, 2)`` ``uint32`` array of the
///     pairs made by name whose hashes are both known and differ, a subset
///     of ``pairs``; and ``summary``, the line ``images paired: 15 by file
///     hash, 2 by name`` (followed by ``, 2 of them different files`` when
///     there are any).
///
/// Raises:
///     ValueError: where a side's names and hashes differ in length, or a hash
///         is not 16 bytes.
#[pyfunction]
fn pair_images<'py>(
    py: Python<'py>,
    source_names: Vec<String>,
    source_file_hashes: Vec<Option<Bound<'py, PyBytes>>>,
    target_names: Vec<String>,
    target_file_hashes: Vec<Option<Bound<'py, PyBytes>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = identities("source", &source_names, &source_file_hashes)?;
    let target = identities("target", &target_names, &target_file_hashes)?;
    let pairing = core_pair_images(&source, &target);
    let out = PyDict::new(py);
    out.set_item("pairs", pairs_array(py, &pairing.pairs))?;
    out.set_item("by_file_hash", pairing.by_file_hash)?;
    out.set_item("by_name", pairing.by_name)?;
    out.set_item(
        "by_name_different_files",
        pairs_array(py, &pairing.by_name_different_files),
    )?;
    out.set_item("summary", pairing.summary())?;
    Ok(out)
}

// ── Registration ──────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pair_images, m)?)?;
    Ok(())
}
