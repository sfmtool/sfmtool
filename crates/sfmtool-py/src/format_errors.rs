// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The one mapping from the file-format crates' errors to Python exceptions.
//!
//! The `.sfmr`, `.sift`, `.matches`, `.camrig` and `.kdf` error enums share
//! their variants, and every binding that calls a reader, writer or verifier of
//! one of those formats raises through [`read_err`] or [`write_err`] (or, for
//! `.kdf`, [`crate::spatial::kdf::to_py_err`], which maps the variants only
//! `.kdf` has and hands the rest to [`fault_to_py`]). A variant therefore
//! raises the same exception class whichever format it came from: an I/O
//! failure is the `OSError` subclass for its kind, a file that cannot be parsed
//! or breaks a rule of its format is an `OSError`, and data a writer was handed
//! that breaks a rule of the format is a `ValueError`. The full table is in
//! `specs/python-bindings.md` § "Format errors".

use pyo3::PyErr;

/// What a format error says went wrong, which decides its Python exception.
pub(crate) enum FormatFault {
    /// A filesystem or stream call failed, with this kind.
    Io(std::io::ErrorKind),
    /// The ZIP container or a JSON entry could not be read or written.
    Container,
    /// The data broke a rule of the format. On a read the file is damaged; on
    /// a write the caller handed over data the format cannot hold.
    Content,
    /// The call asked for something the data cannot give, whichever way the
    /// data was going.
    Request,
}

/// A format crate's error, classified for [`read_err`] and [`write_err`].
pub(crate) trait FormatError: std::fmt::Display {
    fn fault(&self) -> FormatFault;
}

/// Which way the data was going when a format error happened.
#[derive(Clone, Copy)]
pub(crate) enum Direction {
    /// Reading or verifying a file.
    Read,
    /// Writing the caller's data, or building a value from it.
    Write,
}

/// The Python exception for `e`, raised by a binding that reads or verifies a
/// file.
pub(crate) fn read_err<E: FormatError>(e: E) -> PyErr {
    fault_to_py(e.fault(), e.to_string(), Direction::Read)
}

/// The Python exception for `e`, raised by a binding that writes the caller's
/// data, or builds a value from it.
pub(crate) fn write_err<E: FormatError>(e: E) -> PyErr {
    fault_to_py(e.fault(), e.to_string(), Direction::Write)
}

/// The Python exception for a fault of `fault`, carrying `message`.
pub(crate) fn fault_to_py(fault: FormatFault, message: String, direction: Direction) -> PyErr {
    match (fault, direction) {
        // PyO3 picks the `OSError` subclass for the kind, as `open` does.
        (FormatFault::Io(kind), _) => PyErr::from(std::io::Error::new(kind, message)),
        (FormatFault::Container, _) | (FormatFault::Content, Direction::Read) => {
            pyo3::exceptions::PyOSError::new_err(message)
        }
        (FormatFault::Content, Direction::Write) | (FormatFault::Request, _) => {
            pyo3::exceptions::PyValueError::new_err(message)
        }
    }
}

/// Implement [`FormatError`] for an enum with the variants the four
/// `*-format` crates other than `.kdf` share, plus any variants of its own
/// listed after `content:` (a rule of the format broken, as `InvalidFormat`
/// is) or `request:` (something the data cannot give).
macro_rules! impl_format_error {
    ($error:ty $(, content: $content:ident)* $(, request: $request:ident)*) => {
        impl FormatError for $error {
            fn fault(&self) -> FormatFault {
                match self {
                    Self::Io(e) => FormatFault::Io(e.kind()),
                    Self::IoPath { source, .. } => FormatFault::Io(source.kind()),
                    Self::Zip(_) | Self::Json(_) => FormatFault::Container,
                    Self::InvalidFormat(_) | Self::ShapeMismatch(_) => FormatFault::Content,
                    $(Self::$content(_) => FormatFault::Content,)*
                    $(Self::$request(_) => FormatFault::Request,)*
                }
            }
        }
    };
}

impl_format_error!(sfmtool_sfmr_format::SfmrError, request: Unsupported);
impl_format_error!(sfmtool_sift_format::SiftError, content: InvalidMetadata);
impl_format_error!(sfmtool_matches_format::MatchesError);
impl_format_error!(sfmtool_camrig_format::CamRigError);
