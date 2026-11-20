// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`ReadFailure`]: the first read of a photograph that failed inside a
//! bench kernel whose inner steps report their failures as values.

use std::sync::{Mutex, PoisonError};

use crate::camera::PhotographReadError;

/// The first read of a photograph that failed during one kernel call.
///
/// Some of the bench's kernels run inner steps whose failures are values
/// rather than errors: the grown plane's renders return plain samples, and
/// the track-at-pixel cascade and the far-field sweep turn a refused fit into
/// a refused member or a failed refit. A read that fails inside one of them
/// is kept here, and the kernel returns it in place of its result once the
/// inner step has run ([`Self::check`]), so a photograph that could not be
/// read fails the call rather than reading as a refusal.
#[derive(Debug, Default)]
pub(crate) struct ReadFailure(Mutex<Option<PhotographReadError>>);

impl ReadFailure {
    /// Keep `error` unless a read already failed.
    pub(crate) fn keep(&self, error: PhotographReadError) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(error);
    }

    /// The first read that failed, as the kernel's error.
    pub(crate) fn check(&self) -> Result<(), PhotographReadError> {
        match self.0.lock().unwrap_or_else(PoisonError::into_inner).take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
