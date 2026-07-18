// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Python wrapper for the `openjd-expr` integer range expression type.

use openjd_expr::RangeExpr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

/// A set of integer values described by a range expression.
///
/// Parses strings like `"1-10"`, `"1,3,5-7"`, or `"1-10:2"`, and supports
/// iteration, membership testing, `len()`, hashing, equality, and `str()`.
#[pyclass(name = "RangeExpr", module = "sfmtool.reconstruction", from_py_object)]
#[derive(Clone)]
pub struct PyRangeExpr {
    inner: RangeExpr,
}

#[pymethods]
impl PyRangeExpr {
    /// Parse a range expression from a string (e.g. `"1-10"`, `"1,3,5-7"`, `"1-10:2"`).
    #[new]
    fn new(range_str: &str) -> PyResult<Self> {
        RangeExpr::from_str(range_str)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Build a range expression from a list of integer values.
    #[staticmethod]
    fn from_list(values: Vec<i64>) -> PyResult<Self> {
        if values.is_empty() {
            return Err(PyValueError::new_err("range expression cannot be empty"));
        }
        Ok(Self {
            inner: RangeExpr::from_values(values),
        })
    }

    /// Total number of integers across all sub-ranges.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Test whether an integer is a member of the range expression.
    fn __contains__(&self, value: i64) -> bool {
        self.inner.contains(value)
    }

    fn __eq__(&self, other: &PyRangeExpr) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("RangeExpr(\"{}\")", self.inner)
    }

    /// Iterate over every integer in the range expression, in ascending order.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRangeExprIter {
        PyRangeExprIter {
            values: slf.inner.to_vec(),
            index: 0,
        }
    }
}

/// Iterator over the integer values of a [`PyRangeExpr`].
#[pyclass(module = "sfmtool.reconstruction")]
pub struct PyRangeExprIter {
    values: Vec<i64>,
    index: usize,
}

#[pymethods]
impl PyRangeExprIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<i64> {
        let value = self.values.get(self.index).copied();
        if value.is_some() {
            self.index += 1;
        }
        value
    }
}
