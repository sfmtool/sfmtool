// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Bindings for what `sfm explorer-call` checks before it calls a running SfM
//! Explorer's MCP endpoint: the viewer's tool catalog, and the parse its
//! `tools/call` runs on a call's arguments. Both come from the `sfm-explorer`
//! crate this extension also runs the viewer from, so they are the catalog and
//! the parser of the viewer `sfm explorer` starts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Decode serde JSON into Python objects through the standard library's
/// `json`, which is what produced it on the way in and is the reading of JSON
/// the command uses for the rest of the reply.
fn to_python<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?
        .call_method1("loads", (value.to_string(),))
}

/// Every tool the viewer advertises, in the order its `tools/list` reports
/// them, as a list of dicts.
///
/// Each dict has `name`; `kind`, which is `"read"`, `"write"`, `"save"` (the
/// tool writes a file) or `"input"` (it presses what a person could press);
/// and `advertised`, the tool's entry in a `tools/list` reply as this build of
/// the viewer serves it, with `name`, `description`, `inputSchema` and
/// `annotations`.
#[pyfunction]
fn tool_catalog(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    sfm_explorer::tool_catalog()
        .into_iter()
        .map(|tool| {
            let entry = PyDict::new(py);
            entry.set_item("name", tool.name)?;
            entry.set_item("kind", tool.kind)?;
            entry.set_item("advertised", to_python(py, &tool.advertised)?)?;
            Ok(entry)
        })
        .collect()
}

/// Return the message the viewer refuses a call of the tool `tool` with
/// `arguments` with, or `None` when the arguments parse.
///
/// `arguments` is a dict, as `json.loads` gives it, or `None` for no
/// arguments. The check is the viewer's own `tools/call` parse, so the message
/// is the text of the JSON-RPC error the endpoint would send, including for a
/// tool name it does not know. `None` does not promise the call succeeds: the
/// running viewer can still refuse it for what it finds in the scene, such as
/// a label that names no loaded reconstruction.
///
/// Raises `ValueError` when `arguments` is not a dict or `None` or holds a
/// NaN or an infinity, and `TypeError` when it holds a value `json.dumps`
/// cannot encode.
#[pyfunction]
#[pyo3(signature = (tool, arguments=None))]
fn validate_tool_call(
    tool: &str,
    arguments: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let arguments = match arguments {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => {
            if !value.is_instance_of::<PyDict>() {
                return Err(PyValueError::new_err(
                    "arguments must be a dict, as the JSON object of a tools/call is",
                ));
            }
            let keywords = PyDict::new(value.py());
            keywords.set_item("allow_nan", false)?;
            let text: String = value
                .py()
                .import("json")?
                .call_method("dumps", (value,), Some(&keywords))?
                .extract()?;
            match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(serde_json::Value::Object(map)) => Some(map),
                Ok(_) => unreachable!("a dict encodes as a JSON object"),
                Err(error) => {
                    return Err(PyValueError::new_err(format!(
                        "arguments are not JSON: {error}"
                    )))
                }
            }
        }
    };
    Ok(sfm_explorer::validate_tool_call(tool, arguments.as_ref()).err())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tool_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tool_call, m)?)?;
    Ok(())
}
