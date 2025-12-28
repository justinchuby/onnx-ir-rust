// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for ONNX IR in Rust.
//!
//! This module provides Python bindings for the core ONNX IR library using PyO3.

use pyo3::prelude::*;

/// Python module initialization.
#[pymodule]
fn onnx_ir(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
