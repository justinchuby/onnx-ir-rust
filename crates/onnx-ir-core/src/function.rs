// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Function definition support.

use crate::metadata::MetadataStore;
use crate::value::Value;
use std::collections::HashMap;

/// An ONNX function.
#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub domain: String,
    pub overload: String,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub doc_string: String,
    pub opset_imports: HashMap<String, i32>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Function {
    pub fn new(name: impl Into<String>, domain: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            domain: domain.into(),
            overload: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            doc_string: String::new(),
            opset_imports: HashMap::new(),
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }
}
