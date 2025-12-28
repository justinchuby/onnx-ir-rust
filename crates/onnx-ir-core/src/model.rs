// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Top-level model container.

use crate::function::Function;
use crate::graph::Graph;
use crate::metadata::MetadataStore;
use std::collections::HashMap;

/// An ONNX model.
#[derive(Debug)]
pub struct Model {
    pub graph: Graph,
    pub ir_version: i64,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    pub domain: Option<String>,
    pub model_version: Option<i64>,
    pub doc_string: Option<String>,
    pub functions: HashMap<String, Function>,
    pub opset_imports: HashMap<String, i32>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Model {
    pub fn new(graph: Graph) -> Self {
        Self {
            graph,
            ir_version: 9, // Default to ONNX IR version 9
            producer_name: None,
            producer_version: None,
            domain: None,
            model_version: None,
            doc_string: None,
            functions: HashMap::new(),
            opset_imports: HashMap::new(),
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }
}
