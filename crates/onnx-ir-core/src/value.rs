// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Value representation with usage tracking.

use crate::metadata::MetadataStore;
use crate::shape::Shape;
use crate::types::TensorType;
use std::collections::HashMap;

/// A value represents an input or output of a node or graph.
#[derive(Debug)]
pub struct Value {
    pub name: String,
    pub shape: Option<Shape>,
    pub type_: Option<TensorType>,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Value {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape: None,
            type_: None,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }
}
