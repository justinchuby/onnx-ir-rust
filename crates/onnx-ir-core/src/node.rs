// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Node representation with attributes and input/output management.

use crate::attribute::Attr;
use crate::metadata::MetadataStore;
use crate::value::Value;
use indexmap::IndexMap;
use std::collections::HashMap;

/// A node represents an invocation of an operation.
#[derive(Debug)]
pub struct Node {
    pub name: Option<String>,
    pub domain: String,
    pub op_type: String,
    pub overload: String,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub attributes: IndexMap<String, Attr>,
    pub version: Option<i32>,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Node {
    pub fn new(op_type: impl Into<String>) -> Self {
        Self {
            name: None,
            domain: String::new(),
            op_type: op_type.into(),
            overload: String::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: IndexMap::new(),
            version: None,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }
}
