// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph container with mutation support.

use crate::linked_list::DoublyLinkedList;
use crate::metadata::MetadataStore;
use crate::node::Node;
use crate::value::Value;
use std::collections::HashMap;

/// A computational graph.
#[derive(Debug)]
pub struct Graph {
    pub name: Option<String>,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub initializers: HashMap<String, Value>,
    pub nodes: DoublyLinkedList<Node>,
    pub doc_string: String,
    pub opset_imports: HashMap<String, i32>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            name: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
            nodes: DoublyLinkedList::new(),
            doc_string: String::new(),
            opset_imports: HashMap::new(),
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A read-only view of a graph.
#[derive(Debug)]
pub struct GraphView {
    pub name: Option<String>,
    pub doc_string: String,
}
