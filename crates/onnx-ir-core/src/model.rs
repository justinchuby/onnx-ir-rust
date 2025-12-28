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

    /// Adds a function to the model.
    pub fn add_function(&mut self, function: Function) {
        let key = format!("{}::{}", function.domain, function.name);
        self.functions.insert(key, function);
    }

    /// Gets a function by domain and name.
    pub fn get_function(&self, domain: &str, name: &str) -> Option<&Function> {
        let key = format!("{}::{}", domain, name);
        self.functions.get(&key)
    }

    /// Sets an opset import.
    pub fn set_opset_import(&mut self, domain: impl Into<String>, version: i32) {
        self.opset_imports.insert(domain.into(), version);
    }

    /// Gets an opset version for a domain.
    pub fn get_opset_version(&self, domain: &str) -> Option<i32> {
        self.opset_imports.get(domain).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_model_new() {
        let graph = Graph::new();
        let model = Model::new(graph);
        
        assert_eq!(model.ir_version, 9);
        assert!(model.functions.is_empty());
    }

    #[test]
    fn test_model_opset_imports() {
        let graph = Graph::new();
        let mut model = Model::new(graph);
        
        model.set_opset_import("", 18);
        model.set_opset_import("com.example", 1);
        
        assert_eq!(model.get_opset_version(""), Some(18));
        assert_eq!(model.get_opset_version("com.example"), Some(1));
        assert_eq!(model.get_opset_version("unknown"), None);
    }

    #[test]
    fn test_model_functions() {
        let graph = Graph::new();
        let mut model = Model::new(graph);
        
        let func = Function::new("MyFunc", "com.example");
        model.add_function(func);
        
        assert!(model.get_function("com.example", "MyFunc").is_some());
        assert!(model.get_function("com.example", "Unknown").is_none());
    }
}
