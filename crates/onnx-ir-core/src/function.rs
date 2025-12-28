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

    /// Adds an input value to the function.
    pub fn add_input(&mut self, value: Value) {
        self.inputs.push(value);
    }

    /// Adds an output value to the function.
    pub fn add_output(&mut self, value: Value) {
        self.outputs.push(value);
    }

    /// Sets an opset import for the function.
    pub fn set_opset_import(&mut self, domain: impl Into<String>, version: i32) {
        self.opset_imports.insert(domain.into(), version);
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_new() {
        let func = Function::new("MyFunc", "com.example");
        assert_eq!(func.name, "MyFunc");
        assert_eq!(func.domain, "com.example");
        assert_eq!(func.num_inputs(), 0);
        assert_eq!(func.num_outputs(), 0);
    }

    #[test]
    fn test_function_inputs_outputs() {
        let mut func = Function::new("MyFunc", "com.example");
        func.add_input(Value::new("x"));
        func.add_input(Value::new("y"));
        func.add_output(Value::new("z"));
        
        assert_eq!(func.num_inputs(), 2);
        assert_eq!(func.num_outputs(), 1);
    }

    #[test]
    fn test_function_opset_imports() {
        let mut func = Function::new("MyFunc", "com.example");
        func.set_opset_import("", 18);
        
        assert_eq!(func.opset_imports.get(""), Some(&18));
    }
}
