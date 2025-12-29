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

    /// Adds an input value to the node.
    pub fn add_input(&mut self, value: Value) {
        self.inputs.push(value);
    }

    /// Adds an output value to the node.
    pub fn add_output(&mut self, value: Value) {
        self.outputs.push(value);
    }

    /// Sets an attribute on the node.
    pub fn set_attribute(&mut self, attr: Attr) {
        self.attributes.insert(attr.name.clone(), attr);
    }

    /// Gets an attribute by name.
    pub fn get_attribute(&self, name: &str) -> Option<&Attr> {
        self.attributes.get(name)
    }

    /// Removes an attribute by name.
    pub fn remove_attribute(&mut self, name: &str) -> Option<Attr> {
        self.attributes.shift_remove(name)
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Returns true if the node has any attributes.
    pub fn has_attributes(&self) -> bool {
        !self.attributes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::Attr;

    #[test]
    fn test_node_new() {
        let node = Node::new("Add");
        assert_eq!(node.op_type, "Add");
        assert_eq!(node.num_inputs(), 0);
        assert_eq!(node.num_outputs(), 0);
    }

    #[test]
    fn test_node_add_inputs_outputs() {
        let mut node = Node::new("Add");
        node.add_input(Value::new("x"));
        node.add_input(Value::new("y"));
        node.add_output(Value::new("z"));

        assert_eq!(node.num_inputs(), 2);
        assert_eq!(node.num_outputs(), 1);
    }

    #[test]
    fn test_node_attributes() {
        let mut node = Node::new("Conv");

        assert!(!node.has_attributes());

        node.set_attribute(Attr::int("axis", 0));
        node.set_attribute(Attr::float("alpha", 0.5));

        assert!(node.has_attributes());
        assert_eq!(node.attributes.len(), 2);

        let attr = node.get_attribute("axis");
        assert!(attr.is_some());

        let removed = node.remove_attribute("axis");
        assert!(removed.is_some());
        assert_eq!(node.attributes.len(), 1);
    }
}
