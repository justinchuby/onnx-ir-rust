// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Node representation with attributes and input/output management.
//!
//! This module implements the Node data structure which represents operations
//! in the ONNX graph. Nodes hold references to their input and output values
//! using `Rc<RefCell<Value>>` for shared ownership.

use crate::attribute::Attr;
use crate::metadata::MetadataStore;
use crate::value::Value;
use indexmap::IndexMap;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A node represents an invocation of an operation.
///
/// Nodes hold shared references to values to enable graph analysis and
/// transformation. The values track their producer and consumers.
#[derive(Debug)]
pub struct Node {
    /// Optional name for this node (not required to be unique).
    pub name: Option<String>,
    /// The domain of the operator (empty string for default ONNX domain).
    pub domain: String,
    /// The type of operation this node performs.
    pub op_type: String,
    /// The overload name for the operation (for function overloading).
    pub overload: String,
    /// Input values (shared references).
    pub inputs: Vec<Rc<RefCell<Value>>>,
    /// Output values (shared references).
    pub outputs: Vec<Rc<RefCell<Value>>>,
    /// Attributes of the node (order-preserving).
    pub attributes: IndexMap<String, Attr>,
    /// Opset version for this node (optional).
    pub version: Option<i32>,
    /// Documentation string for this node.
    pub doc_string: Option<String>,
    /// Metadata properties that serialize to ONNX.
    pub metadata_props: HashMap<String, String>,
    /// Metadata store for IR passes (does not serialize).
    pub meta: MetadataStore,
}

impl Node {
    /// Creates a new node with the given operation type.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Node;
    ///
    /// let node = Node::new("Add");
    /// assert_eq!(node.op_type, "Add");
    /// ```
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
    ///
    /// **Note**: This method does not automatically set up producer/consumer tracking.
    /// Use the helper function `node_add_input()` with the node's `Rc<RefCell<Node>>` 
    /// wrapper for automatic tracking, or call this method and manually set up tracking.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::{Node, Value};
    /// use std::rc::Rc;
    /// use std::cell::RefCell;
    ///
    /// let mut node = Node::new("Add");
    /// let value = Rc::new(RefCell::new(Value::new("input")));
    /// node.add_input(value);
    /// assert_eq!(node.num_inputs(), 1);
    /// ```
    pub fn add_input(&mut self, value: Rc<RefCell<Value>>) {
        self.inputs.push(value);
    }

    /// Adds an output value to the node.
    ///
    /// **Note**: This method does not automatically set up producer/consumer tracking.
    /// Use the helper function `node_add_output()` with the node's `Rc<RefCell<Node>>` 
    /// wrapper for automatic tracking, or call this method and manually set up tracking.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::{Node, Value};
    /// use std::rc::Rc;
    /// use std::cell::RefCell;
    ///
    /// let mut node = Node::new("Add");
    /// let value = Rc::new(RefCell::new(Value::new("output")));
    /// node.add_output(value);
    /// assert_eq!(node.num_outputs(), 1);
    /// ```
    pub fn add_output(&mut self, value: Rc<RefCell<Value>>) {
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

/// Helper function to add an input value to a node with automatic consumer tracking.
///
/// This function adds the input value to the node and automatically registers
/// the node as a consumer of the value, matching the behavior of onnx/ir-py.
///
/// # Arguments
///
/// * `node` - The node wrapped in `Rc<RefCell<Node>>`
/// * `value` - The input value to add
///
/// # Example
///
/// ```
/// use onnx_ir_core::{Node, Value, node_add_input};
/// use std::rc::Rc;
/// use std::cell::RefCell;
///
/// let node = Rc::new(RefCell::new(Node::new("Add")));
/// let value = Rc::new(RefCell::new(Value::new("input")));
/// node_add_input(&node, &value);
///
/// assert_eq!(node.borrow().num_inputs(), 1);
/// assert_eq!(value.borrow().num_uses(), 1);
/// ```
pub fn node_add_input(node: &Rc<RefCell<Node>>, value: &Rc<RefCell<Value>>) {
    let input_index = node.borrow().inputs.len();
    node.borrow_mut().add_input(Rc::clone(value));
    value.borrow().add_consumer(Rc::downgrade(node), input_index);
}

/// Helper function to add an output value to a node with automatic producer tracking.
///
/// This function adds the output value to the node and automatically sets
/// the node as the producer of the value, matching the behavior of onnx/ir-py.
///
/// # Arguments
///
/// * `node` - The node wrapped in `Rc<RefCell<Node>>`
/// * `value` - The output value to add
///
/// # Example
///
/// ```
/// use onnx_ir_core::{Node, Value, node_add_output};
/// use std::rc::Rc;
/// use std::cell::RefCell;
///
/// let node = Rc::new(RefCell::new(Node::new("Add")));
/// let value = Rc::new(RefCell::new(Value::new("output")));
/// node_add_output(&node, &value);
///
/// assert_eq!(node.borrow().num_outputs(), 1);
/// assert!(value.borrow().producer().is_some());
/// ```
pub fn node_add_output(node: &Rc<RefCell<Node>>, value: &Rc<RefCell<Value>>) {
    node.borrow_mut().add_output(Rc::clone(value));
    value.borrow().set_producer(Some(Rc::downgrade(node)));
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
        let x = Rc::new(RefCell::new(Value::new("x")));
        let y = Rc::new(RefCell::new(Value::new("y")));
        let z = Rc::new(RefCell::new(Value::new("z")));

        node.add_input(x);
        node.add_input(y);
        node.add_output(z);

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
