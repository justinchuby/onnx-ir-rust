// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Value representation with usage tracking.
//!
//! This module implements the Value data structure which represents inputs and outputs
//! of nodes in the ONNX graph. Values track their producer and consumer nodes using
//! weak references to prevent ownership cycles.
//!
//! ## Ownership Model
//!
//! - Values are shared between nodes using `Rc<RefCell<Value>>`
//! - Producer/consumer tracking uses `Weak<RefCell<Node>>` to avoid cycles
//! - Interior mutability via `RefCell` allows usage tracking updates
//!
//! ## Usage Tracking
//!
//! Each value maintains:
//! - A weak reference to its producer node (if any)
//! - A list of weak references to consumer nodes with input indices
//!
//! This enables efficient graph transformations like `replace_all_uses_with`.

use crate::metadata::MetadataStore;
use crate::node::Node;
use crate::shape::Shape;
use crate::types::TensorType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};

/// A value represents an input or output of a node or graph.
///
/// Values track their producer and all consumer nodes to enable graph analysis
/// and transformation operations.
#[derive(Debug)]
pub struct Value {
    /// The name of the value (unique within a graph).
    pub name: String,
    /// The shape of the value, if known.
    pub shape: Option<Shape>,
    /// The type of the value, if known.
    pub type_: Option<TensorType>,
    /// Documentation string for this value.
    pub doc_string: Option<String>,
    /// Metadata properties that serialize to ONNX.
    pub metadata_props: HashMap<String, String>,
    /// Metadata store for IR passes (does not serialize).
    pub meta: MetadataStore,
    /// The node that produces this value (weak reference to prevent cycles).
    producer: RefCell<Option<Weak<RefCell<Node>>>>,
    /// The nodes that consume this value, along with the input index.
    /// Stored as (weak node reference, input index) pairs.
    consumers: RefCell<Vec<(Weak<RefCell<Node>>, usize)>>,
}

impl Value {
    /// Creates a new value with the given name.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Value;
    ///
    /// let value = Value::new("input");
    /// assert_eq!(value.name, "input");
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape: None,
            type_: None,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
            producer: RefCell::new(None),
            consumers: RefCell::new(Vec::new()),
        }
    }

    /// Returns the producer node of this value, if any.
    ///
    /// Returns `None` if:
    /// - The value has no producer (e.g., it's a graph input)
    /// - The producer node has been dropped
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Value;
    ///
    /// let value = Value::new("output");
    /// assert!(value.producer().is_none());
    /// ```
    pub fn producer(&self) -> Option<Rc<RefCell<Node>>> {
        self.producer.borrow().as_ref()?.upgrade()
    }

    /// Sets the producer node of this value.
    ///
    /// This should be called when a value is assigned to a node's output.
    /// Passing `None` clears the producer.
    pub fn set_producer(&self, producer: Option<Weak<RefCell<Node>>>) {
        *self.producer.borrow_mut() = producer;
    }

    /// Returns the consumer nodes and their input indices.
    ///
    /// Only returns consumers that are still alive (not dropped).
    /// Each tuple contains (node, input_index) where input_index indicates
    /// which input of the node uses this value.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Value;
    ///
    /// let value = Value::new("x");
    /// assert_eq!(value.consumers().len(), 0);
    /// ```
    pub fn consumers(&self) -> Vec<(Rc<RefCell<Node>>, usize)> {
        self.consumers
            .borrow()
            .iter()
            .filter_map(|(weak, idx)| weak.upgrade().map(|rc| (rc, *idx)))
            .collect()
    }

    /// Adds a consumer node at the specified input index.
    ///
    /// This should be called when a value is assigned to a node's input.
    /// The `input_index` indicates which input position uses this value.
    pub fn add_consumer(&self, consumer: Weak<RefCell<Node>>, input_index: usize) {
        self.consumers.borrow_mut().push((consumer, input_index));
    }

    /// Removes a specific consumer node.
    ///
    /// This should be called when a node no longer uses this value.
    pub fn remove_consumer(&self, consumer: &Weak<RefCell<Node>>) {
        self.consumers
            .borrow_mut()
            .retain(|(c, _)| !Weak::ptr_eq(c, consumer));
    }

    /// Clears all consumers.
    ///
    /// This should be called when replacing a value or removing it from the graph.
    pub fn clear_consumers(&self) {
        self.consumers.borrow_mut().clear();
    }

    /// Returns the number of active consumers (uses).
    ///
    /// Only counts consumers that are still alive (not dropped).
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Value;
    ///
    /// let value = Value::new("x");
    /// assert_eq!(value.num_uses(), 0);
    /// ```
    pub fn num_uses(&self) -> usize {
        self.consumers
            .borrow()
            .iter()
            .filter(|(weak, _)| weak.upgrade().is_some())
            .count()
    }

    /// Removes dead (dropped) consumer references.
    ///
    /// This is useful for cleanup when nodes are removed from the graph.
    /// It removes any weak references that can no longer be upgraded.
    pub fn prune_dead_consumers(&self) {
        self.consumers
            .borrow_mut()
            .retain(|(weak, _)| weak.upgrade().is_some());
    }

    /// Replaces all uses of this value with another value.
    ///
    /// This is a key operation for IR transformations. It updates all consumer
    /// nodes to use `replacement` instead of `self`.
    ///
    /// # Arguments
    ///
    /// * `replacement` - The value to replace this value with
    ///
    /// # Note
    ///
    /// This does not update the producer node's output list - that must be
    /// done separately if needed.
    pub fn replace_all_uses_with(&self, replacement: &Rc<RefCell<Value>>) {
        // Get all active consumers
        let consumers = self.consumers();

        // Update each consumer to use the replacement value
        for (node_rc, input_idx) in consumers {
            let mut node = node_rc.borrow_mut();
            if input_idx < node.inputs.len() {
                // Update the input to point to the replacement
                node.inputs[input_idx] = Rc::clone(replacement);
                
                // Add this node as a consumer of the replacement
                replacement
                    .borrow()
                    .add_consumer(Rc::downgrade(&node_rc), input_idx);
            }
        }

        // Clear this value's consumers since they now use the replacement
        self.clear_consumers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn test_value_new() {
        let value = Value::new("test_value");
        assert_eq!(value.name, "test_value");
        assert!(value.shape.is_none());
        assert!(value.type_.is_none());
        assert_eq!(value.num_uses(), 0);
        assert!(value.producer().is_none());
    }

    #[test]
    fn test_value_producer() {
        let value = Rc::new(RefCell::new(Value::new("output")));
        let mut node = Node::new("Add");
        node.outputs.push(Rc::clone(&value));

        let node_rc = Rc::new(RefCell::new(node));

        // Set producer
        value
            .borrow()
            .set_producer(Some(Rc::downgrade(&node_rc)));

        // Verify producer is set
        let producer = value.borrow().producer();
        assert!(producer.is_some());
        assert_eq!(producer.unwrap().borrow().op_type, "Add");

        // Clear producer
        value.borrow().set_producer(None);
        assert!(value.borrow().producer().is_none());
    }

    #[test]
    fn test_value_consumers() {
        let value = Rc::new(RefCell::new(Value::new("input")));

        // Create first consumer
        let mut node1 = Node::new("Add");
        node1.inputs.push(Rc::clone(&value));
        let node1_rc = Rc::new(RefCell::new(node1));

        // Add consumer
        value
            .borrow()
            .add_consumer(Rc::downgrade(&node1_rc), 0);

        assert_eq!(value.borrow().num_uses(), 1);

        // Create second consumer
        let mut node2 = Node::new("Mul");
        node2.inputs.push(Rc::clone(&value));
        let node2_rc = Rc::new(RefCell::new(node2));

        value
            .borrow()
            .add_consumer(Rc::downgrade(&node2_rc), 0);

        assert_eq!(value.borrow().num_uses(), 2);

        // Verify consumers
        let consumers = value.borrow().consumers();
        assert_eq!(consumers.len(), 2);
    }

    #[test]
    fn test_value_remove_consumer() {
        let value = Rc::new(RefCell::new(Value::new("x")));

        let mut node1 = Node::new("Add");
        node1.inputs.push(Rc::clone(&value));
        let node1_rc = Rc::new(RefCell::new(node1));
        let weak1 = Rc::downgrade(&node1_rc);

        let mut node2 = Node::new("Mul");
        node2.inputs.push(Rc::clone(&value));
        let node2_rc = Rc::new(RefCell::new(node2));
        let weak2 = Rc::downgrade(&node2_rc);

        value.borrow().add_consumer(weak1.clone(), 0);
        value.borrow().add_consumer(weak2.clone(), 0);

        assert_eq!(value.borrow().num_uses(), 2);

        // Remove first consumer
        value.borrow().remove_consumer(&weak1);
        assert_eq!(value.borrow().num_uses(), 1);

        // Remove second consumer
        value.borrow().remove_consumer(&weak2);
        assert_eq!(value.borrow().num_uses(), 0);
    }

    #[test]
    fn test_value_clear_consumers() {
        let value = Rc::new(RefCell::new(Value::new("x")));

        let mut node1 = Node::new("Add");
        node1.inputs.push(Rc::clone(&value));
        let node1_rc = Rc::new(RefCell::new(node1));

        let mut node2 = Node::new("Mul");
        node2.inputs.push(Rc::clone(&value));
        let node2_rc = Rc::new(RefCell::new(node2));

        value
            .borrow()
            .add_consumer(Rc::downgrade(&node1_rc), 0);
        value
            .borrow()
            .add_consumer(Rc::downgrade(&node2_rc), 0);

        assert_eq!(value.borrow().num_uses(), 2);

        value.borrow().clear_consumers();
        assert_eq!(value.borrow().num_uses(), 0);
    }

    #[test]
    fn test_value_prune_dead_consumers() {
        let value = Rc::new(RefCell::new(Value::new("x")));

        // Create a consumer and add it
        {
            let mut node = Node::new("Add");
            node.inputs.push(Rc::clone(&value));
            let node_rc = Rc::new(RefCell::new(node));

            value
                .borrow()
                .add_consumer(Rc::downgrade(&node_rc), 0);

            assert_eq!(value.borrow().num_uses(), 1);
            // node_rc goes out of scope and is dropped
        }

        // The weak reference is now dead
        assert_eq!(value.borrow().num_uses(), 0);

        // But the consumer list still has the dead reference
        assert_eq!(value.borrow().consumers.borrow().len(), 1);

        // Prune dead consumers
        value.borrow().prune_dead_consumers();
        assert_eq!(value.borrow().consumers.borrow().len(), 0);
    }

    #[test]
    fn test_value_replace_all_uses_with() {
        // Create original value
        let original = Rc::new(RefCell::new(Value::new("original")));
        let replacement = Rc::new(RefCell::new(Value::new("replacement")));

        // Create a consumer node
        let mut node = Node::new("Add");
        node.inputs.push(Rc::clone(&original));
        node.inputs.push(Rc::clone(&original)); // Use original twice
        let node_rc = Rc::new(RefCell::new(node));

        // Set up consumer tracking
        original
            .borrow()
            .add_consumer(Rc::downgrade(&node_rc), 0);
        original
            .borrow()
            .add_consumer(Rc::downgrade(&node_rc), 1);

        assert_eq!(original.borrow().num_uses(), 2);
        assert_eq!(replacement.borrow().num_uses(), 0);

        // Replace all uses
        original.borrow().replace_all_uses_with(&replacement);

        // Verify replacement
        assert_eq!(original.borrow().num_uses(), 0);
        assert_eq!(replacement.borrow().num_uses(), 2);

        // Verify node now uses replacement
        let node_borrowed = node_rc.borrow();
        assert_eq!(node_borrowed.inputs.len(), 2);
        assert_eq!(node_borrowed.inputs[0].borrow().name, "replacement");
        assert_eq!(node_borrowed.inputs[1].borrow().name, "replacement");
    }

    #[test]
    fn test_value_weak_reference_cleanup() {
        let value = Rc::new(RefCell::new(Value::new("x")));

        // Create multiple scopes to test weak reference cleanup
        {
            let mut node1 = Node::new("Add");
            node1.inputs.push(Rc::clone(&value));
            let node1_rc = Rc::new(RefCell::new(node1));
            value
                .borrow()
                .add_consumer(Rc::downgrade(&node1_rc), 0);

            assert_eq!(value.borrow().num_uses(), 1);
        } // node1_rc dropped here

        // Weak reference should now be invalid
        assert_eq!(value.borrow().num_uses(), 0);

        // Verify consumers list behavior
        let consumers = value.borrow().consumers();
        assert_eq!(consumers.len(), 0); // filter_map filters out dead weak refs
    }
}
