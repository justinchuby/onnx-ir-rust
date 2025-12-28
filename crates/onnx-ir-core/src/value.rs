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
    // Note: Usage tracking (producer/consumer) will be added when Node is properly
    // defined with reference counting. For now, these are placeholders.
    // producer: RefCell<Option<Weak<RefCell<Node>>>>,
    // consumers: RefCell<Vec<(Weak<RefCell<Node>>, usize)>>,
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

    // Note: The following methods are stubs for future usage tracking implementation
    // when the ownership model is finalized. They are commented out to avoid
    // the forward declaration issue.

    // /// Returns the producer node of this value, if any.
    // pub fn producer(&self) -> Option<Rc<RefCell<Node>>> {
    //     self.producer.borrow().as_ref()?.upgrade()
    // }

    // /// Sets the producer node of this value.
    // pub fn set_producer(&self, producer: Option<Weak<RefCell<Node>>>) {
    //     *self.producer.borrow_mut() = producer;
    // }

    // /// Returns the consumer nodes and their input indices.
    // pub fn consumers(&self) -> Vec<(Rc<RefCell<Node>>, usize)> {
    //     self.consumers
    //         .borrow()
    //         .iter()
    //         .filter_map(|(weak, idx)| weak.upgrade().map(|rc| (rc, *idx)))
    //         .collect()
    // }

    // /// Adds a consumer node.
    // pub fn add_consumer(&self, consumer: Weak<RefCell<Node>>, input_index: usize) {
    //     self.consumers.borrow_mut().push((consumer, input_index));
    // }

    // /// Removes a consumer node.
    // pub fn remove_consumer(&self, consumer: &Weak<RefCell<Node>>) {
    //     self.consumers.borrow_mut().retain(|(c, _)| {
    //         !Weak::ptr_eq(c, consumer)
    //     });
    // }

    // /// Clears all consumers.
    // pub fn clear_consumers(&self) {
    //     self.consumers.borrow_mut().clear();
    // }

    // /// Returns the number of consumers (uses).
    // pub fn num_uses(&self) -> usize {
    //     self.consumers
    //         .borrow()
    //         .iter()
    //         .filter(|(weak, _)| weak.upgrade().is_some())
    //         .count()
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_new() {
        let value = Value::new("test_value");
        assert_eq!(value.name, "test_value");
        assert!(value.shape.is_none());
        assert!(value.type_.is_none());
    }
}
