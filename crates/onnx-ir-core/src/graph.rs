// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph container with mutation support.
//!
//! This module implements the Graph data structure which contains nodes, values,
//! and initializers. Graphs use a doubly-linked list for nodes to support safe
//! mutation during iteration.

use crate::linked_list::DoublyLinkedList;
use crate::metadata::MetadataStore;
use crate::node::Node;
use crate::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A computational graph.
///
/// Graphs own their nodes via a doubly-linked list and maintain references
/// to input, output, and initializer values.
#[derive(Debug)]
pub struct Graph {
    /// Optional name for this graph.
    pub name: Option<String>,
    /// Graph input values.
    pub inputs: Vec<Rc<RefCell<Value>>>,
    /// Graph output values.
    pub outputs: Vec<Rc<RefCell<Value>>>,
    /// Initializer values (constant tensors), indexed by name.
    pub initializers: HashMap<String, Rc<RefCell<Value>>>,
    /// Nodes in the graph (mutable linked list).
    pub nodes: DoublyLinkedList<Node>,
    /// Documentation string for this graph.
    pub doc_string: String,
    /// Opset imports (domain -> version).
    pub opset_imports: HashMap<String, i32>,
    /// Metadata properties that serialize to ONNX.
    pub metadata_props: HashMap<String, String>,
    /// Metadata store for IR passes (does not serialize).
    pub meta: MetadataStore,
}

impl Graph {
    /// Creates a new empty graph.
    ///
    /// # Example
    ///
    /// ```
    /// use onnx_ir_core::Graph;
    ///
    /// let graph = Graph::new();
    /// assert!(graph.is_empty());
    /// ```
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

    /// Appends a node to the end of the graph.
    pub fn append(&mut self, node: Node) {
        self.nodes.push_back(node);
    }

    /// Prepends a node to the beginning of the graph.
    pub fn prepend(&mut self, node: Node) {
        self.nodes.push_front(node);
    }

    /// Removes and returns the last node from the graph.
    pub fn pop_last(&mut self) -> Option<Node> {
        self.nodes.pop_back()
    }

    /// Removes and returns the first node from the graph.
    pub fn pop_first(&mut self) -> Option<Node> {
        self.nodes.pop_front()
    }

    /// Returns the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns an iterator over the nodes in the graph.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    /// Clears all nodes from the graph.
    pub fn clear_nodes(&mut self) {
        self.nodes.clear();
    }

    /// Adds an input value to the graph.
    pub fn add_input(&mut self, value: Rc<RefCell<Value>>) {
        self.inputs.push(value);
    }

    /// Adds an output value to the graph.
    pub fn add_output(&mut self, value: Rc<RefCell<Value>>) {
        self.outputs.push(value);
    }

    /// Adds an initializer value to the graph.
    ///
    /// The value's name is used as the key in the initializers map.
    pub fn add_initializer(&mut self, value: Rc<RefCell<Value>>) {
        let name = value.borrow().name.clone();
        self.initializers.insert(name, value);
    }

    /// Gets an initializer by name.
    pub fn get_initializer(&self, name: &str) -> Option<&Rc<RefCell<Value>>> {
        self.initializers.get(name)
    }

    /// Removes an initializer by name.
    pub fn remove_initializer(&mut self, name: &str) -> Option<Rc<RefCell<Value>>> {
        self.initializers.remove(name)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_new() {
        let graph = Graph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_graph_append() {
        let mut graph = Graph::new();
        let node1 = Node::new("Add");
        let node2 = Node::new("Mul");

        graph.append(node1);
        graph.append(node2);

        assert_eq!(graph.num_nodes(), 2);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_graph_prepend() {
        let mut graph = Graph::new();
        let node1 = Node::new("Add");
        let node2 = Node::new("Mul");

        graph.prepend(node1);
        graph.prepend(node2);

        assert_eq!(graph.num_nodes(), 2);
    }

    #[test]
    fn test_graph_pop() {
        let mut graph = Graph::new();
        graph.append(Node::new("Add"));
        graph.append(Node::new("Mul"));

        let last = graph.pop_last();
        assert!(last.is_some());
        assert_eq!(last.unwrap().op_type, "Mul");
        assert_eq!(graph.num_nodes(), 1);

        let first = graph.pop_first();
        assert!(first.is_some());
        assert_eq!(first.unwrap().op_type, "Add");
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_graph_clear() {
        let mut graph = Graph::new();
        graph.append(Node::new("Add"));
        graph.append(Node::new("Mul"));
        graph.append(Node::new("Sub"));

        assert_eq!(graph.num_nodes(), 3);

        graph.clear_nodes();

        assert_eq!(graph.num_nodes(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_graph_iter() {
        let mut graph = Graph::new();
        graph.append(Node::new("Add"));
        graph.append(Node::new("Mul"));
        graph.append(Node::new("Sub"));

        let op_types: Vec<_> = graph.iter_nodes().map(|n| n.op_type.as_str()).collect();

        assert_eq!(op_types, vec!["Add", "Mul", "Sub"]);
    }

    #[test]
    fn test_graph_inputs_outputs() {
        let mut graph = Graph::new();

        let input = Rc::new(RefCell::new(Value::new("input")));
        let output = Rc::new(RefCell::new(Value::new("output")));

        graph.add_input(input);
        graph.add_output(output);

        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.inputs[0].borrow().name, "input");
        assert_eq!(graph.outputs[0].borrow().name, "output");
    }

    #[test]
    fn test_graph_initializers() {
        let mut graph = Graph::new();

        let init = Rc::new(RefCell::new(Value::new("weights")));
        graph.add_initializer(Rc::clone(&init));

        assert_eq!(graph.initializers.len(), 1);
        assert!(graph.get_initializer("weights").is_some());

        let removed = graph.remove_initializer("weights");
        assert!(removed.is_some());
        assert_eq!(graph.initializers.len(), 0);
    }
}
