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
        
        let op_types: Vec<_> = graph.iter_nodes()
            .map(|n| n.op_type.as_str())
            .collect();
        
        assert_eq!(op_types, vec!["Add", "Mul", "Sub"]);
    }
}
