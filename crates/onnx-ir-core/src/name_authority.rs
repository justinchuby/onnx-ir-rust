// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Name authority for managing unique names in the IR.
//!
//! This module provides functionality for generating and tracking unique names
//! for values and nodes in the graph.

use std::collections::HashSet;

/// Authority for giving names to values and nodes in the IR.
///
/// The names are generated in the format `val_{counter}` for values and
/// `node_{op_type}_{counter}` for nodes. The counter is incremented each time
/// a new value or node is named.
///
/// This class keeps track of the names it has generated and existing names
/// in the graph to prevent producing duplicated names.
///
/// # Note
///
/// Once a name is tracked, it will not be made available even if the node/value
/// is removed from the graph. It is possible to improve this behavior by keeping
/// track of the names that are no longer used, but it is not implemented yet.
///
/// However, if a value/node already has a name when added to the graph,
/// the name authority will not change its name. It is the responsibility of the
/// user to ensure that the names are unique (typically by running a name-fixing
/// pass on the graph).
///
/// # Examples
///
/// ```
/// use onnx_ir_core::name_authority::NameAuthority;
///
/// let mut authority = NameAuthority::new();
/// let name1 = authority.unique_value_name(None);
/// let name2 = authority.unique_value_name(None);
/// assert_ne!(name1, name2);
///
/// let node_name = authority.unique_node_name("Add", None);
/// assert!(node_name.contains("Add"));
/// ```
pub struct NameAuthority {
    value_counter: usize,
    node_counter: usize,
    value_names: HashSet<String>,
    node_names: HashSet<String>,
}

impl NameAuthority {
    /// Creates a new name authority.
    pub fn new() -> Self {
        Self {
            value_counter: 0,
            node_counter: 0,
            value_names: HashSet::new(),
            node_names: HashSet::new(),
        }
    }

    /// Generates a unique name for a value.
    ///
    /// If `preferred_name` is provided and unique, it will be used.
    /// Otherwise, a new name in the format `val_{counter}` is generated.
    pub fn unique_value_name(&mut self, preferred_name: Option<&str>) -> String {
        if let Some(name) = preferred_name {
            if !self.value_names.contains(name) {
                self.value_names.insert(name.to_string());
                return name.to_string();
            }
        }

        loop {
            let name = format!("val_{}", self.value_counter);
            self.value_counter += 1;
            if !self.value_names.contains(&name) {
                self.value_names.insert(name.clone());
                return name;
            }
        }
    }

    /// Generates a unique name for a node.
    ///
    /// If `preferred_name` is provided and unique, it will be used.
    /// Otherwise, a new name in the format `node_{op_type}_{counter}` is generated.
    pub fn unique_node_name(&mut self, op_type: &str, preferred_name: Option<&str>) -> String {
        if let Some(name) = preferred_name {
            if !self.node_names.contains(name) {
                self.node_names.insert(name.to_string());
                return name.to_string();
            }
        }

        loop {
            let name = format!("node_{}_{}", op_type, self.node_counter);
            self.node_counter += 1;
            if !self.node_names.contains(&name) {
                self.node_names.insert(name.clone());
                return name;
            }
        }
    }

    /// Registers an existing value name to prevent future conflicts.
    ///
    /// Returns true if the name was newly registered, false if it was already tracked.
    pub fn register_value_name(&mut self, name: String) -> bool {
        self.value_names.insert(name)
    }

    /// Registers an existing node name to prevent future conflicts.
    ///
    /// Returns true if the name was newly registered, false if it was already tracked.
    pub fn register_node_name(&mut self, name: String) -> bool {
        self.node_names.insert(name)
    }

    /// Checks if a value name is already registered.
    pub fn has_value_name(&self, name: &str) -> bool {
        self.value_names.contains(name)
    }

    /// Checks if a node name is already registered.
    pub fn has_node_name(&self, name: &str) -> bool {
        self.node_names.contains(name)
    }

    /// Returns the number of registered value names.
    pub fn value_name_count(&self) -> usize {
        self.value_names.len()
    }

    /// Returns the number of registered node names.
    pub fn node_name_count(&self) -> usize {
        self.node_names.len()
    }

    /// Clears all registered names.
    pub fn clear(&mut self) {
        self.value_counter = 0;
        self.node_counter = 0;
        self.value_names.clear();
        self.node_names.clear();
    }
}

impl Default for NameAuthority {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_value_name() {
        let mut authority = NameAuthority::new();

        let name1 = authority.unique_value_name(None);
        let name2 = authority.unique_value_name(None);

        assert_eq!(name1, "val_0");
        assert_eq!(name2, "val_1");
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_unique_value_name_preferred() {
        let mut authority = NameAuthority::new();

        let name1 = authority.unique_value_name(Some("my_val"));
        assert_eq!(name1, "my_val");

        let name2 = authority.unique_value_name(Some("my_val"));
        assert_eq!(name2, "val_0");
    }

    #[test]
    fn test_unique_node_name() {
        let mut authority = NameAuthority::new();

        let name1 = authority.unique_node_name("Add", None);
        let name2 = authority.unique_node_name("Mul", None);

        assert_eq!(name1, "node_Add_0");
        assert_eq!(name2, "node_Mul_1");
    }

    #[test]
    fn test_unique_node_name_preferred() {
        let mut authority = NameAuthority::new();

        let name1 = authority.unique_node_name("Add", Some("my_add"));
        assert_eq!(name1, "my_add");

        let name2 = authority.unique_node_name("Add", Some("my_add"));
        assert_eq!(name2, "node_Add_0");
    }

    #[test]
    fn test_register_names() {
        let mut authority = NameAuthority::new();

        assert!(authority.register_value_name("existing".to_string()));
        assert!(!authority.register_value_name("existing".to_string()));
        assert!(authority.has_value_name("existing"));

        let new_name = authority.unique_value_name(Some("existing"));
        assert_eq!(new_name, "val_0");
    }

    #[test]
    fn test_clear() {
        let mut authority = NameAuthority::new();

        authority.unique_value_name(None);
        authority.unique_node_name("Add", None);

        assert_eq!(authority.value_name_count(), 1);
        assert_eq!(authority.node_name_count(), 1);

        authority.clear();

        assert_eq!(authority.value_name_count(), 0);
        assert_eq!(authority.node_name_count(), 0);

        // Counters should be reset
        let name = authority.unique_value_name(None);
        assert_eq!(name, "val_0");
    }
}
