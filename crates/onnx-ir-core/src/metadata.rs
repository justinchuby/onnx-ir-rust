// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metadata storage for IR objects.
//!
//! This module provides a metadata store that can hold arbitrary key-value pairs
//! and supports marking keys as invalid, useful for graph transformation passes
//! that need to track when cached values need to be recomputed.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// A store for metadata about IR objects.
///
/// Metadata is stored as key-value pairs where keys are strings and values
/// can be any type. The store also supports marking keys as invalid, which is
/// useful when a pass wants to mark a key that needs to be recomputed.
///
/// # Examples
///
/// ```
/// use onnx_ir_core::MetadataStore;
///
/// let mut meta = MetadataStore::new();
/// meta.insert("shape", vec![1, 2, 3]);
/// assert_eq!(meta.get::<Vec<i32>>("shape"), Some(&vec![1, 2, 3]));
///
/// meta.invalidate("shape");
/// assert!(!meta.is_valid("shape"));
/// ```
#[derive(Default)]
pub struct MetadataStore {
    data: HashMap<String, Box<dyn std::any::Any>>,
    invalid_keys: HashSet<String>,
}

impl MetadataStore {
    /// Creates a new empty metadata store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            invalid_keys: HashSet::new(),
        }
    }

    /// Inserts a key-value pair into the store.
    ///
    /// If the key was previously marked as invalid, it will be marked as valid.
    pub fn insert<T: 'static>(&mut self, key: impl Into<String>, value: T) {
        let key = key.into();
        self.data.insert(key.clone(), Box::new(value));
        self.invalid_keys.remove(&key);
    }

    /// Gets a reference to the value associated with the key.
    ///
    /// Returns `None` if the key doesn't exist or the type doesn't match.
    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        self.data.get(key)?.downcast_ref::<T>()
    }

    /// Gets a mutable reference to the value associated with the key.
    ///
    /// Returns `None` if the key doesn't exist or the type doesn't match.
    pub fn get_mut<T: 'static>(&mut self, key: &str) -> Option<&mut T> {
        self.data.get_mut(key)?.downcast_mut::<T>()
    }

    /// Removes a key-value pair from the store.
    ///
    /// Returns the value if it existed and the type matches.
    pub fn remove<T: 'static>(&mut self, key: &str) -> Option<T> {
        let value = self.data.remove(key)?;
        self.invalid_keys.remove(key);
        value.downcast::<T>().ok().map(|boxed| *boxed)
    }

    /// Checks if the store contains a key.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Marks a key as invalid.
    ///
    /// This doesn't remove the value, but marks it as needing recomputation.
    /// Useful for transformation passes that invalidate cached computations.
    pub fn invalidate(&mut self, key: impl Into<String>) {
        self.invalid_keys.insert(key.into());
    }

    /// Checks if a value is valid.
    ///
    /// Returns true if the key exists and has not been marked as invalid.
    ///
    /// Note that default values (like `None`) are not necessarily invalid.
    /// For example, a shape that is unknown (`None`) may still be valid if
    /// shape inference has determined that the shape is unknown.
    ///
    /// Whether a value is valid is solely determined by the user that sets the value.
    pub fn is_valid(&self, key: &str) -> bool {
        self.data.contains_key(key) && !self.invalid_keys.contains(key)
    }

    /// Clears all metadata.
    pub fn clear(&mut self) {
        self.data.clear();
        self.invalid_keys.clear();
    }

    /// Returns the number of key-value pairs in the store.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an iterator over the keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
}

impl fmt::Debug for MetadataStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetadataStore")
            .field("keys", &self.data.keys().collect::<Vec<_>>())
            .field("invalid_keys", &self.invalid_keys)
            .finish()
    }
}

impl Clone for MetadataStore {
    fn clone(&self) -> Self {
        // Note: We can't clone the actual data since it's type-erased,
        // so we create a new empty store with the same invalid keys
        Self {
            data: HashMap::new(),
            invalid_keys: self.invalid_keys.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_store_basic() {
        let mut meta = MetadataStore::new();

        meta.insert("x", 42i32);
        assert_eq!(meta.get::<i32>("x"), Some(&42));
        assert!(meta.is_valid("x"));

        meta.insert("y", "hello".to_string());
        assert_eq!(meta.get::<String>("y"), Some(&"hello".to_string()));
    }

    #[test]
    fn test_metadata_store_invalidate() {
        let mut meta = MetadataStore::new();

        meta.insert("shape", vec![1, 2, 3]);
        assert!(meta.is_valid("shape"));

        meta.invalidate("shape");
        assert!(!meta.is_valid("shape"));
        assert!(meta.contains_key("shape"));

        // Re-inserting should mark as valid
        meta.insert("shape", vec![4, 5, 6]);
        assert!(meta.is_valid("shape"));
    }

    #[test]
    fn test_metadata_store_type_mismatch() {
        let mut meta = MetadataStore::new();

        meta.insert("x", 42i32);
        assert_eq!(meta.get::<i64>("x"), None);
    }

    #[test]
    fn test_metadata_store_remove() {
        let mut meta = MetadataStore::new();

        meta.insert("x", 42i32);
        meta.invalidate("x");

        let value = meta.remove::<i32>("x");
        assert_eq!(value, Some(42));
        assert!(!meta.contains_key("x"));
        // After removal, the key no longer exists, so it's not "invalid" anymore
        assert!(!meta.is_valid("x")); // is_valid returns false when key doesn't exist
    }

    #[test]
    fn test_metadata_store_clear() {
        let mut meta = MetadataStore::new();

        meta.insert("x", 42);
        meta.insert("y", 43);
        meta.invalidate("x");

        meta.clear();
        assert!(meta.is_empty());
        assert_eq!(meta.len(), 0);
    }
}
