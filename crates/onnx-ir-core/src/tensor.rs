// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tensor protocols and implementations.
//!
//! This module provides various tensor implementations with different
//! storage backends.

use crate::enums::DataType;
use crate::metadata::MetadataStore;
use crate::shape::Shape;
use std::collections::HashMap;

/// Base trait for all tensor types.
pub trait TensorProtocol {
    /// Returns the name of the tensor.
    fn name(&self) -> Option<&str>;
    
    /// Returns the shape of the tensor.
    fn shape(&self) -> &Shape;
    
    /// Returns the data type of the tensor.
    fn dtype(&self) -> DataType;
    
    /// Returns the documentation string.
    fn doc_string(&self) -> Option<&str>;
    
    /// Returns the number of elements in the tensor.
    fn size(&self) -> usize;
    
    /// Returns the number of bytes in the tensor.
    fn nbytes(&self) -> usize;
}

/// A concrete tensor with in-memory data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: Option<String>,
    pub dtype: DataType,
    pub shape: Shape,
    pub data: Vec<u8>,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Tensor {
    /// Creates a new tensor with the given data type and shape.
    pub fn new(dtype: DataType, shape: Shape) -> Self {
        let size = shape.size().expect("Shape must not contain symbolic dimensions");
        let itemsize = dtype.itemsize().expect("Data type must have a known size");
        let nbytes = (size as f64 * itemsize).ceil() as usize;
        Self {
            name: None,
            dtype,
            shape,
            data: vec![0u8; nbytes],
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }

    /// Creates a new tensor from raw bytes.
    pub fn from_bytes(dtype: DataType, shape: Shape, data: Vec<u8>) -> Self {
        let size = shape.size().expect("Shape must not contain symbolic dimensions");
        let itemsize = dtype.itemsize().expect("Data type must have a known size");
        let expected_nbytes = (size as f64 * itemsize).ceil() as usize;
        assert_eq!(
            data.len(),
            expected_nbytes,
            "Data length {} does not match expected size {}",
            data.len(),
            expected_nbytes
        );
        Self {
            name: None,
            dtype,
            shape,
            data,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }

    /// Returns a reference to the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable reference to the raw bytes.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

impl TensorProtocol for Tensor {
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn doc_string(&self) -> Option<&str> {
        self.doc_string.as_deref()
    }

    fn size(&self) -> usize {
        self.shape.size().unwrap_or(0)
    }

    fn nbytes(&self) -> usize {
        self.data.len()
    }
}

/// An external tensor with data stored on disk.
#[derive(Debug)]
pub struct ExternalTensor {
    pub name: String,
    pub dtype: DataType,
    pub shape: Shape,
    pub location: String,
    pub offset: Option<usize>,
    pub length: Option<usize>,
    pub base_dir: String,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl ExternalTensor {
    /// Creates a new external tensor.
    pub fn new(
        name: impl Into<String>,
        dtype: DataType,
        shape: Shape,
        location: impl Into<String>,
        base_dir: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            location: location.into(),
            offset: None,
            length: None,
            base_dir: base_dir.into(),
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }

    /// Sets the offset and length for the external data.
    pub fn set_range(&mut self, offset: usize, length: usize) {
        self.offset = Some(offset);
        self.length = Some(length);
    }
}

impl TensorProtocol for ExternalTensor {
    fn name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        self.dtype
    }

    fn doc_string(&self) -> Option<&str> {
        self.doc_string.as_deref()
    }

    fn size(&self) -> usize {
        self.shape.size().unwrap_or(0)
    }

    fn nbytes(&self) -> usize {
        if let Some(len) = self.length {
            len
        } else {
            let itemsize = self.dtype.itemsize().unwrap_or(0.0);
            (self.size() as f64 * itemsize).ceil() as usize
        }
    }
}

/// A tensor for string data.
#[derive(Debug)]
pub struct StringTensor {
    pub name: Option<String>,
    pub shape: Shape,
    pub data: Vec<String>,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl StringTensor {
    /// Creates a new string tensor.
    pub fn new(shape: Shape, data: Vec<String>) -> Self {
        let size = shape.size().expect("Shape must not contain symbolic dimensions");
        assert_eq!(
            data.len(),
            size,
            "Data length {} does not match shape size {}",
            data.len(),
            size
        );
        Self {
            name: None,
            shape,
            data,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
    }

    /// Returns a reference to the string data.
    pub fn as_strings(&self) -> &[String] {
        &self.data
    }
}

impl TensorProtocol for StringTensor {
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        DataType::String
    }

    fn doc_string(&self) -> Option<&str> {
        self.doc_string.as_deref()
    }

    fn size(&self) -> usize {
        self.shape.size().unwrap_or(0)
    }

    fn nbytes(&self) -> usize {
        // For string tensors, return total byte size of all strings
        self.data.iter().map(|s| s.len()).sum()
    }
}

/// A lazy tensor that defers computation.
#[derive(Debug)]
pub struct LazyTensor {
    pub name: Option<String>,
    pub dtype: DataType,
    pub shape: Shape,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

/// A packed tensor for sub-byte types (2-bit, 4-bit).
#[derive(Debug)]
pub struct PackedTensor {
    pub name: Option<String>,
    pub dtype: DataType,
    pub shape: Shape,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::DataType;
    use crate::shape::Shape;

    #[test]
    fn test_tensor_new() {
        let shape = Shape::new(vec![2, 3]);
        let tensor = Tensor::new(DataType::Float, shape.clone());
        
        assert_eq!(tensor.dtype, DataType::Float);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.nbytes(), 6 * 4); // 6 elements * 4 bytes per float
    }

    #[test]
    fn test_tensor_from_bytes() {
        let shape = Shape::new(vec![2, 2]);
        let data = vec![0u8; 16]; // 4 floats * 4 bytes
        let tensor = Tensor::from_bytes(DataType::Float, shape.clone(), data);
        
        assert_eq!(tensor.dtype, DataType::Float);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.as_bytes().len(), 16);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_tensor_from_bytes_wrong_size() {
        let shape = Shape::new(vec![2, 2]);
        let data = vec![0u8; 8]; // Wrong size - should be 16
        let _tensor = Tensor::from_bytes(DataType::Float, shape, data);
    }

    #[test]
    fn test_tensor_protocol() {
        let shape = Shape::new(vec![3, 4]);
        let mut tensor = Tensor::new(DataType::Int32, shape);
        tensor.name = Some("test_tensor".to_string());
        tensor.doc_string = Some("A test tensor".to_string());
        
        assert_eq!(tensor.name(), Some("test_tensor"));
        assert_eq!(tensor.dtype(), DataType::Int32);
        assert_eq!(tensor.size(), 12);
        assert_eq!(tensor.nbytes(), 12 * 4); // 12 int32s * 4 bytes
        assert_eq!(tensor.doc_string(), Some("A test tensor"));
    }

    #[test]
    fn test_tensor_scalar() {
        let shape = Shape::scalar();
        let tensor = Tensor::new(DataType::Float, shape);
        
        assert_eq!(tensor.size(), 1);
        assert_eq!(tensor.nbytes(), 4);
    }

    #[test]
    fn test_external_tensor() {
        let shape = Shape::new(vec![10, 20]);
        let mut ext_tensor = ExternalTensor::new(
            "weights",
            DataType::Float,
            shape,
            "weights.bin",
            "/models",
        );
        
        assert_eq!(ext_tensor.name(), Some("weights"));
        assert_eq!(ext_tensor.dtype(), DataType::Float);
        assert_eq!(ext_tensor.size(), 200);
        
        ext_tensor.set_range(0, 800);
        assert_eq!(ext_tensor.offset, Some(0));
        assert_eq!(ext_tensor.length, Some(800));
        assert_eq!(ext_tensor.nbytes(), 800);
    }

    #[test]
    fn test_string_tensor() {
        let shape = Shape::new(vec![2, 2]);
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "foo".to_string(),
            "bar".to_string(),
        ];
        let tensor = StringTensor::new(shape, data);
        
        assert_eq!(tensor.dtype(), DataType::String);
        assert_eq!(tensor.size(), 4);
        assert_eq!(tensor.as_strings().len(), 4);
        // hello(5) + world(5) + foo(3) + bar(3) = 16 bytes
        assert_eq!(tensor.nbytes(), 16);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_string_tensor_wrong_size() {
        let shape = Shape::new(vec![2, 2]);
        let data = vec!["hello".to_string(), "world".to_string()]; // Wrong size
        let _tensor = StringTensor::new(shape, data);
    }
}
