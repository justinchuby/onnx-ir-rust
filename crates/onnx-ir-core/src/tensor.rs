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
#[derive(Debug)]
pub struct Tensor {
    pub name: Option<String>,
    pub dtype: DataType,
    pub shape: Shape,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}

impl Tensor {
    pub fn new(dtype: DataType, shape: Shape) -> Self {
        Self {
            name: None,
            dtype,
            shape,
            doc_string: None,
            metadata_props: HashMap::new(),
            meta: MetadataStore::new(),
        }
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

/// A tensor for string data.
#[derive(Debug)]
pub struct StringTensor {
    pub name: Option<String>,
    pub shape: Shape,
    pub doc_string: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
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
