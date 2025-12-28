// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX type system.
//!
//! This module defines the type representation for ONNX values, including
//! tensor types, sequence types, optional types, and sparse tensor types.

use crate::enums::DataType;
use std::fmt;

/// Base trait for ONNX types.
pub trait Type: fmt::Debug {
    /// Returns the denotation of the type.
    fn denotation(&self) -> Option<&str>;
    
    /// Returns the data type if this is a tensor-like type.
    fn dtype(&self) -> Option<DataType>;
}

/// A tensor type with element type.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    /// Element data type
    pub elem_type: DataType,
    /// Optional denotation
    pub denotation: Option<String>,
}

impl TensorType {
    pub fn new(elem_type: DataType) -> Self {
        Self {
            elem_type,
            denotation: None,
        }
    }
}

impl Type for TensorType {
    fn denotation(&self) -> Option<&str> {
        self.denotation.as_deref()
    }
    
    fn dtype(&self) -> Option<DataType> {
        Some(self.elem_type)
    }
}

/// A sparse tensor type.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensorType {
    pub elem_type: DataType,
    pub denotation: Option<String>,
}

impl SparseTensorType {
    pub fn new(elem_type: DataType) -> Self {
        Self {
            elem_type,
            denotation: None,
        }
    }
}

impl Type for SparseTensorType {
    fn denotation(&self) -> Option<&str> {
        self.denotation.as_deref()
    }
    
    fn dtype(&self) -> Option<DataType> {
        Some(self.elem_type)
    }
}

/// A sequence type with element type.
#[derive(Debug)]
pub struct SequenceType {
    pub elem_type: Box<dyn Type>,
    pub denotation: Option<String>,
}

/// An optional type.
#[derive(Debug)]
pub struct OptionalType {
    pub elem_type: Box<dyn Type>,
    pub denotation: Option<String>,
}
