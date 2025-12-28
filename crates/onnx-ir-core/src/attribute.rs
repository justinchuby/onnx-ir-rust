// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Attribute types for nodes and functions.

use crate::enums::{AttributeType, DataType};
use crate::graph::Graph;
use crate::tensor::Tensor;
use std::rc::Rc;

/// An ONNX attribute value.
#[derive(Debug, Clone)]
pub enum AttrValue {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(Box<Tensor>),
    Graph(Rc<Graph>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Box<Tensor>>),
    Graphs(Vec<Rc<Graph>>),
    DataType(DataType),
    DataTypes(Vec<DataType>),
}

impl AttrValue {
    /// Returns the attribute type of this value.
    pub fn attr_type(&self) -> AttributeType {
        match self {
            AttrValue::Float(_) => AttributeType::Float,
            AttrValue::Int(_) => AttributeType::Int,
            AttrValue::String(_) => AttributeType::String,
            AttrValue::Tensor(_) => AttributeType::Tensor,
            AttrValue::Graph(_) => AttributeType::Graph,
            AttrValue::Floats(_) => AttributeType::Floats,
            AttrValue::Ints(_) => AttributeType::Ints,
            AttrValue::Strings(_) => AttributeType::Strings,
            AttrValue::Tensors(_) => AttributeType::Tensors,
            AttrValue::Graphs(_) => AttributeType::Graphs,
            AttrValue::DataType(_) => AttributeType::TypeProto,
            AttrValue::DataTypes(_) => AttributeType::TypeProtos,
        }
    }
}

/// An ONNX attribute.
#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub value: AttrValue,
    pub doc_string: Option<String>,
}

impl Attr {
    pub fn new(name: impl Into<String>, value: AttrValue) -> Self {
        Self {
            name: name.into(),
            value,
            doc_string: None,
        }
    }

    /// Returns the attribute type.
    pub fn attr_type(&self) -> AttributeType {
        self.value.attr_type()
    }

    // Convenience constructors
    pub fn float(name: impl Into<String>, value: f32) -> Self {
        Self::new(name, AttrValue::Float(value))
    }

    pub fn int(name: impl Into<String>, value: i64) -> Self {
        Self::new(name, AttrValue::Int(value))
    }

    pub fn string(name: impl Into<String>, value: String) -> Self {
        Self::new(name, AttrValue::String(value))
    }

    pub fn floats(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self::new(name, AttrValue::Floats(values))
    }

    pub fn ints(name: impl Into<String>, values: Vec<i64>) -> Self {
        Self::new(name, AttrValue::Ints(values))
    }

    pub fn strings(name: impl Into<String>, values: Vec<String>) -> Self {
        Self::new(name, AttrValue::Strings(values))
    }
}

/// A reference attribute used in function definitions.
#[derive(Debug, Clone)]
pub struct RefAttr {
    pub name: String,
    pub ref_attr_name: String,
    pub type_: AttributeType,
    pub doc_string: Option<String>,
}

impl RefAttr {
    pub fn new(
        name: impl Into<String>,
        ref_attr_name: impl Into<String>,
        type_: AttributeType,
    ) -> Self {
        Self {
            name: name.into(),
            ref_attr_name: ref_attr_name.into(),
            type_,
            doc_string: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attr_float() {
        let attr = Attr::float("alpha", 0.5);
        assert_eq!(attr.name, "alpha");
        assert_eq!(attr.attr_type(), AttributeType::Float);
        match attr.value {
            AttrValue::Float(v) => assert_eq!(v, 0.5),
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn test_attr_int() {
        let attr = Attr::int("axis", 1);
        assert_eq!(attr.name, "axis");
        assert_eq!(attr.attr_type(), AttributeType::Int);
    }

    #[test]
    fn test_attr_string() {
        let attr = Attr::string("name", "test".to_string());
        assert_eq!(attr.name, "name");
        assert_eq!(attr.attr_type(), AttributeType::String);
    }

    #[test]
    fn test_attr_ints() {
        let attr = Attr::ints("shape", vec![1, 2, 3]);
        assert_eq!(attr.name, "shape");
        assert_eq!(attr.attr_type(), AttributeType::Ints);
    }
}
