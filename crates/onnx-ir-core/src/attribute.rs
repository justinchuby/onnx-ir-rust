// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Attribute types for nodes and functions.

use crate::enums::AttributeType;

/// An ONNX attribute.
#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub type_: AttributeType,
    pub doc_string: Option<String>,
}

impl Attr {
    pub fn new(name: impl Into<String>, type_: AttributeType) -> Self {
        Self {
            name: name.into(),
            type_,
            doc_string: None,
        }
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
