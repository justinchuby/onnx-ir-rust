// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX IR enums that match the ONNX specification.
//!
//! This module defines the type enums that correspond to `DataType` and
//! `AttributeType` in the ONNX protobuf specification.

use std::fmt;

/// Enum for the types of ONNX attributes.
///
/// Corresponds to `AttributeProto.AttributeType` in the ONNX specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensor = 11,
    SparseTensors = 12,
    TypeProto = 13,
    TypeProtos = 14,
}

impl fmt::Display for AttributeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Enum for the data types of ONNX tensors.
///
/// Corresponds to `TensorProto.DataType` in the ONNX specification.
///
/// # Naming Convention
///
/// The naming follows the ONNX specification for consistency (e.g., `FLOAT`, `INT64`)
/// rather than more modern Rust conventions (e.g., `f32`, `i64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
    Float8E4M3Fn = 17,
    Float8E4M3Fnuz = 18,
    Float8E5M2 = 19,
    Float8E5M2Fnuz = 20,
    Uint4 = 21,
    Int4 = 22,
    Float4E2M1 = 23,
    Float8E8M0 = 24,
    Uint2 = 25,
    Int2 = 26,
}

impl DataType {
    /// Returns the bit width of the data type.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert_eq!(DataType::Float.bitwidth(), Some(32));
    /// assert_eq!(DataType::Int64.bitwidth(), Some(64));
    /// assert_eq!(DataType::Float16.bitwidth(), Some(16));
    /// assert_eq!(DataType::Int4.bitwidth(), Some(4));
    /// assert_eq!(DataType::String.bitwidth(), None);
    /// ```
    pub fn bitwidth(&self) -> Option<usize> {
        match self {
            DataType::Float => Some(32),
            DataType::Uint8 => Some(8),
            DataType::Int8 => Some(8),
            DataType::Uint16 => Some(16),
            DataType::Int16 => Some(16),
            DataType::Int32 => Some(32),
            DataType::Int64 => Some(64),
            DataType::Bool => Some(8),
            DataType::Float16 => Some(16),
            DataType::Double => Some(64),
            DataType::Uint32 => Some(32),
            DataType::Uint64 => Some(64),
            DataType::Complex64 => Some(64),
            DataType::Complex128 => Some(128),
            DataType::Bfloat16 => Some(16),
            DataType::Float8E4M3Fn => Some(8),
            DataType::Float8E4M3Fnuz => Some(8),
            DataType::Float8E5M2 => Some(8),
            DataType::Float8E5M2Fnuz => Some(8),
            DataType::Uint4 => Some(4),
            DataType::Int4 => Some(4),
            DataType::Float4E2M1 => Some(4),
            DataType::Float8E8M0 => Some(8),
            DataType::Int2 => Some(2),
            DataType::Uint2 => Some(2),
            _ => None,
        }
    }

    /// Returns the size of the data type in bytes.
    ///
    /// For sub-byte types (2-bit, 4-bit), returns a fractional value.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert_eq!(DataType::Float.itemsize(), Some(4.0));
    /// assert_eq!(DataType::Int4.itemsize(), Some(0.5));
    /// assert_eq!(DataType::Int2.itemsize(), Some(0.25));
    /// ```
    pub fn itemsize(&self) -> Option<f64> {
        self.bitwidth().map(|bits| bits as f64 / 8.0)
    }

    /// Returns true if the data type is a floating point type.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert!(DataType::Float.is_floating_point());
    /// assert!(DataType::Float16.is_floating_point());
    /// assert!(DataType::Double.is_floating_point());
    /// assert!(DataType::Bfloat16.is_floating_point());
    /// assert!(!DataType::Int32.is_floating_point());
    /// ```
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            DataType::Float
                | DataType::Float16
                | DataType::Double
                | DataType::Bfloat16
                | DataType::Float8E4M3Fn
                | DataType::Float8E4M3Fnuz
                | DataType::Float8E5M2
                | DataType::Float8E5M2Fnuz
                | DataType::Float4E2M1
                | DataType::Float8E8M0
        )
    }

    /// Returns true if the data type is an integer type.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert!(DataType::Int32.is_integer());
    /// assert!(DataType::Uint8.is_integer());
    /// assert!(DataType::Int4.is_integer());
    /// assert!(!DataType::Float.is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Uint8
                | DataType::Int8
                | DataType::Uint16
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint32
                | DataType::Uint64
                | DataType::Uint4
                | DataType::Int4
                | DataType::Int2
                | DataType::Uint2
        )
    }

    /// Returns true if the data type is a signed type.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert!(DataType::Int32.is_signed());
    /// assert!(DataType::Float.is_signed());
    /// assert!(!DataType::Uint32.is_signed());
    /// ```
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DataType::Float
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float16
                | DataType::Double
                | DataType::Complex64
                | DataType::Complex128
                | DataType::Bfloat16
                | DataType::Float8E4M3Fn
                | DataType::Float8E4M3Fnuz
                | DataType::Float8E5M2
                | DataType::Float8E5M2Fnuz
                | DataType::Int4
                | DataType::Float4E2M1
                | DataType::Float8E8M0
                | DataType::Int2
        )
    }

    /// Returns true if the data type is a string type.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert!(DataType::String.is_string());
    /// assert!(!DataType::Int32.is_string());
    /// ```
    pub fn is_string(&self) -> bool {
        matches!(self, DataType::String)
    }

    /// Returns a short name for the data type.
    ///
    /// The short name is a compact string representation used for display
    /// and debugging purposes (e.g., "f32" for Float, "i64" for Int64).
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert_eq!(DataType::Float.short_name(), "f32");
    /// assert_eq!(DataType::Int64.short_name(), "i64");
    /// assert_eq!(DataType::Bfloat16.short_name(), "bf16");
    /// ```
    pub fn short_name(&self) -> &'static str {
        match self {
            DataType::Undefined => "undefined",
            DataType::Float => "f32",
            DataType::Uint8 => "u8",
            DataType::Int8 => "i8",
            DataType::Uint16 => "u16",
            DataType::Int16 => "i16",
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::String => "s",
            DataType::Bool => "b8",
            DataType::Float16 => "f16",
            DataType::Double => "f64",
            DataType::Uint32 => "u32",
            DataType::Uint64 => "u64",
            DataType::Complex64 => "c64",
            DataType::Complex128 => "c128",
            DataType::Bfloat16 => "bf16",
            DataType::Float8E4M3Fn => "f8e4m3fn",
            DataType::Float8E4M3Fnuz => "f8e4m3fnuz",
            DataType::Float8E5M2 => "f8e5m2",
            DataType::Float8E5M2Fnuz => "f8e5m2fnuz",
            DataType::Uint4 => "u4",
            DataType::Int4 => "i4",
            DataType::Float4E2M1 => "f4e2m1",
            DataType::Float8E8M0 => "f8e8m0",
            DataType::Uint2 => "u2",
            DataType::Int2 => "i2",
        }
    }

    /// Creates a DataType from a short name string.
    ///
    /// # Examples
    ///
    /// ```
    /// use onnx_ir_core::DataType;
    ///
    /// assert_eq!(DataType::from_short_name("f32"), Some(DataType::Float));
    /// assert_eq!(DataType::from_short_name("i64"), Some(DataType::Int64));
    /// assert_eq!(DataType::from_short_name("invalid"), None);
    /// ```
    pub fn from_short_name(name: &str) -> Option<DataType> {
        match name {
            "undefined" => Some(DataType::Undefined),
            "f32" => Some(DataType::Float),
            "u8" => Some(DataType::Uint8),
            "i8" => Some(DataType::Int8),
            "u16" => Some(DataType::Uint16),
            "i16" => Some(DataType::Int16),
            "i32" => Some(DataType::Int32),
            "i64" => Some(DataType::Int64),
            "s" => Some(DataType::String),
            "b8" => Some(DataType::Bool),
            "f16" => Some(DataType::Float16),
            "f64" => Some(DataType::Double),
            "u32" => Some(DataType::Uint32),
            "u64" => Some(DataType::Uint64),
            "c64" => Some(DataType::Complex64),
            "c128" => Some(DataType::Complex128),
            "bf16" => Some(DataType::Bfloat16),
            "f8e4m3fn" => Some(DataType::Float8E4M3Fn),
            "f8e4m3fnuz" => Some(DataType::Float8E4M3Fnuz),
            "f8e5m2" => Some(DataType::Float8E5M2),
            "f8e5m2fnuz" => Some(DataType::Float8E5M2Fnuz),
            "u4" => Some(DataType::Uint4),
            "i4" => Some(DataType::Int4),
            "f4e2m1" => Some(DataType::Float4E2M1),
            "f8e8m0" => Some(DataType::Float8E8M0),
            "u2" => Some(DataType::Uint2),
            "i2" => Some(DataType::Int2),
            _ => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datatype_bitwidth() {
        assert_eq!(DataType::Float.bitwidth(), Some(32));
        assert_eq!(DataType::Int64.bitwidth(), Some(64));
        assert_eq!(DataType::Int4.bitwidth(), Some(4));
        assert_eq!(DataType::Int2.bitwidth(), Some(2));
        assert_eq!(DataType::String.bitwidth(), None);
    }

    #[test]
    fn test_datatype_itemsize() {
        assert_eq!(DataType::Float.itemsize(), Some(4.0));
        assert_eq!(DataType::Int64.itemsize(), Some(8.0));
        assert_eq!(DataType::Int4.itemsize(), Some(0.5));
        assert_eq!(DataType::Int2.itemsize(), Some(0.25));
    }

    #[test]
    fn test_datatype_predicates() {
        assert!(DataType::Float.is_floating_point());
        assert!(!DataType::Int32.is_floating_point());
        
        assert!(DataType::Int32.is_integer());
        assert!(!DataType::Float.is_integer());
        
        assert!(DataType::Int32.is_signed());
        assert!(!DataType::Uint32.is_signed());
        
        assert!(DataType::String.is_string());
        assert!(!DataType::Float.is_string());
    }

    #[test]
    fn test_short_name_roundtrip() {
        let types = vec![
            DataType::Float,
            DataType::Int64,
            DataType::Bfloat16,
            DataType::Int4,
        ];
        
        for dtype in types {
            let short = dtype.short_name();
            assert_eq!(DataType::from_short_name(short), Some(dtype));
        }
    }
}
