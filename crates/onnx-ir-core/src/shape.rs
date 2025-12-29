// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shape and symbolic dimension types.
//!
//! This module provides the shape representation for ONNX tensors,
//! including support for symbolic/dynamic dimensions.

use std::fmt;

/// A symbolic or dynamic dimension in a shape.
///
/// A dimension can be either a concrete integer value or a symbolic
/// parameter represented by a string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicDim {
    /// A concrete integer dimension.
    Int(i64),
    /// A symbolic dimension with an optional parameter name.
    Symbol(Option<String>),
}

impl SymbolicDim {
    /// Creates a new symbolic dimension from an integer.
    pub fn from_int(value: i64) -> Self {
        Self::Int(value)
    }

    /// Creates a new symbolic dimension from a symbol name.
    pub fn from_symbol(name: Option<String>) -> Self {
        Self::Symbol(name)
    }

    /// Returns true if this is a concrete integer dimension.
    pub fn is_int(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Returns true if this is a symbolic dimension.
    pub fn is_symbol(&self) -> bool {
        matches!(self, Self::Symbol(_))
    }

    /// Returns the integer value if this is a concrete dimension.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            _ => None,
        }
    }
}

impl fmt::Display for SymbolicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{}", v),
            Self::Symbol(Some(name)) => write!(f, "{}", name),
            Self::Symbol(None) => write!(f, "?"),
        }
    }
}

impl From<i64> for SymbolicDim {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

/// A shape is a sequence of dimensions.
///
/// Each dimension can be either a concrete integer or a symbolic parameter.
/// Shapes are immutable once frozen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<SymbolicDim>,
    denotations: Vec<Option<String>>,
    frozen: bool,
}

impl Shape {
    /// Creates a new shape from a sequence of dimensions.
    pub fn new(dims: impl IntoIterator<Item = impl Into<SymbolicDim>>) -> Self {
        let dims: Vec<_> = dims.into_iter().map(Into::into).collect();
        let denotations = vec![None; dims.len()];
        Self {
            dims,
            denotations,
            frozen: false,
        }
    }

    /// Creates an empty shape (scalar).
    pub fn scalar() -> Self {
        Self::new(Vec::<i64>::new())
    }

    /// Returns the dimensions of the shape.
    pub fn dims(&self) -> &[SymbolicDim] {
        &self.dims
    }

    /// Returns the rank (number of dimensions) of the shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements (product of all dimensions).
    ///
    /// Returns 1 for scalar shapes (rank 0).
    /// Returns `None` if any dimension is symbolic.
    pub fn size(&self) -> Option<usize> {
        if self.is_scalar() {
            return Some(1);
        }

        let mut total = 1usize;
        for dim in &self.dims {
            match dim {
                SymbolicDim::Int(v) => {
                    total = total.saturating_mul(*v as usize);
                }
                SymbolicDim::Symbol(_) => {
                    // Symbolic dimensions mean we can't compute a concrete size
                    return None;
                }
            }
        }
        Some(total)
    }

    /// Returns true if this is a scalar (rank 0).
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Freezes the shape, making it immutable.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Returns true if the shape is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Sets a dimension at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the shape is frozen or the index is out of bounds.
    pub fn set_dim(&mut self, index: usize, dim: impl Into<SymbolicDim>) {
        assert!(!self.frozen, "Cannot modify frozen shape");
        self.dims[index] = dim.into();
    }

    /// Gets the denotation at the given index.
    pub fn get_denotation(&self, index: usize) -> Option<&str> {
        self.denotations.get(index)?.as_deref()
    }

    /// Sets the denotation at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the shape is frozen or the index is out of bounds.
    pub fn set_denotation(&mut self, index: usize, denotation: Option<String>) {
        assert!(!self.frozen, "Cannot modify frozen shape");
        self.denotations[index] = denotation;
    }

    /// Converts to a vector of concrete integers.
    ///
    /// Returns `None` if any dimension is symbolic.
    pub fn to_vec(&self) -> Option<Vec<i64>> {
        self.dims.iter().map(|d| d.as_int()).collect()
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl<T: Into<SymbolicDim>> FromIterator<T> for Shape {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_dim() {
        let dim1 = SymbolicDim::from_int(10);
        assert!(dim1.is_int());
        assert_eq!(dim1.as_int(), Some(10));

        let dim2 = SymbolicDim::from_symbol(Some("N".to_string()));
        assert!(dim2.is_symbol());
        assert_eq!(dim2.as_int(), None);
    }

    #[test]
    fn test_shape() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.to_vec(), Some(vec![1, 2, 3]));

        let scalar = Shape::scalar();
        assert!(scalar.is_scalar());
        assert_eq!(scalar.rank(), 0);
    }

    #[test]
    fn test_shape_mutation() {
        let mut shape = Shape::new(vec![1, 2, 3]);
        shape.set_dim(1, 5);
        assert_eq!(shape.to_vec(), Some(vec![1, 5, 3]));

        shape.freeze();
        assert!(shape.is_frozen());
    }

    #[test]
    fn test_shape_size() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.size(), Some(24));

        let scalar = Shape::scalar();
        assert_eq!(scalar.size(), Some(1));
    }

    #[test]
    fn test_symbolic_shape_size() {
        use super::SymbolicDim;

        let mut shape = Shape::new(vec![2, 3]);
        assert_eq!(shape.size(), Some(6));

        // Set one dimension to symbolic
        shape.set_dim(1, SymbolicDim::Symbol(Some("N".to_string())));
        assert_eq!(shape.size(), None); // Should return None for symbolic dimensions
    }

    #[test]
    #[should_panic(expected = "Cannot modify frozen shape")]
    fn test_frozen_shape() {
        let mut shape = Shape::new(vec![1, 2, 3]);
        shape.freeze();
        shape.set_dim(0, 10); // Should panic
    }
}
