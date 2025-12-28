// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! # ONNX IR Core
//!
//! An in-memory intermediate representation for ONNX models in Rust.
//!
//! This crate provides a complete implementation of the ONNX specification
//! as an intermediate representation (IR) designed for graph construction,
//! analysis, and transformation.
//!
//! ## Features
//!
//! - **Full ONNX spec support**: All valid models representable by ONNX protobuf
//! - **Low memory footprint**: Memory-mapped external tensors, zero-copy operations
//! - **Safe mutation**: Robust graph mutation with safe iterator invalidation
//! - **Type-safe APIs**: Leverages Rust's type system for correctness
//! - **No protobuf runtime dependency**: IR is decoupled from serialization format
//!
//! ## Architecture
//!
//! The IR is built around several core concepts:
//!
//! - **Model**: Top-level container for a graph and metadata
//! - **Graph**: A computational graph with nodes, values, and initializers
//! - **Node**: An operation invocation in the graph
//! - **Value**: Named entities representing inputs/outputs of nodes
//! - **Tensor**: Concrete data with various storage backends
//!
//! ## Design Philosophy
//!
//! This implementation follows the design of the [onnx/ir-py](https://github.com/onnx/ir-py)
//! project while adapting to Rust idioms and best practices:
//!
//! - **Protocols → Traits**: Python protocols are implemented as Rust traits
//! - **Mutable Sequences → Interior Mutability**: Graph mutation uses interior mutability patterns
//! - **Duck Typing → Generic Bounds**: Type compatibility via trait bounds
//! - **Reference Counting**: `Rc`/`Arc` for shared ownership where needed
//!
//! ## Module Organization
//!
//! - [`enums`]: ONNX data type and attribute type enumerations
//! - [`metadata`]: Metadata storage for IR objects
//! - [`name_authority`]: Name generation and uniqueness management
//! - [`linked_list`]: Doubly-linked list for safe node container
//! - [`shape`]: Shape and symbolic dimension types
//! - [`tensor`]: Tensor protocols and implementations
//! - [`types`]: ONNX type system (TensorType, OptionalType, etc.)
//! - [`value`]: Value representation with usage tracking
//! - [`node`]: Node with attributes and input/output management
//! - [`graph`]: Graph container with mutation support
//! - [`function`]: Function definition support
//! - [`model`]: Top-level model container
//! - Serialization/deserialization to ONNX protobuf (planned)

pub mod enums;
pub mod metadata;
pub mod name_authority;
pub mod linked_list;
pub mod shape;
pub mod types;
pub mod tensor;
pub mod value;
pub mod attribute;
pub mod node;
pub mod graph;
pub mod function;
pub mod model;

// Re-export commonly used types
pub use enums::{AttributeType, DataType};
pub use metadata::MetadataStore;
pub use shape::{Shape, SymbolicDim};
pub use tensor::{Tensor, ExternalTensor, StringTensor, LazyTensor, PackedTensor};
pub use types::{TensorType, OptionalType, SequenceType, SparseTensorType};
pub use value::Value;
pub use attribute::{Attr, RefAttr};
pub use node::Node;
pub use graph::{Graph, GraphView};
pub use function::Function;
pub use model::Model;

/// Version of the ONNX IR implementation
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Debug mode flag for additional validation
pub static DEBUG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
