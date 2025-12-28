# ONNX IR Rust Design Document

## Overview

This document describes the design decisions, architecture, and implementation details of the ONNX IR Rust library. The design is based on the [onnx/ir-py](https://github.com/onnx/ir-py) project, adapted to Rust idioms and best practices.

## Design Goals

1. **Type Safety**: Leverage Rust's type system to catch errors at compile time
2. **Memory Safety**: Eliminate use-after-free, data races, and other memory bugs
3. **Performance**: Match or exceed ir-py performance while maintaining safety
4. **API Compatibility**: Provide similar APIs to ir-py for familiarity
5. **Extensibility**: Allow users to define custom tensor types and operations
6. **Zero Dependencies at Runtime**: Decouple from protobuf after deserialization

## Architecture

### Module Organization

The library is organized into focused modules, each with a single responsibility:

```
onnx-ir-core/
├── enums       - ONNX data types and attribute types
├── metadata    - Metadata storage for passes
├── name_authority - Unique name generation
├── linked_list - Safe mutation container
├── shape       - Shape and symbolic dimensions
├── types       - ONNX type system
├── tensor      - Tensor implementations
├── value       - Values with usage tracking
├── attribute   - Node and function attributes
├── node        - Operation nodes
├── graph       - Graph container
├── function    - Function definitions
└── model       - Top-level model
```

### Core Concepts

#### 1. Enums (DataType and AttributeType)

**Design**: Enum-based representation matching ONNX protobuf exactly.

```rust
#[repr(i32)]
pub enum DataType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    // ... matches TensorProto.DataType
}
```

**Rationale**:
- Direct mapping to ONNX spec ensures correctness
- `#[repr(i32)]` allows zero-cost conversion to/from protobuf
- Methods like `is_floating_point()`, `bitwidth()` provide convenience
- Naming follows ONNX convention (e.g., `Float` not `f32`) for clarity

#### 2. Metadata Store

**Design**: Type-erased key-value store with invalidation support.

```rust
pub struct MetadataStore {
    data: HashMap<String, Box<dyn Any>>,
    invalid_keys: HashSet<String>,
}
```

**Rationale**:
- Type erasure allows storing arbitrary metadata
- Invalidation mechanism supports incremental passes
- Separate from `metadata_props` which serialize to ONNX
- Uses `Any` for maximum flexibility

**Trade-offs**:
- Type safety lost at storage boundary (recovered at retrieval)
- Cannot clone arbitrary data (limitation accepted)
- Small runtime overhead for type checking

#### 3. Name Authority

**Design**: Centralized name generation and tracking.

```rust
pub struct NameAuthority {
    value_counter: usize,
    node_counter: usize,
    value_names: HashSet<String>,
    node_names: HashSet<String>,
}
```

**Rationale**:
- Prevents name collisions during graph construction
- Format matches ir-py: `val_{n}`, `node_{op_type}_{n}`
- Tracks all names even after removal (can be improved)
- Simple counter-based approach is fast and predictable

#### 4. Doubly-Linked List

**Design**: Intrusive doubly-linked list with `Rc<RefCell<_>>` nodes.

```rust
struct LinkBox<T> {
    prev: Cell<Option<Rc<LinkBox<T>>>>,
    next: Cell<Option<Rc<LinkBox<T>>>>,
    value: RefCell<Option<T>>,
}
```

**Rationale**:
- Supports safe iteration while mutating (requirement from ir-py)
- Interior mutability via `Cell` for link updates
- `RefCell` for value access
- Sentinel node simplifies edge cases

**Challenges**:
- Circular references require careful handling
- Pop operations need refinement (currently buggy)
- More complex than Vec but necessary for mutation guarantees

**Future Work**:
- Fix pop_front/pop_back operations
- Consider using `IndexSet` for simpler implementation
- Add benchmarks vs. alternative approaches

#### 5. Shape and Symbolic Dimensions

**Design**: Enum for concrete/symbolic dimensions, struct for shape.

```rust
pub enum SymbolicDim {
    Int(i64),
    Symbol(Option<String>),
}

pub struct Shape {
    dims: Vec<SymbolicDim>,
    denotations: Vec<Option<String>>,
    frozen: bool,
}
```

**Rationale**:
- Enum captures two cases clearly
- Freezing prevents accidental mutation
- Denotations match ONNX spec
- Immutable-by-default with explicit mutation

#### 6. Type System

**Design**: Trait for common interface, structs for specific types.

```rust
pub trait Type {
    fn denotation(&self) -> Option<&str>;
    fn dtype(&self) -> Option<DataType>;
}

pub struct TensorType {
    pub elem_type: DataType,
    pub denotation: Option<String>,
}
```

**Rationale**:
- Trait allows generic code over all type variants
- Separate structs for TensorType, SequenceType, OptionalType
- Recursive types use `Box<dyn Type>` (no Clone/PartialEq)
- Matches ONNX TypeProto structure

**Trade-offs**:
- Lost Clone/PartialEq for recursive types
- Could use enum instead of trait (future consideration)

#### 7. Tensor Implementations

**Design**: Multiple tensor types for different storage backends.

```rust
pub trait TensorProtocol {
    fn name(&self) -> Option<&str>;
    fn shape(&self) -> &Shape;
    fn dtype(&self) -> DataType;
    fn size(&self) -> usize;
    fn nbytes(&self) -> usize;
}

pub struct Tensor { /* in-memory */ }
pub struct ExternalTensor { /* mmap */ }
pub struct StringTensor { /* strings */ }
pub struct LazyTensor { /* deferred */ }
pub struct PackedTensor { /* sub-byte */ }
```

**Rationale**:
- Protocol pattern allows user-defined tensor types
- Different backends for different use cases
- ExternalTensor uses mmap for large data
- LazyTensor defers computation/loading
- PackedTensor for 2-bit/4-bit types

**Current Status**: Stubs only, full implementation pending

#### 8. Value with Usage Tracking

**Design**: Value owns its metadata, tracks producer/consumers.

```rust
pub struct Value {
    pub name: String,
    pub shape: Option<Shape>,
    pub type_: Option<TensorType>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
}
```

**Rationale**:
- Values are always named in a graph
- Shape and type are optional (may be unknown)
- Metadata for serialization vs. passes
- Usage tracking (producer, consumers) to be added

**Future Work**:
- Add producer: Option<Weak<Node>>
- Add consumers: Vec<(Weak<Node>, usize)>
- Implement replace_all_uses_with

#### 9. Node

**Design**: Node owns its attributes, references values.

```rust
pub struct Node {
    pub name: Option<String>,
    pub domain: String,
    pub op_type: String,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub attributes: IndexMap<String, Attr>,
    // ...
}
```

**Rationale**:
- IndexMap preserves attribute order
- Domain + op_type + overload identify operator
- Version allows mixed opsets
- Values are owned (may change to Rc)

**Current Issues**:
- Ownership model needs refinement
- Should inputs/outputs be Rc<Value>?

#### 10. Graph

**Design**: Graph owns nodes in a linked list, owns values.

```rust
pub struct Graph {
    pub nodes: DoublyLinkedList<Node>,
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
    pub initializers: HashMap<String, Value>,
    pub opset_imports: HashMap<String, i32>,
    // ...
}
```

**Rationale**:
- DoublyLinkedList allows safe mutation during iteration
- Inputs/outputs are ordered
- Initializers are named
- Opset imports per graph (not just model)

**Future Work**:
- Implement append/remove/insert operations
- Add topological sort
- Add validation

## Python Bindings Design

### Approach

Using PyO3 to create native Python extensions that implement Python protocols.

```rust
#[pyclass]
pub struct PyValue {
    inner: Rc<RefCell<Value>>,
}

#[pymethods]
impl PyValue {
    #[getter]
    fn name(&self) -> String {
        self.inner.borrow().name.clone()
    }
}
```

**Rationale**:
- PyO3 provides excellent Python integration
- Rc<RefCell<>> allows Python to hold references
- Protocols map directly to Python's duck typing
- Zero-copy where possible

### Protocol Implementation

Python protocols become trait bounds:

```python
# Python (ir-py)
class ValueProtocol(Protocol):
    name: str
    shape: ShapeProtocol | None
    type: TypeProtocol | None
```

```rust
// Rust equivalent
pub trait ValueProtocol {
    fn name(&self) -> &str;
    fn shape(&self) -> Option<&dyn ShapeProtocol>;
    fn type_(&self) -> Option<&dyn TypeProtocol>;
}
```

## Memory Management

### Ownership Strategy

**Model → Graph → Nodes → Values**

- Model owns Graph
- Graph owns Nodes (via DoublyLinkedList)
- Nodes reference Values (Rc or direct ownership TBD)
- Values may share Tensors (Rc)

### Reference Counting

Use Rc for shared ownership:
- Tensors (may be referenced by multiple values)
- Values (may be used by multiple nodes)

Use Weak for non-owning references:
- Node → producer (back-reference)
- Value → consumers (back-references)

### Interior Mutability

Use RefCell where needed:
- Graph mutation while iterating
- Node attribute updates
- Value usage tracking

## Error Handling

### Strategy

**Principle**: Use Result for recoverable errors, panic for programmer errors.

```rust
// Recoverable errors
pub fn from_proto(proto: ModelProto) -> Result<Model, DeserializationError> {
    // ...
}

// Programmer errors
pub fn set_dim(&mut self, index: usize, dim: SymbolicDim) {
    assert!(!self.frozen, "Cannot modify frozen shape");
    self.dims[index] = dim;
}
```

**Error Types**:
- `DeserializationError` - Proto parsing failures
- `SerializationError` - Proto creation failures  
- `ValidationError` - Graph validation failures
- `TypeError` - Type mismatches

## Performance Considerations

### Zero-Cost Abstractions

- Traits compile to static dispatch (monomorphization)
- Enums as small as their largest variant
- No virtual dispatch unless using trait objects

### Allocations

- Pre-allocate vectors when size is known
- Use SmallVec for small collections
- Intern strings where appropriate
- Memory-map external tensors

### Benchmarks

To be added:
- Graph traversal
- Serialization/deserialization
- Large model loading
- Pass execution

## Testing Strategy

### Unit Tests

Each module has comprehensive unit tests:
- Edge cases
- Error conditions
- Property-based tests (proptest)

### Integration Tests

Test entire workflows:
- Load ONNX model
- Transform graph
- Save ONNX model

### Compatibility Tests

Ensure compatibility with ir-py:
- Same model produces same IR
- Same transformations produce same result

## Future Enhancements

### Short Term
1. Complete tensor implementations
2. Fix linked list pop/clear
3. Implement serialization/deserialization
4. Add graph traversal utilities

### Medium Term
1. Optimization pass infrastructure
2. Common passes (constant folding, DCE, etc.)
3. Python bindings completion
4. Comprehensive documentation

### Long Term
1. Parallel graph analysis
2. Custom allocators for memory efficiency
3. GPU tensor support
4. Compilation to machine code

## References

1. [onnx/ir-py Design](https://github.com/onnx/ir-py/blob/main/src/onnx_ir/_protocols.py)
2. [ONNX Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)
3. [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
4. [PyO3 User Guide](https://pyo3.rs/)

## Appendix: Rust-Specific Patterns

### Trait Objects vs Enums

**When to use trait objects**:
- Unknown number of types
- User-defined types
- Plugin architecture

**When to use enums**:
- Fixed set of variants
- Performance critical
- Need exhaustiveness checking

### Interior Mutability Patterns

**Cell**: Copy types, no borrowing
**RefCell**: Non-Copy types, runtime borrow checking
**Mutex**: Thread-safe mutation
**RwLock**: Multiple readers, single writer

### Lifetime Annotations

Explicit lifetimes document ownership:
```rust
pub struct Iter<'a, T> {
    current: Option<&'a LinkBox<T>>,
    // 'a ties iterator lifetime to collection
}
```

This prevents use-after-free at compile time.
