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

**Design**: Value uses interior mutability for usage tracking with Rc/Weak references.

```rust
pub struct Value {
    pub name: String,
    pub shape: Option<Shape>,
    pub type_: Option<TensorType>,
    pub metadata_props: HashMap<String, String>,
    pub meta: MetadataStore,
    // Usage tracking with interior mutability
    producer: RefCell<Option<Weak<RefCell<Node>>>>,
    consumers: RefCell<Vec<(Weak<RefCell<Node>>, usize)>>,
}
```

**Rationale**:
- Values are always named in a graph
- Shape and type are optional (may be unknown)
- Metadata for serialization vs. passes
- Interior mutability (RefCell) allows tracking updates without mutable reference
- Weak references prevent circular ownership (Graph → Node → Value → Node cycles)
- Consumer tracking includes input index for precise replacement

**Design Decision: Ownership Model for Values**

Three options were considered:

**Option 1: Direct Ownership (Current Stub)**
```rust
pub struct Node {
    pub inputs: Vec<Value>,
    pub outputs: Vec<Value>,
}
```
Pros:
- Simple ownership model
- No reference counting overhead
- Clear ownership semantics

Cons:
- Cannot track usage (producer/consumers)
- Cannot share values between nodes
- Values copied when passed between nodes
- No way to implement replace_all_uses_with
- **REJECTED**: Doesn't meet ONNX IR requirements

**Option 2: Rc<Value> with Interior Mutability**
```rust
pub struct Node {
    pub inputs: Vec<Rc<Value>>,
    pub outputs: Vec<Rc<Value>>,
}
pub struct Value {
    // ... fields
    producer: RefCell<Option<Weak<RefCell<Node>>>>,
    consumers: RefCell<Vec<(Weak<RefCell<Node>>, usize)>>,
}
```
Pros:
- Shared ownership allows multiple nodes to reference same value
- Can track all users (producer + consumers)
- Supports replace_all_uses_with operation
- Interior mutability allows updates without &mut
- Matches ir-py semantics

Cons:
- Runtime overhead from Rc reference counting
- Runtime borrow checking with RefCell (can panic)
- More complex ownership model
- **SELECTED**: Best fit for ONNX IR requirements

**Option 3: Arena-based with IDs**
```rust
pub struct Graph {
    values: Arena<Value>,
    nodes: Arena<Node>,
}
pub struct Node {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}
```
Pros:
- No reference counting overhead
- Predictable memory layout
- Easy serialization (IDs map to indices)

Cons:
- Lifetime complexity (nodes tied to graph lifetime)
- Cannot move values between graphs easily
- Requires ID indirection for all access
- Less ergonomic API
- **REJECTED**: Lifetime constraints too restrictive

**Selected Approach: Option 2 (Rc<RefCell<Value>>)**

This provides the necessary flexibility for graph transformations while maintaining correctness:
1. Values can be shared across multiple nodes
2. Usage tracking enables optimization passes
3. replace_all_uses_with enables IR transformation
4. Interior mutability matches Python's mutation model
5. Runtime checks prevent use-after-free

**Trade-offs Accepted**:
- Runtime reference counting cost (acceptable for IR manipulation)
- RefCell borrow checking overhead (prevents bugs, worth the cost)
- Slightly more complex API (necessary for correctness)

#### 9. Node

**Design**: Node owns its attributes, holds Rc references to values.

```rust
pub struct Node {
    pub name: Option<String>,
    pub domain: String,
    pub op_type: String,
    pub inputs: Vec<Rc<RefCell<Value>>>,
    pub outputs: Vec<Rc<RefCell<Value>>>,
    pub attributes: IndexMap<String, Attr>,
    // ...
}
```

**Rationale**:
- IndexMap preserves attribute order
- Domain + op_type + overload identify operator
- Version allows mixed opsets
- Rc<RefCell<Value>> enables shared ownership and usage tracking
- Node wrapped in RefCell when stored in graph for mutation

**Usage Tracking Integration**:
When a node is created and values are assigned:
1. Node increments Rc count by holding reference
2. Value's producer/consumer list updated via RefCell
3. Weak references prevent cycles (Value -Weak-> Node)
4. Allows efficient replace_all_uses_with operations

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

**Finalized Ownership Model:**

```
Model → Graph → Rc<RefCell<Node>> → Rc<RefCell<Value>>
                     ↓ (Weak)              ↓ (Weak)
                     Node ← - - - - - - - Value
```

**Detailed Design**:

1. **Model owns Graph**: Single ownership, no sharing
2. **Graph owns Nodes**: Via DoublyLinkedList<Rc<RefCell<Node>>>
   - Rc allows nodes to be referenced during iteration
   - RefCell enables mutation while iterating
3. **Nodes hold Rc<RefCell<Value>>**: Shared ownership
   - Multiple nodes can reference same value
   - RefCell allows usage tracking updates
4. **Values hold Weak<RefCell<Node>>**: Non-owning back-references
   - Prevents ownership cycles
   - Enables producer/consumer tracking
   - Automatically cleared when node is dropped

**Ownership Invariants**:

1. Graph is the source of truth for node lifetime
2. Values outlive their users only while Rc references exist
3. No value can exist without being in some node's input/output
4. Weak references never prevent node garbage collection
5. Usage tracking is always consistent with actual references

**Memory Safety Guarantees**:

1. No use-after-free: Weak references check validity before access
2. No data races: RefCell provides runtime borrow checking
3. No memory leaks: No Rc cycles (only Rc → Weak)
4. Deterministic cleanup: When node dropped, Weak refs invalidate

### Reference Counting

**Rc for shared ownership:**
- Tensors (may be referenced by multiple values)
- Values (referenced by multiple nodes)
- Nodes (held in graph, may be referenced during iteration)

**Weak for non-owning references:**
- Value → Node (producer back-reference)
- Value → Node (consumer back-references)
- Prevents cycles in ownership graph

**Design Rationale**:

The key insight is that the ownership graph must be acyclic:
- Graph → Node: Strong (Rc), Graph owns nodes
- Node → Value: Strong (Rc), Nodes share values
- Value → Node: Weak (Weak), Prevents cycles

This ensures:
1. When a graph is dropped, all nodes are dropped
2. When all nodes using a value are dropped, value is dropped
3. No reference cycles, no memory leaks
4. Values can track their users without preventing cleanup

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
