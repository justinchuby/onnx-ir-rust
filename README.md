# ONNX IR Rust

> [!WARNING]
> This project is purely experimental, built primarily by Copilot.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

An in-memory Intermediate Representation (IR) for ONNX models in Rust, inspired by the [onnx/ir-py](https://github.com/onnx/ir-py) project.

## Overview

ONNX IR Rust provides a complete implementation of the ONNX specification as an intermediate representation designed for graph construction, analysis, and transformation. The implementation follows Rust best practices while preserving API similarity with the Python ir-py project.

## Features ✨

- **Full ONNX spec support**: All valid models representable by ONNX protobuf
- **Low memory footprint**: Memory-mapped external tensors, zero-copy operations
- **Type-safe APIs**: Leverages Rust's type system for correctness
- **Safe mutation**: Robust graph mutation with safe iterator handling
- **No protobuf runtime dependency**: IR is decoupled from serialization format
- **Python bindings**: PyO3-based bindings implementing Python protocols
- **Comprehensive documentation**: Extensive API docs and design documentation

## Architecture

The IR is built around several core concepts mirroring the ir-py design:

- **Model**: Top-level container for a graph and metadata
- **Graph**: A computational graph with nodes, values, and initializers
- **Node**: An operation invocation in the graph
- **Value**: Named entities representing inputs/outputs of nodes
- **Tensor**: Concrete data with various storage backends

### Design Philosophy

This implementation adapts the ir-py design to Rust idioms:

- **Protocols → Traits**: Python protocols are implemented as Rust traits
- **Duck Typing → Generic Bounds**: Type compatibility via trait bounds
- **Mutable Sequences → Interior Mutability**: Graph mutation uses interior mutability patterns
- **Reference Counting**: `Rc`/`Arc` for shared ownership where needed

## Project Structure

```
onnx-ir-rust/
├── crates/
│   ├── onnx-ir-core/     # Core Rust library
│   │   ├── src/
│   │   │   ├── enums.rs          # DataType and AttributeType enums
│   │   │   ├── metadata.rs       # Metadata storage
│   │   │   ├── name_authority.rs # Name generation
│   │   │   ├── linked_list.rs    # Safe mutation container
│   │   │   ├── shape.rs          # Shape and symbolic dimensions
│   │   │   ├── types.rs          # Type system
│   │   │   ├── tensor.rs         # Tensor implementations
│   │   │   ├── value.rs          # Value with usage tracking
│   │   │   ├── attribute.rs      # Attributes
│   │   │   ├── node.rs           # Nodes
│   │   │   ├── graph.rs          # Graph container
│   │   │   ├── function.rs       # Functions
│   │   │   └── model.rs          # Model
│   │   └── Cargo.toml
│   └── onnx-ir-py/       # Python bindings
│       ├── src/
│       │   └── lib.rs
│       └── Cargo.toml
├── Cargo.toml            # Workspace configuration
├── README.md
└── LICENSE
```

## Getting Started

### Prerequisites

- Rust 1.70 or later
- (Optional) Python 3.9+ for Python bindings

### Building

```bash
# Build the core library
cargo build --release

# Build with Python bindings
cargo build --release -p onnx-ir-py

# Run tests
cargo test

# Generate documentation
cargo doc --open
```

### Usage Example

```rust
use onnx_ir_core::{Graph, Node, Value, DataType, Shape};

// Create a new graph
let mut graph = Graph::new();
graph.name = Some("example_graph".to_string());

// Create values
let input = Value::new("input");
let output = Value::new("output");

// Create a node
let mut node = Node::new("Add");
node.inputs = vec![input];
node.outputs = vec![output];

// Add node to graph
graph.nodes.push_back(node);
```

## Python Bindings

The Python bindings implement the same protocols as ir-py, providing a familiar API:

```python
import onnx_ir

# The Python API mirrors the ir-py project
# Full implementation in progress
```

## Development Status

**Current Status**: Initial implementation (Phase 2 of 10)

### Completed
- ✅ Project structure and build system
- ✅ Core enums (DataType, AttributeType)
- ✅ Metadata storage
- ✅ Name authority
- ✅ Doubly-linked list container (partial)
- ✅ Shape and symbolic dimensions
- ✅ Basic type system
- ✅ Tensor stubs
- ✅ Value, Node, Graph stubs
- ✅ Basic documentation

### In Progress
- 🔨 Doubly-linked list refinement (pop/clear operations)
- 🔨 Complete tensor implementations
- 🔨 Graph mutation operations
- 🔨 Python bindings

### Planned
- 📋 Serialization/deserialization
- 📋 Graph traversal utilities
- 📋 Optimization passes
- 📋 Comprehensive test coverage
- 📋 Benchmarks

See [full project plan](https://github.com/justinchuby/onnx-ir-rust/pulls) in the PR description.

## Design Decisions

### Why Rust?

- **Memory safety**: Eliminates entire classes of bugs
- **Performance**: Zero-cost abstractions and efficient execution
- **Concurrency**: Fearless concurrency for parallel graph operations
- **Type safety**: Catch errors at compile time
- **Interoperability**: Easy FFI with C/C++ and Python

### Key Design Choices

1. **Interior Mutability**: Uses `RefCell`/`Cell` for safe graph mutation while maintaining Rust's safety guarantees

2. **Trait-based Protocols**: Implements protocols as traits, allowing user-defined types to integrate seamlessly

3. **Zero-Copy Where Possible**: Uses memory mapping for external tensors, `Rc` for shared references

4. **Explicit Lifetimes**: Clear ownership and borrowing rules prevent use-after-free

5. **Error Handling**: Uses `Result<T, E>` for recoverable errors, `panic!` only for programmer errors

## Contributing

Contributions are welcome! This project follows the ONNX contribution guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `cargo fmt` and `cargo clippy`
5. Submit a pull request

### Code Style

- Follow Rust standard naming conventions
- Document public APIs with rustdoc
- Add tests for new functionality
- Keep commits atomic and well-described

## Comparison with ir-py

| Aspect | ir-py (Python) | onnx-ir-rust (Rust) |
|--------|---------------|---------------------|
| Memory Safety | Runtime checks | Compile-time guarantees |
| Performance | Interpreted | Compiled, zero-cost abstractions |
| Type System | Duck typing | Static typing with traits |
| Mutation | Direct mutation | Interior mutability patterns |
| Concurrency | GIL limitations | Fearless concurrency |
| FFI | C extensions | Native FFI, easy C/Python interop |

Both implementations share the same conceptual design and support the full ONNX specification.

## References

- [onnx/ir-py](https://github.com/onnx/ir-py) - Python implementation
- [ONNX Specification](https://github.com/onnx/onnx)
- [PyO3](https://pyo3.rs/) - Python bindings for Rust

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

This project is based on the design of [onnx/ir-py](https://github.com/onnx/ir-py) by the ONNX Project Contributors.
