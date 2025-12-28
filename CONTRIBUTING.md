# Contributing to ONNX IR Rust

Thank you for your interest in contributing to ONNX IR Rust! This document provides guidelines and information for contributors.

## Code of Conduct

This project follows the [ONNX Code of Conduct](https://github.com/onnx/onnx/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Git
- (Optional) Python 3.9+ for Python bindings development

### Setting Up Development Environment

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/onnx-ir-rust.git
cd onnx-ir-rust
```

2. Build the project:
```bash
cargo build
```

3. Run tests:
```bash
cargo test
```

4. Check code quality:
```bash
cargo clippy
cargo fmt --check
```

## Development Workflow

### Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the coding standards

3. Add tests for new functionality

4. Run tests and linters:
```bash
cargo test
cargo clippy -- -D warnings
cargo fmt
```

5. Commit your changes with clear messages:
```bash
git commit -m "Add feature: description of your changes"
```

6. Push to your fork and create a pull request

### Coding Standards

#### Rust Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` with default settings
- Address all `cargo clippy` warnings
- Prefer explicit over implicit
- Document public APIs with rustdoc comments

**Example**:
```rust
/// Calculates the number of elements in a tensor.
///
/// # Examples
///
/// ```
/// use onnx_ir_core::Shape;
///
/// let shape = Shape::new(vec![2, 3, 4]);
/// assert_eq!(shape.size(), 24);
/// ```
pub fn size(&self) -> usize {
    self.dims.iter()
        .filter_map(|d| d.as_int())
        .product()
}
```

#### Naming Conventions

- Types: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Module names: `snake_case`

#### Documentation

- All public items must have doc comments
- Include examples in doc comments where helpful
- Explain *why* not just *what*
- Link to related items
- Document panics, errors, and safety

**Doc comment template**:
```rust
/// Brief one-line description.
///
/// Longer description explaining the functionality,
/// design decisions, and usage patterns.
///
/// # Arguments
///
/// * `arg_name` - Description of the argument
///
/// # Returns
///
/// Description of the return value
///
/// # Examples
///
/// ```
/// use onnx_ir_core::Example;
///
/// let example = Example::new();
/// ```
///
/// # Panics
///
/// Describe panic conditions if any
///
/// # Errors
///
/// Describe error conditions if any
///
/// # Safety
///
/// Required for unsafe functions
pub fn example_function(arg_name: Type) -> ReturnType {
    // implementation
}
```

### Testing

#### Unit Tests

- Add unit tests in the same file as the code
- Use the `#[cfg(test)]` module
- Test edge cases and error conditions
- Aim for high coverage

**Example**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let result = function_under_test();
        assert_eq!(result, expected_value);
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_panic_condition() {
        panic_inducing_code();
    }

    #[test]
    fn test_edge_case() {
        // Test boundary conditions
    }
}
```

#### Integration Tests

- Place in `tests/` directory
- Test complete workflows
- Test interop with other components

#### Property-Based Tests

- Use `proptest` for property-based testing
- Good for testing invariants
- Catches edge cases

**Example**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_property(value in 0..100) {
        let result = function(value);
        assert!(result >= 0);
    }
}
```

### Performance

- Profile before optimizing
- Use benchmarks to verify improvements
- Document performance characteristics
- Avoid premature optimization

**Benchmark example**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_function(c: &mut Criterion) {
    c.bench_function("operation", |b| {
        b.iter(|| function(black_box(input)))
    });
}

criterion_group!(benches, benchmark_function);
criterion_main!(benches);
```

## Architecture and Design

### Design Principles

1. **Safety First**: Leverage Rust's type system
2. **Zero-Cost Abstractions**: No runtime overhead
3. **Explicit is Better**: Clear ownership and lifetimes
4. **Compatibility**: Match ir-py design where sensible
5. **Documentation**: Code should be self-documenting

### Module Organization

- Each module should have a single, clear responsibility
- Public API should be minimal but complete
- Internal helpers should be private
- Re-export commonly used types at crate root

### Error Handling

- Use `Result<T, E>` for recoverable errors
- Use `panic!` only for programmer errors
- Provide context with error messages
- Define custom error types when appropriate

## Pull Request Process

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Ran `cargo fmt` and `cargo clippy`
- [ ] Updated CHANGELOG.md if applicable

### PR Description

Include:
- Clear description of the change
- Motivation and context
- How it was tested
- Breaking changes (if any)
- Related issues

### Review Process

1. Maintainers will review your PR
2. Address feedback and requested changes
3. Once approved, your PR will be merged

## Communication

- GitHub Issues for bug reports and feature requests
- GitHub Discussions for questions and ideas
- Pull Requests for code contributions

## Areas for Contribution

### High Priority

- [ ] Complete tensor implementations
- [ ] Fix linked list pop/clear operations
- [ ] Implement graph mutation operations
- [ ] Serialization/deserialization
- [ ] Python bindings

### Medium Priority

- [ ] Graph traversal utilities
- [ ] Optimization passes
- [ ] Performance benchmarks
- [ ] More comprehensive tests

### Documentation

- [ ] More examples
- [ ] Tutorial documentation
- [ ] API documentation improvements
- [ ] Design document updates

## License

By contributing to ONNX IR Rust, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to open an issue or discussion if you have questions!
