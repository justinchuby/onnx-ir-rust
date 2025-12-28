// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the ONNX IR library.

use onnx_ir_core::{
    attribute::{Attr, AttrValue},
    enums::DataType,
    graph::Graph,
    model::Model,
    node::{node_add_input, node_add_output, Node},
    shape::Shape,
    tensor::{Tensor, TensorProtocol},
    value::Value,
};

#[test]
fn test_build_simple_model() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Create a simple graph with Add operation
    let mut graph = Graph::new();
    graph.name = Some("test_graph".to_string());

    // Create input values
    let input_x = Rc::new(RefCell::new(Value::new("x")));
    let input_y = Rc::new(RefCell::new(Value::new("y")));
    let output_z = Rc::new(RefCell::new(Value::new("z")));

    // Create an Add node
    let mut node = Node::new("Add");
    node.name = Some("add_node".to_string());
    node.add_input(input_x);
    node.add_input(input_y);
    node.add_output(output_z);

    // Add node to graph
    graph.append(node);

    // Create model
    let mut model = Model::new(graph);
    model.producer_name = Some("test_producer".to_string());
    model.set_opset_import("", 18);

    // Verify model structure
    assert_eq!(model.producer_name, Some("test_producer".to_string()));
    assert_eq!(model.get_opset_version(""), Some(18));
    assert_eq!(model.graph.num_nodes(), 1);
}

#[test]
fn test_build_conv_model() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Create a graph with a Conv operation
    let mut graph = Graph::new();

    // Create values
    let input = Rc::new(RefCell::new(Value::new("input")));
    let weights = Rc::new(RefCell::new(Value::new("weights")));
    let output = Rc::new(RefCell::new(Value::new("output")));

    // Create Conv node with attributes
    let mut conv_node = Node::new("Conv");
    conv_node.add_input(input);
    conv_node.add_input(weights);
    conv_node.add_output(output);

    // Add attributes
    conv_node.set_attribute(Attr::ints("kernel_shape", vec![3, 3]));
    conv_node.set_attribute(Attr::ints("pads", vec![1, 1, 1, 1]));
    conv_node.set_attribute(Attr::ints("strides", vec![1, 1]));

    // Verify attributes
    assert_eq!(conv_node.num_inputs(), 2);
    assert_eq!(conv_node.num_outputs(), 1);
    assert!(conv_node.has_attributes());
    assert!(conv_node.get_attribute("kernel_shape").is_some());

    graph.append(conv_node);
    assert_eq!(graph.num_nodes(), 1);
}

#[test]
fn test_tensor_creation() {
    // Create a tensor with concrete data
    let shape = Shape::new(vec![2, 3]);
    let tensor = Tensor::new(DataType::Float, shape.clone());

    assert_eq!(tensor.size(), 6);
    assert_eq!(tensor.nbytes(), 24); // 6 floats * 4 bytes
    assert_eq!(tensor.shape(), &shape);
}

#[test]
fn test_graph_mutation() {
    // Test graph mutation operations
    let mut graph = Graph::new();

    // Add multiple nodes
    graph.append(Node::new("Add"));
    graph.append(Node::new("Mul"));
    graph.prepend(Node::new("Sub"));

    assert_eq!(graph.num_nodes(), 3);

    // Verify iteration order
    let op_types: Vec<_> = graph.iter_nodes().map(|n| n.op_type.as_str()).collect();
    assert_eq!(op_types, vec!["Sub", "Add", "Mul"]);

    // Remove nodes
    let first = graph.pop_first();
    assert!(first.is_some());
    assert_eq!(first.unwrap().op_type, "Sub");
    assert_eq!(graph.num_nodes(), 2);

    // Clear all nodes
    graph.clear_nodes();
    assert!(graph.is_empty());
}

#[test]
fn test_model_with_function() {
    use onnx_ir_core::function::Function;
    use std::cell::RefCell;
    use std::rc::Rc;

    let graph = Graph::new();
    let mut model = Model::new(graph);

    // Add a custom function
    let mut func = Function::new("MyCustomOp", "com.example");
    func.add_input(Rc::new(RefCell::new(Value::new("x"))));
    func.add_output(Rc::new(RefCell::new(Value::new("y"))));

    model.add_function(func);

    // Verify function is stored
    assert!(model.get_function("com.example", "MyCustomOp").is_some());
}

#[test]
fn test_attribute_values() {
    // Test different attribute value types
    let float_attr = Attr::float("alpha", 0.5);
    assert!(matches!(float_attr.value, AttrValue::Float(v) if v == 0.5));

    let int_attr = Attr::int("axis", 1);
    assert!(matches!(int_attr.value, AttrValue::Int(1)));

    let string_attr = Attr::string("mode", "NOTSET".to_string());
    assert!(matches!(
        string_attr.value,
        AttrValue::String(ref s) if s == "NOTSET"
    ));

    let ints_attr = Attr::ints("shape", vec![1, 2, 3]);
    assert!(matches!(
        ints_attr.value,
        AttrValue::Ints(ref v) if v == &vec![1, 2, 3]
    ));
}

#[test]
fn test_value_usage_tracking_integration() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Create a graph with multiple nodes sharing values
    let _graph = Graph::new();

    // Create shared values
    let input = Rc::new(RefCell::new(Value::new("input")));
    let intermediate1 = Rc::new(RefCell::new(Value::new("intermediate1")));
    let intermediate2 = Rc::new(RefCell::new(Value::new("intermediate2")));
    let output = Rc::new(RefCell::new(Value::new("output")));

    // Create first node: Relu(input) -> intermediate1
    let relu_node = Rc::new(RefCell::new(Node::new("Relu")));
    node_add_input(&relu_node, &input);
    node_add_output(&relu_node, &intermediate1);

    // Create second node: Add(intermediate1, intermediate1) -> intermediate2
    let add_node = Rc::new(RefCell::new(Node::new("Add")));
    node_add_input(&add_node, &intermediate1);
    node_add_input(&add_node, &intermediate1);
    node_add_output(&add_node, &intermediate2);

    // Create third node: Mul(intermediate2, input) -> output
    let mul_node = Rc::new(RefCell::new(Node::new("Mul")));
    node_add_input(&mul_node, &intermediate2);
    node_add_input(&mul_node, &input);
    node_add_output(&mul_node, &output);

    // Verify usage tracking is automatically set up
    assert!(input.borrow().producer().is_none()); // input has no producer
    assert_eq!(input.borrow().num_uses(), 2); // used by Relu and Mul

    assert!(intermediate1.borrow().producer().is_some());
    assert_eq!(intermediate1.borrow().num_uses(), 2); // used twice by Add

    assert!(intermediate2.borrow().producer().is_some());
    assert_eq!(intermediate2.borrow().num_uses(), 1); // used by Mul

    // Test replace_all_uses_with
    // Replace intermediate1 with a new constant value
    let constant_value = Rc::new(RefCell::new(Value::new("constant")));
    intermediate1
        .borrow()
        .replace_all_uses_with(&constant_value);

    // Verify replacement
    assert_eq!(intermediate1.borrow().num_uses(), 0); // no longer used
    assert_eq!(constant_value.borrow().num_uses(), 2); // now used twice

    // Verify Add node now uses constant_value
    {
        let add_borrowed = add_node.borrow();
        assert_eq!(add_borrowed.inputs[0].borrow().name, "constant");
        assert_eq!(add_borrowed.inputs[1].borrow().name, "constant");
    } // add_borrowed is dropped here

    // Test dead consumer cleanup
    drop(add_node); // Drop the Add node

    // Prune dead consumers
    let removed = constant_value.borrow().prune_dead_consumers();
    assert_eq!(removed, 2); // Both uses of constant_value were from the Add node
    assert_eq!(constant_value.borrow().num_uses(), 0); // Add node is gone
}

#[test]
fn test_graph_with_value_tracking() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Build a complete graph with proper value tracking
    let mut graph = Graph::new();
    graph.name = Some("value_tracking_test".to_string());

    // Create values
    let x = Rc::new(RefCell::new(Value::new("x")));
    let y = Rc::new(RefCell::new(Value::new("y")));
    let sum = Rc::new(RefCell::new(Value::new("sum")));

    // Set as graph inputs
    graph.add_input(Rc::clone(&x));
    graph.add_input(Rc::clone(&y));

    // Create Add node with automatic tracking
    let add_node = Rc::new(RefCell::new(Node::new("Add")));
    node_add_input(&add_node, &x);
    node_add_input(&add_node, &y);
    node_add_output(&add_node, &sum);

    // Set as graph output
    graph.add_output(Rc::clone(&sum));

    // Verify graph structure
    assert_eq!(graph.inputs.len(), 2);
    assert_eq!(graph.outputs.len(), 1);
    assert_eq!(graph.inputs[0].borrow().name, "x");
    assert_eq!(graph.outputs[0].borrow().name, "sum");

    // Verify usage tracking is automatically set up
    assert_eq!(x.borrow().num_uses(), 1);
    assert_eq!(y.borrow().num_uses(), 1);
    assert!(sum.borrow().producer().is_some());
}

#[test]
fn test_initializer_usage_tracking() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let mut graph = Graph::new();

    // Create an initializer (constant weights)
    let weights = Rc::new(RefCell::new(Value::new("weights")));
    graph.add_initializer(Rc::clone(&weights));

    // Create input
    let input = Rc::new(RefCell::new(Value::new("input")));
    graph.add_input(Rc::clone(&input));

    // Create output
    let output = Rc::new(RefCell::new(Value::new("output")));

    // Create MatMul node using the initializer with automatic tracking
    let matmul_node = Rc::new(RefCell::new(Node::new("MatMul")));
    node_add_input(&matmul_node, &input);
    node_add_input(&matmul_node, &weights);
    node_add_output(&matmul_node, &output);

    // Verify initializer is tracked
    assert!(graph.get_initializer("weights").is_some());
    assert_eq!(weights.borrow().num_uses(), 1);

    // Remove initializer
    let removed = graph.remove_initializer("weights");
    assert!(removed.is_some());
    assert!(graph.get_initializer("weights").is_none());

    // Value still exists and tracks usage
    assert_eq!(weights.borrow().num_uses(), 1);
}
