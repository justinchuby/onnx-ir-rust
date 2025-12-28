// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the ONNX IR library.

use onnx_ir_core::{
    attribute::{Attr, AttrValue},
    enums::DataType,
    graph::Graph,
    model::Model,
    node::Node,
    shape::Shape,
    tensor::{Tensor, TensorProtocol},
    value::Value,
};

#[test]
fn test_build_simple_model() {
    // Create a simple graph with Add operation
    let mut graph = Graph::new();
    graph.name = Some("test_graph".to_string());

    // Create input values
    let input_x = Value::new("x");
    let input_y = Value::new("y");
    let output_z = Value::new("z");

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
    // Create a graph with a Conv operation
    let mut graph = Graph::new();

    // Create values
    let input = Value::new("input");
    let weights = Value::new("weights");
    let output = Value::new("output");

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

    let graph = Graph::new();
    let mut model = Model::new(graph);

    // Add a custom function
    let mut func = Function::new("MyCustomOp", "com.example");
    func.add_input(Value::new("x"));
    func.add_output(Value::new("y"));

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
