// graph_test.cpp
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "graph.hpp"

using namespace tensor_compiler;

// ----------------------------- Constructors ------------------------------------

TEST(Graph, DefaultConstructorInitializesEmptyGraph) {
    Graph g;

    EXPECT_TRUE(g.get_name().empty());
    EXPECT_TRUE(g.get_tensors().empty());
    EXPECT_TRUE(g.get_nodes().empty());
    EXPECT_TRUE(g.get_inputs().empty());
    EXPECT_TRUE(g.get_outputs().empty());
}

TEST(Graph, NameConstructorStoresName) {
    Graph g("MyGraph");

    EXPECT_EQ(g.get_name(), "MyGraph");
    EXPECT_TRUE(g.get_tensors().empty());
    EXPECT_TRUE(g.get_nodes().empty());
    EXPECT_TRUE(g.get_inputs().empty());
    EXPECT_TRUE(g.get_outputs().empty());
}

// -------------------------------- Getters -------------------------------------

TEST(Graph, GettersReturnConstReferencesToInternals) {
    Graph g("G");

    g.set_inputs({"in1", "in2"});
    g.set_outputs({"out1"});

    Tensor t = Tensor::create("X", {1, 2}, {1.0f, 2.0f}, Tensor_kind::input);
    g.add_tensor(t);

    Node n("N", "Add", 7);
    n.set_inputs(std::vector<std::string>{"in1", "in2"});
    n.set_outputs(std::vector<std::string>{"out1"});
    g.add_node(n);

    EXPECT_EQ(g.get_name(), "G");

    ASSERT_EQ(g.get_inputs().size(), 2u);
    EXPECT_EQ(g.get_inputs()[0], "in1");
    EXPECT_EQ(g.get_inputs()[1], "in2");

    ASSERT_EQ(g.get_outputs().size(), 1u);
    EXPECT_EQ(g.get_outputs()[0], "out1");

    ASSERT_EQ(g.get_nodes().size(), 1u);
    EXPECT_EQ(g.get_nodes()[0].get_name(), "N");
    EXPECT_EQ(g.get_nodes()[0].get_opcode(), "Add");
    EXPECT_EQ(g.get_nodes()[0].get_id(), 7u);

    ASSERT_EQ(g.get_tensors().size(), 1u);
    EXPECT_TRUE(g.get_tensors().find("X") != g.get_tensors().end());
    EXPECT_EQ(g.get_tensors().at("X").get_name(), "X");
}

// -------------------------------- set_name ------------------------------------

TEST(Graph, SetNameUpdatesName) {
    Graph g("A");
    EXPECT_EQ(g.get_name(), "A");

    g.set_name("B");
    EXPECT_EQ(g.get_name(), "B");
}

// -------------------------------- set_inputs ----------------------------------

TEST(Graph, SetInputsStoresVectorAndOverwritesPrevious) {
    Graph g;

    g.set_inputs({"a", "b"});
    ASSERT_EQ(g.get_inputs().size(), 2u);
    EXPECT_EQ(g.get_inputs()[0], "a");
    EXPECT_EQ(g.get_inputs()[1], "b");

    g.set_inputs({"x"});
    ASSERT_EQ(g.get_inputs().size(), 1u);
    EXPECT_EQ(g.get_inputs()[0], "x");
}

// -------------------------------- set_outputs ---------------------------------

TEST(Graph, SetOutputsStoresVectorAndOverwritesPrevious) {
    Graph g;

    g.set_outputs({"o1", "o2"});
    ASSERT_EQ(g.get_outputs().size(), 2u);
    EXPECT_EQ(g.get_outputs()[0], "o1");
    EXPECT_EQ(g.get_outputs()[1], "o2");

    g.set_outputs({});
    EXPECT_TRUE(g.get_outputs().empty());
}

// -------------------------------- add_tensor ----------------------------------

TEST(Graph, AddTensorInsertsTensorByName) {
    Graph g;

    Tensor t = Tensor::create("W", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, Tensor_kind::constant);
    g.add_tensor(t);

    ASSERT_NE(g.get_tensor("W"), nullptr);
    EXPECT_EQ(g.get_tensor("W")->get_name(), "W");
    EXPECT_TRUE(g.get_tensor("missing") == nullptr);
}

TEST(Graph, AddTensorWithSameNameOverwritesExisting) {
    Graph g;

    Tensor t1 = Tensor::create("X", {1}, {1.0f}, Tensor_kind::input);
    Tensor t2 = Tensor::create("X", {2}, {2.0f, 3.0f}, Tensor_kind::input);

    g.add_tensor(t1);
    ASSERT_NE(g.get_tensor("X"), nullptr);
    ASSERT_EQ(g.get_tensor("X")->get_shape().size(), 1u);
    EXPECT_EQ(g.get_tensor("X")->get_shape()[0], 1);

    g.add_tensor(t2);
    ASSERT_NE(g.get_tensor("X"), nullptr);
    ASSERT_EQ(g.get_tensor("X")->get_shape().size(), 1u);
    EXPECT_EQ(g.get_tensor("X")->get_shape()[0], 2);
}

// -------------------------------- add_node ------------------------------------

TEST(Graph, AddNodeAppendsToNodesVector) {
    Graph g;

    g.add_node(Node("N1", "Relu", 1));
    g.add_node(Node("N2", "Add", 2));

    ASSERT_EQ(g.get_nodes().size(), 2u);
    EXPECT_EQ(g.get_nodes()[0].get_name(), "N1");
    EXPECT_EQ(g.get_nodes()[0].get_opcode(), "Relu");
    EXPECT_EQ(g.get_nodes()[0].get_id(), 1u);

    EXPECT_EQ(g.get_nodes()[1].get_name(), "N2");
    EXPECT_EQ(g.get_nodes()[1].get_opcode(), "Add");
    EXPECT_EQ(g.get_nodes()[1].get_id(), 2u);
}

// -------------------------------- add_input -----------------------------------

TEST(Graph, AddInputAppendsInputs) {
    Graph g;

    g.add_input("a");
    g.add_input("b");

    ASSERT_EQ(g.get_inputs().size(), 2u);
    EXPECT_EQ(g.get_inputs()[0], "a");
    EXPECT_EQ(g.get_inputs()[1], "b");
}

// -------------------------------- add_output ----------------------------------

TEST(Graph, AddOutputAppendsOutputs) {
    Graph g;

    g.add_output("y");
    g.add_output("z");

    ASSERT_EQ(g.get_outputs().size(), 2u);
    EXPECT_EQ(g.get_outputs()[0], "y");
    EXPECT_EQ(g.get_outputs()[1], "z");
}

// -------------------------------- get_tensor ----------------------------------

TEST(Graph, GetTensorReturnsPointerToStoredTensorOrNull) {
    Graph g;

    EXPECT_EQ(g.get_tensor("X"), nullptr);

    g.add_tensor(Tensor::create("X", {1, 2}, {1.0f, 2.0f}, Tensor_kind::input));
    const Tensor* px = g.get_tensor("X");

    ASSERT_NE(px, nullptr);
    EXPECT_EQ(px->get_name(), "X");
    EXPECT_EQ(px->get_kind(), Tensor_kind::input);

    EXPECT_EQ(g.get_tensor("missing"), nullptr);
}

TEST(Graph, GetTensorPointerRemainsValidWhileGraphUnmodified) {
    Graph g;
    g.add_tensor(Tensor::create("A", {1}, {1.0f}, Tensor_kind::input));

    const Tensor* pa = g.get_tensor("A");
    ASSERT_NE(pa, nullptr);
    EXPECT_EQ(pa->get_name(), "A");

    const Tensor* pa2 = g.get_tensor("A");
    EXPECT_EQ(pa, pa2);
}