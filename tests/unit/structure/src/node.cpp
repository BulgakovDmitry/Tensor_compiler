#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "node.hpp"

using namespace tensor_compiler;

// ----------------------------- Constructors ------------------------------------

TEST(Node, ConstructorStoresNameOpcodeAndIdDefault) {
    Node n("MyNode", "Add");

    EXPECT_EQ(n.get_name(), "MyNode");
    EXPECT_EQ(n.get_opcode(), "Add");
    EXPECT_EQ(n.get_id(), 0u);

    EXPECT_TRUE(n.get_inputs().empty());
    EXPECT_TRUE(n.get_outputs().empty());
    EXPECT_TRUE(n.get_attributes().empty());
}

TEST(Node, ConstructorStoresExplicitId) {
    Node n("N", "Relu", 123);

    EXPECT_EQ(n.get_name(), "N");
    EXPECT_EQ(n.get_opcode(), "Relu");
    EXPECT_EQ(n.get_id(), 123u);
}

// -------------------------------- set_name ------------------------------------

TEST(Node, SetNameUpdatesName) {
    Node n("A", "Mul", 1);
    EXPECT_EQ(n.get_name(), "A");

    n.set_name("B");
    EXPECT_EQ(n.get_name(), "B");
    EXPECT_EQ(n.get_opcode(), "Mul"); 
    EXPECT_EQ(n.get_id(), 1u);        
}

// -------------------------------- set_inputs ----------------------------------

TEST(Node, SetInputsFromVectorStoresInputs) {
    Node n("N", "Op");

    std::vector<std::string> in = {"x", "y", "z"};
    n.set_inputs(in);

    const auto& got = n.get_inputs();
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], "x");
    EXPECT_EQ(got[1], "y");
    EXPECT_EQ(got[2], "z");
}

TEST(Node, SetInputsFromRepeatedPtrFieldCopiesAndClearsPrevious) {
    Node n("N", "Op");
    n.set_inputs(std::vector<std::string>{"old1", "old2"});
    ASSERT_EQ(n.get_inputs().size(), 2u);

    Node::name_t rp;
    rp.Add("a");
    rp.Add("b");
    rp.Add("c");

    n.set_inputs(rp);

    const auto& got = n.get_inputs();
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], "a");
    EXPECT_EQ(got[1], "b");
    EXPECT_EQ(got[2], "c");
}

// -------------------------------- set_outputs ---------------------------------

TEST(Node, SetOutputsFromVectorStoresOutputs) {
    Node n("N", "Op");

    std::vector<std::string> out = {"o1", "o2"};
    n.set_outputs(out);

    const auto& got = n.get_outputs();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "o1");
    EXPECT_EQ(got[1], "o2");
}

TEST(Node, SetOutputsFromRepeatedPtrFieldCopiesAndClearsPrevious) {
    Node n("N", "Op");
    n.set_outputs(std::vector<std::string>{"prev"});
    ASSERT_EQ(n.get_outputs().size(), 1u);

    Node::name_t rp;
    rp.Add("u");
    rp.Add("v");

    n.set_outputs(rp);

    const auto& got = n.get_outputs();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "u");
    EXPECT_EQ(got[1], "v");
}

// ------------------------------ set_attribute ----------------------------------

TEST(Node, SetAttributeInsertsAndCanOverwrite) {
    Node n("N", "Op");

    EXPECT_FALSE(n.has_attribute("alpha"));

    n.set_attribute("alpha", Attribute::AttrValue{1.5f});
    EXPECT_TRUE(n.has_attribute("alpha"));

    const auto& attrs1 = n.get_attributes();
    ASSERT_TRUE(attrs1.find("alpha") != attrs1.end());
    ASSERT_TRUE(std::holds_alternative<float>(attrs1.at("alpha").get_value()));
    EXPECT_FLOAT_EQ(std::get<float>(attrs1.at("alpha").get_value()), 1.5f);

    n.set_attribute("alpha", Attribute::AttrValue{int64_t{7}});

    const auto& attrs2 = n.get_attributes();
    ASSERT_TRUE(attrs2.find("alpha") != attrs2.end());
    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs2.at("alpha").get_value()));
    EXPECT_EQ(std::get<int64_t>(attrs2.at("alpha").get_value()), 7);
}

TEST(Node, HasAttributeReflectsPresence) {
    Node n("N", "Op");

    EXPECT_FALSE(n.has_attribute("missing"));
    n.set_attribute("present", Attribute::AttrValue{std::string{"x"}});
    EXPECT_TRUE(n.has_attribute("present"));
    EXPECT_FALSE(n.has_attribute("missing"));
}

// ----------------------------- parse_attributes --------------------------------

TEST(Node, ParseAttributesParsesFloatIntStringFloatsInts) {
    onnx::NodeProto proto;

    // FLOAT
    {
        auto* a = proto.add_attribute();
        a->set_name("f");
        a->set_type(onnx::AttributeProto_AttributeType_FLOAT);
        a->set_f(2.25f);
    }

    // INT
    {
        auto* a = proto.add_attribute();
        a->set_name("i");
        a->set_type(onnx::AttributeProto_AttributeType_INT);
        a->set_i(42);
    }

    // STRING
    {
        auto* a = proto.add_attribute();
        a->set_name("s");
        a->set_type(onnx::AttributeProto_AttributeType_STRING);
        a->set_s("hello");
    }

    // FLOATS
    {
        auto* a = proto.add_attribute();
        a->set_name("fs");
        a->set_type(onnx::AttributeProto_AttributeType_FLOATS);
        a->add_floats(0.5f);
        a->add_floats(-1.0f);
        a->add_floats(3.25f);
    }

    // INTS
    {
        auto* a = proto.add_attribute();
        a->set_name("is");
        a->set_type(onnx::AttributeProto_AttributeType_INTS);
        a->add_ints(1);
        a->add_ints(2);
        a->add_ints(10000000000LL);
    }

    Node n("N", "Op");
    n.parse_attributes(proto);

    EXPECT_TRUE(n.has_attribute("f"));
    EXPECT_TRUE(n.has_attribute("i"));
    EXPECT_TRUE(n.has_attribute("s"));
    EXPECT_TRUE(n.has_attribute("fs"));
    EXPECT_TRUE(n.has_attribute("is"));

    const auto& attrs = n.get_attributes();

    ASSERT_TRUE(std::holds_alternative<float>(attrs.at("f").get_value()));
    EXPECT_FLOAT_EQ(std::get<float>(attrs.at("f").get_value()), 2.25f);

    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs.at("i").get_value()));
    EXPECT_EQ(std::get<int64_t>(attrs.at("i").get_value()), 42);

    ASSERT_TRUE(std::holds_alternative<std::string>(attrs.at("s").get_value()));
    EXPECT_EQ(std::get<std::string>(attrs.at("s").get_value()), "hello");

    ASSERT_TRUE(std::holds_alternative<std::vector<float>>(attrs.at("fs").get_value()));
    const auto& vf = std::get<std::vector<float>>(attrs.at("fs").get_value());
    ASSERT_EQ(vf.size(), 3u);
    EXPECT_FLOAT_EQ(vf[0], 0.5f);
    EXPECT_FLOAT_EQ(vf[1], -1.0f);
    EXPECT_FLOAT_EQ(vf[2], 3.25f);

    ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(attrs.at("is").get_value()));
    const auto& vi = std::get<std::vector<int64_t>>(attrs.at("is").get_value());
    ASSERT_EQ(vi.size(), 3u);
    EXPECT_EQ(vi[0], 1);
    EXPECT_EQ(vi[1], 2);
    EXPECT_EQ(vi[2], 10000000000LL);
}

TEST(Node, ParseAttributesIgnoresUnsupportedAttributeTypes) {
    onnx::NodeProto proto;

    {
        auto* a = proto.add_attribute();
        a->set_name("ok");
        a->set_type(onnx::AttributeProto_AttributeType_INT);
        a->set_i(1);
    }

    {
        auto* a = proto.add_attribute();
        a->set_name("ignored");
        a->set_type(onnx::AttributeProto_AttributeType_TENSOR);
    }

    Node n("N", "Op");
    n.parse_attributes(proto);

    EXPECT_TRUE(n.has_attribute("ok"));
    EXPECT_FALSE(n.has_attribute("ignored"));
}

TEST(Node, ParseAttributesOverwritesExistingAttributeWithSameName) {
    Node n("N", "Op");
    n.set_attribute("axis", Attribute::AttrValue{int64_t{0}});
    ASSERT_TRUE(n.has_attribute("axis"));
    {
        const auto& attrs = n.get_attributes();
        ASSERT_TRUE(std::holds_alternative<int64_t>(attrs.at("axis").get_value()));
        EXPECT_EQ(std::get<int64_t>(attrs.at("axis").get_value()), 0);
    }

    onnx::NodeProto proto;
    auto* a = proto.add_attribute();
    a->set_name("axis");
    a->set_type(onnx::AttributeProto_AttributeType_INT);
    a->set_i(3);

    n.parse_attributes(proto);

    const auto& attrs2 = n.get_attributes();
    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs2.at("axis").get_value()));
    EXPECT_EQ(std::get<int64_t>(attrs2.at("axis").get_value()), 3);
}