#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "node.h"

using namespace tensor_compiler;

// ----------------------------- Constructors ------------------------------------

TEST(Node, ConstructorStoresNameOpcodeAndIdDefault) {
    Node n("MyNode", "Add");

    EXPECT_EQ(n.name(), "MyNode");
    EXPECT_EQ(n.opcode(), "Add");
    EXPECT_EQ(n.id(), 0u);

    EXPECT_TRUE(n.inputs().empty());
    EXPECT_TRUE(n.outputs().empty());
    EXPECT_TRUE(n.attributes().empty());
}

TEST(Node, ConstructorStoresExplicitId) {
    Node n("N", "Relu", 123);

    EXPECT_EQ(n.name(), "N");
    EXPECT_EQ(n.opcode(), "Relu");
    EXPECT_EQ(n.id(), 123u);
}

// -------------------------------- setName ------------------------------------

TEST(Node, SetNameUpdatesName) {
    Node n("A", "Mul", 1);
    EXPECT_EQ(n.name(), "A");

    n.setName("B");
    EXPECT_EQ(n.name(), "B");
    EXPECT_EQ(n.opcode(), "Mul");
    EXPECT_EQ(n.id(), 1u);
}

// -------------------------------- setInputs ----------------------------------

TEST(Node, SetInputsFromVectorStoresInputs) {
    Node n("N", "Op");

    std::vector<std::string> in = {"x", "y", "z"};
    n.setInputs(in);

    const auto& got = n.inputs();
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], "x");
    EXPECT_EQ(got[1], "y");
    EXPECT_EQ(got[2], "z");
}

TEST(Node, SetInputsFromRepeatedPtrFieldCopiesAndClearsPrevious) {
    Node n("N", "Op");
    n.setInputs(std::vector<std::string>{"old1", "old2"});
    ASSERT_EQ(n.inputs().size(), 2u);

    Node::name_t rp;
    rp.Add("a");
    rp.Add("b");
    rp.Add("c");

    n.setInputs(rp);

    const auto& got = n.inputs();
    ASSERT_EQ(got.size(), 3u);
    EXPECT_EQ(got[0], "a");
    EXPECT_EQ(got[1], "b");
    EXPECT_EQ(got[2], "c");
}

// -------------------------------- setOutputs ---------------------------------

TEST(Node, SetOutputsFromVectorStoresOutputs) {
    Node n("N", "Op");

    std::vector<std::string> out = {"o1", "o2"};
    n.setOutputs(out);

    const auto& got = n.outputs();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "o1");
    EXPECT_EQ(got[1], "o2");
}

TEST(Node, SetOutputsFromRepeatedPtrFieldCopiesAndClearsPrevious) {
    Node n("N", "Op");
    n.setOutputs(std::vector<std::string>{"prev"});
    ASSERT_EQ(n.outputs().size(), 1u);

    Node::name_t rp;
    rp.Add("u");
    rp.Add("v");

    n.setOutputs(rp);

    const auto& got = n.outputs();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0], "u");
    EXPECT_EQ(got[1], "v");
}

// ------------------------------ setAttribute ----------------------------------

TEST(Node, SetAttributeInsertsAndCanOverwrite) {
    Node n("N", "Op");

    EXPECT_FALSE(n.hasAttribute("alpha"));

    n.setAttribute("alpha", Attribute::AttrValue{1.5f});
    EXPECT_TRUE(n.hasAttribute("alpha"));

    const auto& attrs1 = n.attributes();
    ASSERT_TRUE(attrs1.find("alpha") != attrs1.end());
    ASSERT_TRUE(std::holds_alternative<float>(attrs1.at("alpha").value()));
    EXPECT_FLOAT_EQ(std::get<float>(attrs1.at("alpha").value()), 1.5f);

    n.setAttribute("alpha", Attribute::AttrValue{int64_t{7}});

    const auto& attrs2 = n.attributes();
    ASSERT_TRUE(attrs2.find("alpha") != attrs2.end());
    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs2.at("alpha").value()));
    EXPECT_EQ(std::get<int64_t>(attrs2.at("alpha").value()), 7);
}

TEST(Node, HasAttributeReflectsPresence) {
    Node n("N", "Op");

    EXPECT_FALSE(n.hasAttribute("missing"));
    n.setAttribute("present", Attribute::AttrValue{std::string{"x"}});
    EXPECT_TRUE(n.hasAttribute("present"));
    EXPECT_FALSE(n.hasAttribute("missing"));
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
    n.parseAttributes(proto);

    EXPECT_TRUE(n.hasAttribute("f"));
    EXPECT_TRUE(n.hasAttribute("i"));
    EXPECT_TRUE(n.hasAttribute("s"));
    EXPECT_TRUE(n.hasAttribute("fs"));
    EXPECT_TRUE(n.hasAttribute("is"));

    const auto& attrs = n.attributes();

    ASSERT_TRUE(std::holds_alternative<float>(attrs.at("f").value()));
    EXPECT_FLOAT_EQ(std::get<float>(attrs.at("f").value()), 2.25f);

    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs.at("i").value()));
    EXPECT_EQ(std::get<int64_t>(attrs.at("i").value()), 42);

    ASSERT_TRUE(std::holds_alternative<std::string>(attrs.at("s").value()));
    EXPECT_EQ(std::get<std::string>(attrs.at("s").value()), "hello");

    ASSERT_TRUE(std::holds_alternative<std::vector<float>>(attrs.at("fs").value()));
    const auto& vf = std::get<std::vector<float>>(attrs.at("fs").value());
    ASSERT_EQ(vf.size(), 3u);
    EXPECT_FLOAT_EQ(vf[0], 0.5f);
    EXPECT_FLOAT_EQ(vf[1], -1.0f);
    EXPECT_FLOAT_EQ(vf[2], 3.25f);

    ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(attrs.at("is").value()));
    const auto& vi = std::get<std::vector<int64_t>>(attrs.at("is").value());
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
    n.parseAttributes(proto);

    EXPECT_TRUE(n.hasAttribute("ok"));
    EXPECT_FALSE(n.hasAttribute("ignored"));
}

TEST(Node, ParseAttributesOverwritesExistingAttributeWithSameName) {
    Node n("N", "Op");
    n.setAttribute("axis", Attribute::AttrValue{int64_t{0}});
    ASSERT_TRUE(n.hasAttribute("axis"));
    {
        const auto& attrs = n.attributes();
        ASSERT_TRUE(std::holds_alternative<int64_t>(attrs.at("axis").value()));
        EXPECT_EQ(std::get<int64_t>(attrs.at("axis").value()), 0);
    }

    onnx::NodeProto proto;
    auto* a = proto.add_attribute();
    a->set_name("axis");
    a->set_type(onnx::AttributeProto_AttributeType_INT);
    a->set_i(3);

    n.parseAttributes(proto);

    const auto& attrs2 = n.attributes();
    ASSERT_TRUE(std::holds_alternative<int64_t>(attrs2.at("axis").value()));
    EXPECT_EQ(std::get<int64_t>(attrs2.at("axis").value()), 3);
}
