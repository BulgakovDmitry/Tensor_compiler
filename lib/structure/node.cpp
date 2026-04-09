#include "structure/node.h"

namespace tensor_compiler {

// ----------------------------------------------------------------------------
// @section Implementations
// Implementation of node methods.
// ----------------------------------------------------------------------------
void Node::setName(const std::string &name) { name_ = name; }

Node::node_id Node::id() const { return id_; }
const std::string &Node::opcode() const { return opcode_; }
const std::string &Node::name() const { return name_; }
const std::vector<std::string> &Node::inputs() const {
    return inputs_;
}
const std::vector<std::string> &Node::outputs() const {
    return outputs_;
}
const Attributes &Node::attributes() const { return attributes_; }

void Node::setInputs(const std::vector<std::string> &inputs) {
    inputs_ = inputs;
}

void Node::setInputs(const name_t &inputs) {
    inputs_.clear();
    inputs_.reserve(static_cast<std::size_t>(inputs.size()));
    for (const auto &s : inputs)
        inputs_.push_back(s);
}

void Node::setOutputs(const std::vector<std::string> &outputs) {
    outputs_ = outputs;
}

void Node::setOutputs(const name_t &outputs) {
    outputs_.clear();
    outputs_.reserve(static_cast<std::size_t>(outputs.size()));
    for (const auto &s : outputs)
        outputs_.push_back(s);
}

void Node::parseAttributes(const onnx::NodeProto &node) {
    for (const auto &attr : node.attribute()) {
        const std::string &name = attr.name();

        switch (attr.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT: {
            setAttribute(name, attr.f());
            break;
        }

        case onnx::AttributeProto_AttributeType_INT: {
            setAttribute(name, static_cast<int64_t>(attr.i()));
            break;
        }

        case onnx::AttributeProto_AttributeType_STRING: {
            setAttribute(name, attr.s());
            break;
        }

        case onnx::AttributeProto_AttributeType_FLOATS: {
            std::vector<float> v;
            v.reserve(attr.floats_size());
            for (int i = 0; i < attr.floats_size(); ++i)
                v.push_back(attr.floats(i));
            setAttribute(name, v);
            break;
        }

        case onnx::AttributeProto_AttributeType_INTS: {
            std::vector<int64_t> v;
            v.reserve(attr.ints_size());
            for (int i = 0; i < attr.ints_size(); ++i)
                v.push_back(static_cast<int64_t>(attr.ints(i)));
            setAttribute(name, v);
            break;
        }

        default:
            break;
        }
    }
}

void Node::addInput(const std::string &input) {
    inputs_.push_back(input);
}
void Node::addOutput(const std::string &output) {
    outputs_.push_back(output);
}

void Node::setAttribute(const std::string &name,
                                const Attribute::AttrValue &value) {
    attributes_[name] = Attribute{name, value};
}

bool Node::hasAttribute(const std::string &name) const {
    return attributes_.find(name) != attributes_.end();
}

} // namespace tensor_compiler
