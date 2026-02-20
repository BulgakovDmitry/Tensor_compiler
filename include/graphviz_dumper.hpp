#ifndef INCLUDE_GRAPHVIZ_DUMPER_HPP
#define INCLUDE_GRAPHVIZ_DUMPER_HPP

#include "structure/attribute.hpp"
#include "structure/graph.hpp"
#include "structure/node.hpp"
#include "structure/tensor.hpp"
#include <iostream>
#include <ostream>
#include <string>

namespace tensor_compiler {

// Helper function for escaping special characters in DOT
static std::string escape_dot(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\' || c == '{' || c == '}' || c == '|' ||
            c == '<' || c == '>') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

class Graphviz_dumper {
  public:
    static void dump(const Graph &g, std::ostream &gv) {
        gv << "digraph G {\n"
           << "    rankdir=TB;\n"
           << "    node [style=filled, fontname=\"Helvetica\", "
              "fontcolor=darkblue, "
           << "fillcolor=peachpuff, color=\"#252A34\", penwidth=2.5];\n"
           << "    bgcolor=\"lemonchiffon\";\n\n";

        make_styles(gv);
        // 1. Output all tensors as nodes
        dump_tensors(g, gv);
        // 2. Output all operation nodes
        dump_nodes(g, gv);
        // 3. Creating edges: from tensors to operations (inputs) and from
        // operations to tensors (outputs)
        add_edges(g, gv);
        // 4. Select the inputs and outputs of the entire graph
        add_inputs_outputs(g, gv);

        gv << "}\n";
    }

  private:
    static void make_styles(std::ostream &gv) {
        // Styles for tensors depending on their type
        gv << "    // Tensor styles\n"
           << "    tensor_input [shape=box, style=filled, fillcolor=lightblue, "
              "color=\"#252A34\"];\n"
           << "    tensor_output [shape=box, style=filled, "
              "fillcolor=lightgreen, color=\"#252A34\"];\n"
           << "    tensor_constant [shape=box, style=filled, "
              "fillcolor=lightgrey, color=\"#252A34\"];\n"
           << "    tensor_intermediate [shape=box, style=filled, "
              "fillcolor=white, color=\"#252A34\"];\n\n";

        // Style for operation nodes
        gv << "    // Node style\n"
           << "    node [shape=record, style=filled, fillcolor=peachpuff, "
              "fontcolor=darkblue];\n\n";
        gv << "\n";
    }

    static void add_inputs_outputs(const Graph &g, std::ostream &gv) {
        gv << "    // Mark graph inputs and outputs\n";
        for (const auto &in : g.get_inputs()) {
            std::string tid = "tensor_" + escape_dot(in);
            gv << "    " << tid
               << " [peripheries=2];\n"; // double frame for inputs
        }
        for (const auto &out : g.get_outputs()) {
            std::string tid = "tensor_" + escape_dot(out);
            gv << "    " << tid
               << " [peripheries=2];\n"; // double frame for outputs
        }
    }

    static void add_edges(const Graph &g, std::ostream &gv) {
        for (const auto &node : g.get_nodes()) {
            std::string node_id = "node_" + escape_dot(node.get_name());

            // Input tensors → node
            for (const auto &input_name : node.get_inputs()) {
                std::string tensor_id = "tensor_" + escape_dot(input_name);
                gv << "    " << tensor_id << " -> " << node_id << ";\n";
            }

            // Node → output tensors
            for (const auto &output_name : node.get_outputs()) {
                std::string tensor_id = "tensor_" + escape_dot(output_name);
                gv << "    " << node_id << " -> " << tensor_id << ";\n";
            }
        }
        gv << "\n";
    }

    static void dump_tensors(const Graph &g, std::ostream &gv) {
        for (const auto &[name, tensor] : g.get_tensors()) {
            std::string tensor_id = "tensor_" + escape_dot(name);
            std::string kind_str;
            std::string style;

            switch (tensor.get_kind()) {
            case Tensor_kind::input:
                kind_str = "input";
                style = "tensor_input";
                break;
            case Tensor_kind::output:
                kind_str = "output";
                style = "tensor_output";
                break;
            case Tensor_kind::constant:
                kind_str = "constant";
                style = "tensor_constant";
                break;
            default:
                kind_str = "intermediate";
                style = "tensor_intermediate";
                break;
            }

            // Forming a label: name, data type, dimension, kind
            std::string label =
                "{" + escape_dot(name) +
                " | type: " + std::to_string(tensor.get_type()) + " | shape: [";
            const auto &shape = tensor.get_shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0)
                    label += ",";
                label += std::to_string(shape[i]);
            }
            label += "] | " + kind_str + "}";

            gv << "    " << tensor_id << " [" << style << ", label=\"" << label
               << "\"];\n";
        }
    }

    static void dump_nodes(const Graph &g, std::ostream &gv) {
        for (const auto &node : g.get_nodes()) {
            std::string node_id = "node_" + escape_dot(node.get_name());
            std::string label = "{" + escape_dot(node.get_opcode()) +
                                " | name: " + escape_dot(node.get_name()) +
                                " | id: " + std::to_string(node.get_id());

            // Add attributes if any.
            const auto &attrs = node.get_attributes();
            if (!attrs.empty())
                dump_attributes(attrs, gv, label);

            label += "}";

            gv << "    " << node_id << " [label=\"" << label << "\"];\n";
        }
        gv << "\n";
    }

    static void dump_attributes(const Node::Attributes &attrs, std::ostream &gv,
                                std::string &label) {
        label += " | attributes: ";
        bool first = true;
        for (const auto &[attr_name, attr] : attrs) {
            if (!first)
                label += " ";
            first = false;
            label += escape_dot(attr_name) + "=";
            // Depending on the value type, add a representation
            std::visit(
                [&label](const auto &val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, float>) {
                        label += std::to_string(val);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        label += std::to_string(val);
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        label += "\"" + escape_dot(val) + "\"";
                    } else if constexpr (std::is_same_v<T,
                                                        std::vector<float>>) {
                        label += "[";
                        for (size_t i = 0; i < val.size(); ++i) {
                            if (i > 0)
                                label += ",";
                            label += std::to_string(val[i]);
                        }
                        label += "]";
                    } else if constexpr (std::is_same_v<T,
                                                        std::vector<int64_t>>) {
                        label += "[";
                        for (size_t i = 0; i < val.size(); ++i) {
                            if (i > 0)
                                label += ",";
                            label += std::to_string(val[i]);
                        }
                        label += "]";
                    }
                },
                attr.get_value());
        }
    }
};

} // namespace tensor_compiler

#endif // INCLUDE_GRAPHVIZ_DUMPER_HPP
