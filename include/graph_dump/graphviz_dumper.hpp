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

static std::string escape_html(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

static std::string tensor_type_to_string(int type) {
    switch (type) {
    case onnx::TensorProto_DataType_UNDEFINED:
        return "undefined";
    case onnx::TensorProto_DataType_FLOAT:
        return "float";
    case onnx::TensorProto_DataType_UINT8:
        return "uint8";
    case onnx::TensorProto_DataType_INT8:
        return "int8";
    case onnx::TensorProto_DataType_UINT16:
        return "uint16";
    case onnx::TensorProto_DataType_INT16:
        return "int16";
    case onnx::TensorProto_DataType_INT32:
        return "int32";
    case onnx::TensorProto_DataType_INT64:
        return "int64";
    case onnx::TensorProto_DataType_STRING:
        return "string";
    case onnx::TensorProto_DataType_BOOL:
        return "bool";
    case onnx::TensorProto_DataType_FLOAT16:
        return "float16";
    case onnx::TensorProto_DataType_DOUBLE:
        return "double";
    case onnx::TensorProto_DataType_UINT32:
        return "uint32";
    case onnx::TensorProto_DataType_UINT64:
        return "uint64";
    case onnx::TensorProto_DataType_COMPLEX64:
        return "complex64";
    case onnx::TensorProto_DataType_COMPLEX128:
        return "complex128";
    case onnx::TensorProto_DataType_BFLOAT16:
        return "bfloat16";
    default:
        return "unknown";
    }
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
            std::string bgcolor;

            switch (tensor.get_kind()) {
            case Tensor_kind::input: {
                kind_str = "input";
                bgcolor = "lightblue";
                break;
            }
            case Tensor_kind::output: {
                kind_str = "output";
                bgcolor = "lightgreen";
                break;
            }
            case Tensor_kind::constant: {
                kind_str = "constant";
                bgcolor = "lightgrey";
                break;
            }
            case Tensor_kind::intermediate: {
                kind_str = "intermediate";
                bgcolor = "pink";
                break;
            }
            default:
                throw std::runtime_error("unknown tentor kind\n");
                break;
            }

            // Forming an HTML table
            std::string label = "<";
            label += "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" "
                     "cellpadding=\"4\">";

            // String with name
            label += "<tr><td bgcolor=\"" + bgcolor + "\"><b>" +
                     escape_html(name) + "</b></td></tr>";

            // String with data type
            label += "<tr><td align=\"left\">type: " +
                     tensor_type_to_string(tensor.get_type()) + "</td></tr>";

            // String with dimension
            label += "<tr><td align=\"left\">shape: [";
            const auto &shape = tensor.get_shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0)
                    label += ",";
                label += std::to_string(shape[i]);
            }
            label += "]</td></tr>";

            // String with kind
            label += "<tr><td align=\"left\">" + escape_html(kind_str) +
                     "</td></tr>";

            label += "</table>>";

            gv << "    " << tensor_id << " [shape=plaintext, label=" << label
               << ", color=\"#252A34\", penwidth=2.5];\n";
        }
    }

    static void dump_nodes(const Graph &g, std::ostream &gv) {
        for (const auto &node : g.get_nodes()) {
            std::string node_id = "node_" + escape_dot(node.get_name());

            std::string label = "<";
            label += "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" "
                     "cellpadding=\"4\">";

            // String with opcode (bold)
            label += "<tr><td bgcolor=\"lightcoral\"><b>" +
                     escape_html(node.get_opcode()) + "</b></td></tr>";

            // String with node name
            label +=
                "<tr><td align=\"left\">name: " + escape_html(node.get_name()) +
                "</td></tr>";

            // String with id
            label +=
                "<tr><td align=\"left\">id: " + std::to_string(node.get_id()) +
                "</td></tr>";

            // Add attributes if any.
            const auto &attrs = node.get_attributes();
            if (!attrs.empty())
                dump_attributes(attrs, label);

            label += "</table>>";

            gv << "    " << node_id << " [shape=plaintext, label=" << label
               << ", color=\"#252A34\", penwidth=2.5];\n";
        }
        gv << "\n";
    }

    static void dump_attributes(const Attributes &attrs, std::string &label) {
        for (const auto &[attr_name, attr] : attrs) {
            std::string attr_str = escape_html(attr_name) + "=";
            std::visit(
                [&attr_str](const auto &val) {
                    using T = std::decay_t<decltype(val)>;
                    if constexpr (std::is_same_v<T, float>) {
                        attr_str += std::to_string(val);
                    } else if constexpr (std::is_same_v<T, int64_t>) {
                        attr_str += std::to_string(val);
                    } else if constexpr (std::is_same_v<T, std::string>) {
                        attr_str += "\"" + escape_html(val) + "\"";
                    } else if constexpr (std::is_same_v<T,
                                                        std::vector<float>>) {
                        attr_str += "[";
                        for (size_t i = 0; i < val.size(); ++i) {
                            if (i > 0)
                                attr_str += ",";
                            attr_str += std::to_string(val[i]);
                        }
                        attr_str += "]";
                    } else if constexpr (std::is_same_v<T,
                                                        std::vector<int64_t>>) {
                        attr_str += "[";
                        for (size_t i = 0; i < val.size(); ++i) {
                            if (i > 0)
                                attr_str += ",";
                            attr_str += std::to_string(val[i]);
                        }
                        attr_str += "]";
                    }
                },
                attr.get_value());
            label += "<tr><td align=\"left\">" + attr_str + "</td></tr>";
        }
    }
};

} // namespace tensor_compiler

#endif // INCLUDE_GRAPHVIZ_DUMPER_HPP
