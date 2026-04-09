#ifndef INCLUDE_GRAPHVIZ_DUMPER_HDRIVER_HPP
#define INCLUDE_GRAPHVIZ_DUMPER_HDRIVER_HPP

#include "structure/attribute.h"
#include "structure/graph.h"
#include "structure/node.h"
#include "structure/tensor.h"
#include <iostream>
#include <ostream>
#include <string>

namespace tensor_compiler {

// Helper function for escaping special characters in DOT
static std::string escapeDot(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (c == '"' || c == '\\' || c == '{' || c == '}' || c == '|' || c == '<' ||
        c == '>') {
      out += '\\';
    }
    out += c;
  }
  return out;
}

static std::string escapeHtml(const std::string &s) {
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

static std::string tensorTypeToString(int type) {
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

class GraphvizDumper {
public:
  static void dump(const Graph &g, std::ostream &gv) {
    gv << "digraph G {\n"
       << "    rankdir=TB;\n"
       << "    node [style=filled, fontname=\"Helvetica\", "
          "fontcolor=darkblue, "
       << "fillcolor=peachpuff, color=\"#252A34\", penwidth=2.5];\n"
       << "    bgcolor=\"lemonchiffon\";\n\n";

    // 1. Output all tensors as nodes
    dumpTensors(g, gv);
    // 2. Output all operation nodes
    dumpNodes(g, gv);
    // 3. Creating edges: from tensors to operations (inputs) and from
    // operations to tensors (outputs)
    addEdges(g, gv);
    // 4. Select the inputs and outputs of the entire graph
    addInputsOutputs(g, gv);

    gv << "}\n";
  }

private:
  static void addInputsOutputs(const Graph &g, std::ostream &gv) {
    gv << "    // Mark graph inputs and outputs\n";
    for (const auto &in : g.inputs()) {
      std::string tid = "tensor_" + escapeDot(in);
      gv << "    " << tid << " [peripheries=2];\n"; // double frame for inputs
    }
    for (const auto &out : g.outputs()) {
      std::string tid = "tensor_" + escapeDot(out);
      gv << "    " << tid << " [peripheries=2];\n"; // double frame for outputs
    }
  }

  static void addEdges(const Graph &g, std::ostream &gv) {
    for (const auto &node : g.nodes()) {
      std::string nodeId = "node_" + escapeDot(node.name());

      // Input tensors → node
      for (const auto &inputName : node.inputs()) {
        std::string tensorId = "tensor_" + escapeDot(inputName);
        gv << "    " << tensorId << " -> " << nodeId << ";\n";
      }

      // Node → output tensors
      for (const auto &outputName : node.outputs()) {
        std::string tensorId = "tensor_" + escapeDot(outputName);
        gv << "    " << nodeId << " -> " << tensorId << ";\n";
      }
    }
    gv << "\n";
  }

  static void dumpTensors(const Graph &g, std::ostream &gv) {
    for (const auto &[name, tensor] : g.tensors()) {
      std::string tensorId = "tensor_" + escapeDot(name);
      std::string kindStr;
      std::string bgcolor;

      switch (tensor.kind()) {
      case Tensor_kind::input: {
        kindStr = "input";
        bgcolor = "lightblue";
        break;
      }
      case Tensor_kind::output: {
        kindStr = "output";
        bgcolor = "lightgreen";
        break;
      }
      case Tensor_kind::constant: {
        kindStr = "constant";
        bgcolor = "lightgrey";
        break;
      }
      case Tensor_kind::intermediate: {
        kindStr = "intermediate";
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
      label += "<tr><td bgcolor=\"" + bgcolor + "\"><b>" + escapeHtml(name) +
               "</b></td></tr>";

      // String with data type
      label +=
          "<tr><td align=\"left\">type: " + tensorTypeToString(tensor.type()) +
          "</td></tr>";

      // String with dimension
      label += "<tr><td align=\"left\">shape: [";
      const auto &shape = tensor.shape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
          label += ",";
        label += std::to_string(shape[i]);
      }
      label += "]</td></tr>";

      // String with kind
      label += "<tr><td align=\"left\">" + escapeHtml(kindStr) + "</td></tr>";

      label += "</table>>";

      gv << "    " << tensorId << " [shape=plaintext, label=" << label
         << ", color=\"#252A34\", penwidth=2.5];\n";
    }
  }

  static void dumpNodes(const Graph &g, std::ostream &gv) {
    for (const auto &node : g.nodes()) {
      std::string nodeId = "node_" + escapeDot(node.name());

      std::string label = "<";
      label += "<table border=\"0\" cellborder=\"1\" cellspacing=\"0\" "
               "cellpadding=\"4\">";

      // String with opcode (bold)
      label += "<tr><td bgcolor=\"lightcoral\"><b>" +
               escapeHtml(node.opcode()) + "</b></td></tr>";

      // String with node name
      label += "<tr><td align=\"left\">name: " + escapeHtml(node.name()) +
               "</td></tr>";

      // String with id
      label += "<tr><td align=\"left\">id: " + std::to_string(node.id()) +
               "</td></tr>";

      // Add attributes if any.
      const auto &attrs = node.attributes();
      if (!attrs.empty())
        dumpAttributes(attrs, label);

      label += "</table>>";

      gv << "    " << nodeId << " [shape=plaintext, label=" << label
         << ", color=\"#252A34\", penwidth=2.5];\n";
    }
    gv << "\n";
  }

  static void dumpAttributes(const Attributes &attrs, std::string &label) {
    for (const auto &[attrName, attr] : attrs) {
      std::string attrStr = escapeHtml(attrName) + "=";
      std::visit(
          [&attrStr](const auto &val) {
            using T = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<T, float>) {
              attrStr += std::to_string(val);
            } else if constexpr (std::is_same_v<T, int64_t>) {
              attrStr += std::to_string(val);
            } else if constexpr (std::is_same_v<T, std::string>) {
              attrStr += "\"" + escapeHtml(val) + "\"";
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
              attrStr += "[";
              for (size_t i = 0; i < val.size(); ++i) {
                if (i > 0)
                  attrStr += ",";
                attrStr += std::to_string(val[i]);
              }
              attrStr += "]";
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
              attrStr += "[";
              for (size_t i = 0; i < val.size(); ++i) {
                if (i > 0)
                  attrStr += ",";
                attrStr += std::to_string(val[i]);
              }
              attrStr += "]";
            }
          },
          attr.value());
      label += "<tr><td align=\"left\">" + attrStr + "</td></tr>";
    }
  }
};

} // namespace tensor_compiler

#endif // INCLUDE_GRAPHVIZ_DUMPER_H
