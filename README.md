<div align="center">

# ⚔️🐉 Tensor compiler in C++ 🐉⚔️
  ![C++](https://img.shields.io/badge/C++-23-blue?style=for-the-badge&logo=cplusplus)
  ![CMake](https://img.shields.io/badge/CMake-3.20+-green?style=for-the-badge&logo=cmake)
  ![Testing](https://img.shields.io/badge/Google_Test-Framework-red?style=for-the-badge&logo=google)
  ![ONNX](https://img.shields.io/badge/ONNX-Supported-005CED?style=for-the-badge&logo=onnx)
  
</div>

## Table of Contents 📖
- [Running the program](#running-the-program)
- [Using dump](#using-dump)
- [Introduction](#introduction)
- [Computational Graph Architecture](#computational-graph-architecture)
- [Project structure](#project-structure)
- [Project authors](#project-authors)

## Documentation 📗
[Documentation](https://bulgakovdmitry.github.io/Tensor_compiler/)

## <a id="running-the-program"></a>Running the program 🛡️
Repository `cloning`, `build` and `compilation` is performed using the following commands:

```
git clone git@github.com:BulgakovDmitry/Tensor_compiler.git
cd Tensor_compiler
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Program `execution` is performed in the following format:
```
./build/tensor_compiler <model.onnx>
```

## <a id="introduction"></a> Introduction 🎍
In the era of deep learning and artificial intelligence, neural networks have become increasingly complex and computationally intensive. While high-level frameworks like PyTorch and TensorFlow provide convenient APIs for designing and training models, they often introduce performance overhead when executing these models in production environments.

Tensor compilers address this critical challenge by transforming high-level neural network descriptions into highly optimized execution code. Unlike traditional compilers that work with scalar values, tensor compilers operate on multidimensional arrays (tensors) and apply domain-specific optimizations that dramatically improve performance and efficiency.

## <a id="computational-graph-architecture"></a> Computational Graph Architecture 📐

### Core Component Design

Our tensor compiler implements a robust frontend architecture that processes ONNX models through a well-defined computational graph representation. The system follows strict separation of concerns with four primary components forming the Intermediate Representation (IR) layer.

#### Tensor Representation
The Tensor class serves as the fundamental data container, representing multidimensional arrays with comprehensive metadata management:

<details>
<summary>Implementation of Tensor</summary>

```C++
// tensor.hpp
enum class Tensor_kind {
    unknown = 0,
    input,
    output,
    intermediate,
    constant,
};

class Tensor {
private:
    std::string name_;
    int type_ = data_type::TensorProto_DataType_UNDEFINED;
    Tensor_kind kind_ = Tensor_kind::unknown;
    std::string data_;
    std::vector<int64_t> shape_;
    // Additional implementation details...
};
```

</details>

Key features include:
- Strict typing through Tensor_kind enumeration for precise role identification
- Comprehensive shape management with dimension tracking
- Efficient raw data storage using protocol buffer-compatible serialization
- Type-safe accessors ensuring data integrity throughout the compilation pipeline

#### Attribute System
The attribute system employs modern C++ type-safe techniques to handle diverse parameter types:

<details>
<summary>Implementation of Attribute</summary>

```C++
// attribute.hpp
class Attribute {
public:
    using AttrValue = std::variant<float, int64_t, std::string,
                  std::vector<float>, std::vector<int64_t>>;
private:
    std::string name_;
    AttrValue value;
public:
    Attribute(const std::string &name, const AttrValue &value)
        : name_{name}, value{value} {}
    // Type-safe access methods...
};
```
</details>

This implementation provides:
- Strong type safety through std::variant for heterogeneous attribute values
- O(1) complexity attribute lookup via hash-based storage
- Memory-efficient representation avoiding unnecessary boxing
- Compile-time verification of attribute types during optimization passes

#### Node Structure
The computational operations are represented through a sophisticated node structure with comprehensive ONNX compatibility:

<details>
<summary>Implementation of Node</summary>

```C++
// node.hpp
class Node {
public:
    using node_id = std::size_t;
    using name_t = google::protobuf::RepeatedPtrField<std::string>;
private:
    node_id id_{0};
    std::string opcode_;
    std::string name_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    Attributes attributes_; // std::unordered_map<std::string, Attribute>
    
public:
    void parse_attributes(const onnx::NodeProto &node);
    // Additional implementation details...
    
    // Attribute parsing implementation
    inline void Node::parse_attributes(const onnx::NodeProto &node) {
        for (const auto &attr : node.attribute()) {
            const std::string &name = attr.name();
            switch (attr.type()) {
                case onnx::AttributeProto_AttributeType_FLOAT:
                    set_attribute(name, attr.f());
                    break;
                case onnx::AttributeProto_AttributeType_INT:
                    set_attribute(name, static_cast<int64_t>(attr.i()));
                    break;
                // Additional type handling...
            }
        }
    }
};
```

</details>

The node implementation ensures:
- Complete ONNX attribute type coverage through protocol buffer integration
- Efficient string-based input/output handling with bidirectional compatibility
- Deterministic attribute processing through type-safe conversion
- Extensible architecture for future operation support

#### Graph Container
The top-level graph structure maintains the complete computational topology:

<details>
<summary>Implementation of Graph</summary>
  
```C++
// graph.hpp
class Graph {
private:
    std::string name_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<Node> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    
public:
    const Tensor *get_tensor(const std::string &name) const;
    void add_tensor(Tensor tensor);
    void add_node(Node node);
    // Additional implementation details...
    
    inline const Tensor *Graph::get_tensor(const std::string &name) const {
        auto it = tensors_.find(name);
        if (it != tensors_.end())
            return &(it->second);
        return nullptr;
    }
};
```
</details>

This design provides:
- Efficient tensor lookup via hash map (O(1) complexity)
- Preservation of topological ordering for execution planning
- Clear boundary definition through explicit input/output tracking
- Memory-safe tensor management through value semantics

### Processing Pipeline
The compiler's frontend processing follows a structured transformation pipeline:

#### Model Loading Phase

```C++
// driver.hpp
int driver(const std::string &model_onnx);
```

The driver function initiates the compilation process by loading the ONNX model file and verifying its integrity.

#### Graph Construction Phase

```C++
// driver.hpp
Graph build_compute_graph(const onnx::GraphProto &graph);
```

This critical function implements the transformation from ONNX representation to our internal computational graph:

1. Tensor Initialization
- Input tensors are registered with Tensor_kind::input
- Output tensors are registered with Tensor_kind::output
- Initializers are converted to Tensor_kind::constant with proper data handling
- Intermediate tensors are created for internal computation flow
2. Node Transformation
- ONNX nodes are converted to our Node representation
- Operation codes are preserved for backend processing
- Input/output connections are established through tensor references
- Attributes are parsed into type-safe structures
3. Graph Assembly
- The complete computational topology is constructed
- Input/output boundaries are explicitly defined
- Reference integrity is verified through tensor lookups
- Topological ordering is established for execution planning

This architecture provides a solid foundation for subsequent optimization passes while maintaining strict compatibility with the ONNX specification. The separation between the frontend representation and future backend implementations ensures extensibility and maintainability throughout the compiler's development lifecycle.

## <a id="using-dump"></a>Using dump 🏰
To enable the graph dump option for the `compute graph`, you need to set the `-GRAPH_DUMP` flag, which is disabled by default:
```bash
cmake -S . -B build -DGRAPH_DUMP=ON
```
The constructed `graph` can be viewed in graphical representation using `graphviz`. To generate an image, you can enter:
```bash
dot dump/dump.gv -Tsvg -o dump/dump.svg
```
This produces the following `graph` representation:

<details>
<summary>example of generated graph</summary>
  
<div align="center">
  <img src="img/dump_mnist_12_onnx.svg" alt="Dump Banner" width="1200">
</div>

</details>

## <a id="project-structure"></a>Project structure 🏯

<details>
<summary>Project structure</summary>

```
.
├── CMakeLists.txt
├── include
│   ├── driver.hpp
│   ├── dump_path_gen.hpp
│   ├── graphviz_dumper.hpp
│   ├── handlers.hpp
│   └── structure
│       ├── attribute.hpp
│       ├── graph.hpp
│       ├── node.hpp
│       └── tensor.hpp
├── onnx
│   └── onnx.proto
└── src
    ├── driver.cpp
    └── main.cpp

```

</details>

## <a id="project-authors"></a>Project authors 🧙‍♂️

<div align="center">

  <a href="https://github.com/RTCupid">
    <img src="https://raw.githubusercontent.com/BulgakovDmitry/3D_triangles/main/img/A.jpeg" width="160" height="160" style="border-radius: 50%;">
  </a>
  <a href="https://github.com/BulgakovDmitry">
    <img src="https://raw.githubusercontent.com/BulgakovDmitry/3D_triangles/main/img/D.jpeg" width="160" height="160" style="border-radius: 50%;">
  </a>
  <br>
  <a href="https://github.com/RTCupid"><strong>@RTCupid, </strong></a>
  <a href="https://github.com/BulgakovDmitry"><strong>@BulgakovDmitry, </strong></a>
  <br>
</div>
