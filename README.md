<div align="center">

# ⚔️🐉 Tensor compiler in C++ 🐉⚔️
  ![C++](https://img.shields.io/badge/C++-23-blue?style=for-the-badge&logo=cplusplus)
  ![CMake](https://img.shields.io/badge/CMake-3.20+-green?style=for-the-badge&logo=cmake)
  ![Testing](https://img.shields.io/badge/Google_Test-Framework-red?style=for-the-badge&logo=google)
  ![ONNX](https://img.shields.io/badge/ONNX-Supported-005CED?style=for-the-badge&logo=onnx)
  ![MLIR](https://img.shields.io/badge/MLIR-18.1+-yellow?style=for-the-badge&logo=llvm&logoColor=white)
  ![LLVM](https://img.shields.io/badge/LLVM-18.1+-blue?style=for-the-badge&logo=llvm)
  
</div>

## Table of Contents 📖
- [Running the program](#running-the-program)
- [Using dump](#using-dump)
- [Introduction](#introduction)
- [Methodology](#methodology)
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

## <a id="methodology"></a> Methodology


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
