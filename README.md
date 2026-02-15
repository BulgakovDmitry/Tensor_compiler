<div align="center">

# Tensor compiler in C++
  ![C++](https://img.shields.io/badge/C++-23-blue?style=for-the-badge&logo=cplusplus)
  ![CMake](https://img.shields.io/badge/CMake-3.20+-green?style=for-the-badge&logo=cmake)
  ![Testing](https://img.shields.io/badge/Google_Test-Framework-red?style=for-the-badge&logo=google)
  ![ONNX](https://img.shields.io/badge/ONNX-Supported-005CED?style=for-the-badge&logo=onnx)
  
</div>


## Running the program
Repository cloning, build and compilation is performed using the following commands:

```
git clone git@github.com:BulgakovDmitry/Tensor_compiler.git
cd Tensor_compiler
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Program execution is performed in the following format:
```
./build/tensor_compiler <model.onnx>
```
