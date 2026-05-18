#!/usr/bin/env python3
import ctypes
import numpy as np
from PIL import Image
import onnxruntime as ort

FUNC_NAME = "tensorCompForward"

lib = ctypes.CDLL("./build/lib/libtensor_model.so")
func = getattr(lib, FUNC_NAME)

func.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
func.restype = ctypes.c_int

img = Image.open("Driver/mnist-12/digit_5.png").convert("L").resize((28, 28))
inp = np.array(img, dtype=np.float32) / 255.0
inp = inp[np.newaxis, np.newaxis, ...]
assert inp.flags["C_CONTIGUOUS"], "Input must be C-contiguous for NCHW layout"
out = np.zeros((1, 10), dtype=np.float32)

inp_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

ret = func(inp_ptr, out_ptr)

print(f"Return code: {ret}")
print(f"Raw output buffer: {out.flatten()[:10]}")  # первые 10 значений
print(f"Max score: {np.max(out):.6f}, Argmax: {np.argmax(out)}")

# Если все нули — модель не записала результат
if np.allclose(out, 0.0):
    print("❌ WARNING: Output buffer is all zeros! Check:")
    print("  1. Сигнатура функции в ctypes.argtypes")
    print("  2. Порядок аргументов: input, output (не наоборот!)")
    print("  3. Что модель действительно выполняется (нет early return)")

if ret == 0:
    print(f"Success. Prediction: class={np.argmax(out[0])}, max_score={np.max(out[0]):.4f}")
else:
    print(f"Execution failed with return code: {ret}")

ort_out = ort.InferenceSession("models/mnist-12.onnx").run(None, {"Input3": inp})[0]
print("Model:", np.argmax(out), " | Reference:", np.argmax(ort_out))
