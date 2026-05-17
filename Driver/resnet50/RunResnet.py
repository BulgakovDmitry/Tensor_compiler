#!usr/bin/python3

import ctypes
import numpy as np
import time
from PIL import Image
from pathlib import Path

LIB_PATH = Path(__file__).parent.parent.parent / "build" / "lib" / "libtensor_model.so"
LABELS_PATH = Path(__file__).parent.parent / "imagenet_classes.txt"

class AlignedArray:
    def __init__(self, shape, dtype=np.float32, alignment=64):
        import ctypes
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        total_bytes = int(np.prod(self.shape)) * self.dtype.itemsize

        try:
            self.libc = ctypes.CDLL("libc.so.6")
        except OSError:
            self.libc = ctypes.CDLL("libc.dylib")

        self.ptr = ctypes.c_void_p()
        result = self.libc.posix_memalign(ctypes.byref(self.ptr), alignment, total_bytes)
        if result != 0:
            raise MemoryError(f"posix_memalign failed: {result}")
        ctypes.memset(self.ptr, 0, total_bytes)

        ctype_map = {
            np.float32: ctypes.c_float, np.float64: ctypes.c_double,
            np.int32: ctypes.c_int32, np.int64: ctypes.c_int64,
            np.int8: ctypes.c_int8, np.uint8: ctypes.c_uint8,
        }
        if self.dtype.type not in ctype_map:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        ptr_typed = ctypes.cast(self.ptr, ctypes.POINTER(ctype_map[self.dtype.type]))
        self._array = np.ctypeslib.as_array(ptr_typed, shape=self.shape)
        self._alignment = alignment

    def __array__(self):
        return self._array

    @property
    def ctypes(self):
        return self._array.ctypes

    @property
    def data_ptr(self):
        return self.ptr

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr and self.ptr.value:
            try:
                self.libc.free(self.ptr)
            except:
                pass

    def __getattr__(self, name):
        return getattr(self._array, name)

class ResNetPredictor:
    def __init__(self, lib_path=LIB_PATH):
        if not lib_path.exists():
            raise RuntimeError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.tensorCompForward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.tensorCompForward.restype = ctypes.c_int

        self.labels = self._load_labels()
        self._setup_denormals()

    def _setup_denormals(self):
        try:
            import ctypes as c
            _mm_setcsr = c.cdll.libc._mm_setcsr
            _mm_getcsr = c.cdll.libc._mm_getcsr
            csr = _mm_getcsr()
            _mm_setcsr(csr | 0x8040)
        except:
            pass

    def _load_labels(self):
        if not LABELS_PATH.exists():
            return {i: f"class_{i}" for i in range(1000)}
        with open(LABELS_PATH) as f:
            return [line.strip() for line in f.readlines()]

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        scale = 256.0 / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        left = (img.width - 224) // 2
        top = (img.height - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.ascontiguousarray(arr.reshape(1, 3, 224, 224))
        return arr

    def predict(self, image_path, warmup=3, runs=10):
        x_np = self.preprocess(image_path)

        x_aligned = AlignedArray(x_np.shape, dtype=np.float32, alignment=64)
        np.copyto(x_aligned._array, x_np)

        out = AlignedArray((1, 1000), dtype=np.float32, alignment=64)

        x_ptr = x_aligned.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        assert x_aligned.ctypes.data % 64 == 0, f"Input not aligned: {x_aligned.ctypes.data % 64}"
        assert out.ctypes.data % 64 == 0, f"Output not aligned: {out.ctypes.data % 64}"

        for _ in range(warmup):
            self.lib.tensorCompForward(x_ptr, out_ptr)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            rc = self.lib.tensorCompForward(x_ptr, out_ptr)
            times.append(time.perf_counter() - start)
            if rc != 0:
                raise RuntimeError(f"Forward failed with code {rc}")

        out = out.reshape(1000)
        top5_idx = np.argsort(out)[-5:][::-1]
        results = [(self.labels[i], out[i]) for i in top5_idx]

        return {
            "latency_ms": np.mean(times) * 1000,
            "latency_std_ms": np.std(times) * 1000,
            "predictions": results
        }

if __name__ == "__main__":
    predictor = ResNetPredictor()
    res = predictor.predict("Driver/resnet50/cat.png")
    print(f"Latency: {res['latency_ms']:.2f} ± {res['latency_std_ms']:.2f} ms")
    for cls, prob in res["predictions"]:
        print(f"{cls:<30} {prob:.4f}")
