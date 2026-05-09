import ctypes
import numpy as np
import time
from PIL import Image
from pathlib import Path

LIB_PATH = Path(__file__).parent.parent / "build" / "lib" / "libtensor_model.so"
LABELS_PATH = Path(__file__).parent / "imagenet_classes.txt"

class ResNetPredictor:
    def __init__(self, lib_path=LIB_PATH):
        if not lib_path.exists():
            raise RuntimeError(f"Library not found: {lib_path}")

        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.tensor_comp_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.tensor_comp_forward.restype = ctypes.c_int

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
        x = self.preprocess(image_path)
        out = np.zeros((1, 1000), dtype=np.float32)

        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        for _ in range(warmup):
            self.lib.tensor_comp_forward(x_ptr, out_ptr)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            rc = self.lib.tensor_comp_forward(x_ptr, out_ptr)
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
    res = predictor.predict("input.jpg")
    print(f"Latency: {res['latency_ms']:.2f} ± {res['latency_std_ms']:.2f} ms")
    for cls, prob in res["predictions"]:
        print(f"{cls:<30} {prob:.4f}")
