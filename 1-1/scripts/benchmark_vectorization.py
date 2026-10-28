"""Timing reference only: the production image pipeline contains no image loops."""
import platform
from time import perf_counter

import numpy as np


def benchmark_images(images, count=1024, repeats=5):
    """Compare the same float32 mean/std work; exclude IO and one-time casting."""
    sample = np.ascontiguousarray(images[:count], dtype=np.float32)

    def vectorized():
        return np.column_stack((sample.mean(axis=(1, 2, 3)), sample.std(axis=(1, 2, 3))))

    def loop_reference():
        # A deliberately slow, benchmark-only baseline; never used for features.
        return np.asarray([(image.mean(), image.std()) for image in sample])

    expected, actual = loop_reference(), vectorized()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    timings = {"vectorized_ms": [], "loop_ms": []}
    for repeat in range(repeats):
        # Alternate the order to reduce systematic warm-cache/order bias.
        cases = [("vectorized_ms", vectorized), ("loop_ms", loop_reference)]
        if repeat % 2:
            cases.reverse()
        for name, function in cases:
            start = perf_counter()
            function()
            timings[name].append((perf_counter() - start) * 1000)
    vector_ms, loop_ms = (float(np.median(timings[name])) for name in ["vectorized_ms", "loop_ms"])
    return {
        "images": len(sample), "shape": list(sample.shape), "dtype": str(sample.dtype),
        "c_contiguous": bool(sample.flags.c_contiguous), "repeats": repeats,
        "vectorized_median_ms": vector_ms, "loop_median_ms": loop_ms,
        "speedup": loop_ms / vector_ms, "max_absolute_difference": float(np.abs(actual - expected).max()),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "machine": platform.machine(), "platform": platform.system(),
        "timing_scope": "mean/std only; excludes disk IO, casting, edge features and DataFrame joins",
        "trials_ms": timings,
    }
