# b200_nvfp4_autotune

Same as `../b200_nvfp4_initial/` but sets

```python
torch._inductor.config.max_autotune_gemm_backends = "ATEN,CUTLASS"
```

so the torch.compile autotuner has ATEN in the candidate list and can dispatch `_scaled_mm` through cuBLAS Lt. With this flag the compiled path picks the same cuBLAS algo as our C++ harness (verified via `CUBLASLT_LOG_LEVEL=5`). Triton is skipped.

```
modal run bench.py::run                                 # default compile mode
modal run bench.py::run --compile-mode max-autotune     # autotune sweep per shape
modal run bench.py::run --compile-mode reduce-overhead  # capture via cuda graph
```

Runtime: 10-15 min cold, 4-6 min warm. `max-autotune` adds a per-shape autotune sweep that dominates wall time.
