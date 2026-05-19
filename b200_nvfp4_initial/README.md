# b200_nvfp4_initial

Baseline: `torch.compile(_scaled_mm, mode=...)` with default inductor backends. Inductor does not dispatch through cuBLAS for `_scaled_mm`, so the compiled path stays on its own Triton/CUTLASS codegen and lands far behind direct cuBLAS Lt.

```
modal run bench.py::run                                 # default compile mode
modal run bench.py::run --compile-mode max-autotune     # autotune sweep per shape
modal run bench.py::run --compile-mode reduce-overhead  # capture via cuda graph
```

Runtime: 10-15 min cold, 4-6 min warm. `max-autotune` adds a per-shape autotune sweep that dominates wall time.

Compiles `bench_cublaslt.cu` for sm_100, runs both, prints a side-by-side table and the cuBLAS Lt algo each path picks.
