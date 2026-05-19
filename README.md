# NVFP4 GEMM benchmarks on Modal

cuBLAS Lt vs `torch._scaled_mm` for NVFP4 on B200.

## Folders

`b200_nvfp4_initial/` is the naive bench, default inductor backends.

`b200_nvfp4_autotune/` sets `_inductor.config.max_autotune_gemm_backends = "ATEN,CUTLASS"` so the autotuner dispatches `_scaled_mm` through cuBLAS Lt. Triton is skipped.

## Setup

```
pip install modal
modal setup
```

## Run

```
cd b200_nvfp4_initial      # or b200_nvfp4_autotune
modal run bench.py::run                              # default: --compile-mode default
modal run bench.py::run --compile-mode max-autotune
modal run bench.py::run --compile-mode reduce-overhead  # capture via cuda graph
```

## Timing

First run: 10-15 min (image build + cold B200 container + 7-shape sweep). Subsequent runs reuse the cached image and finish in 4-6 min.

`--compile-mode max-autotune` dominates wall time (autotune sweep per shape). The default `--compile-mode default` runs much faster.

## Output

Per shape: `cuBLAS default us/TFLOPS`, `cuBLAS tuned us/TFLOPS`, `torch eager us/TFLOPS`, `torch compile us/TFLOPS`, and a side-by-side of the cuBLAS Lt algorithm IDs both paths picked.
