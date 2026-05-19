"""torch._scaled_mm NVFP4 bench: eager + torch.compile, with inductor's
GEMM-autotune backend list expanded to include ATEN so the autotuner can
pick the cuBLAS Lt path. Corrected version of ../b200_nvfp4_initial/.

Env vars:
  COMPILE_MODE   torch.compile mode (default: "default").
                 Other useful values: "max-autotune", "reduce-overhead".

Emits CSV: M,N,K,eager_us,eager_tflops,compile_us,compile_tflops
"""
import os
import torch
import torch._inductor.config as _icfg

COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")

# Inductor autotune backends: ATEN routes _scaled_mm through cuBLAS Lt;
# CUTLASS stays as a fallback. Triton is skipped.
_icfg.max_autotune = True
_icfg.max_autotune_gemm = True
_icfg.max_autotune_gemm_backends = "ATEN,CUTLASS"
_icfg.coordinate_descent_tuning = True

SHAPES = [
    (128, 128, 256),
    (256, 256, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 6144, 6144),
    (8192, 8192, 8192),
]


def _alloc_fp4(rows: int, cols: int) -> torch.Tensor:
    raw = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device="cuda")
    return raw.view(torch.float4_e2m1fn_x2)


def _alloc_scale(rows: int, cols: int) -> torch.Tensor:
    raw = torch.full((rows, cols // 16), 0x38, dtype=torch.uint8, device="cuda")
    return raw.view(torch.float8_e4m3fn)


def _scaled_mm(a, b, sa, sb):
    return torch._scaled_mm(a, b, scale_a=sa, scale_b=sb,
                            bias=None, out_dtype=torch.bfloat16)


def time_callable(fn, warmup=20, iters=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_shape(M, N, K):
    a, b = _alloc_fp4(M, K), _alloc_fp4(N, K).T
    sa, sb = _alloc_scale(M, K), _alloc_scale(N, K)

    eager_ms = time_callable(lambda: _scaled_mm(a, b, sa, sb))

    compiled = torch.compile(_scaled_mm, mode=COMPILE_MODE, fullgraph=True)
    compile_ms = time_callable(lambda: compiled(a, b, sa, sb))

    flops = 2.0 * M * N * K
    return {
        "eager_us": eager_ms * 1e3,
        "eager_tflops": flops / (eager_ms * 1e-3) / 1e12,
        "compile_us": compile_ms * 1e3,
        "compile_tflops": flops / (compile_ms * 1e-3) / 1e12,
    }


def main():
    print(f"# torch {torch.__version__}  cuda={torch.version.cuda}  "
          f"device={torch.cuda.get_device_name(0)}  "
          f"sm={torch.cuda.get_device_capability(0)}  "
          f"compile_mode={COMPILE_MODE}", flush=True)
    print("M,N,K,eager_us,eager_tflops,compile_us,compile_tflops")
    for M, N, K in SHAPES:
        try:
            r = bench_shape(M, N, K)
            print(f"{M},{N},{K},"
                  f"{r['eager_us']:.3f},{r['eager_tflops']:.2f},"
                  f"{r['compile_us']:.3f},{r['compile_tflops']:.2f}",
                  flush=True)
        except Exception as exc:
            print(f"{M},{N},{K},FAIL,FAIL,FAIL,FAIL  # {type(exc).__name__}: {exc}",
                  flush=True)


if __name__ == "__main__":
    main()
