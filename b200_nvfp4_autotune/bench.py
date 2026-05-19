"""Modal driver: NVFP4 GEMM on B200, cuBLAS Lt vs torch._scaled_mm.

Sets `_inductor.config.max_autotune_gemm_backends = "ATEN,CUTLASS"`
in bench_torch.py so the torch.compile autotuner can dispatch through
cuBLAS Lt. See ../b200_nvfp4_initial/ for the version without that flag.

Usage:
    modal run bench.py::run
"""

import csv
import io
import os
import re
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04",
                              add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install("torch", "tabulate", "numpy")
    .add_local_file(str(HERE / "bench_cublaslt.cu"), "/root/bench_cublaslt.cu")
    .add_local_file(str(HERE / "bench_torch.py"),    "/root/bench_torch.py")
)

app = modal.App("nvfp4-gemm-b200-autotune", image=IMAGE)


@app.function(gpu="B200", timeout=60 * 30)
def bench_b200(compile_mode: str = "default") -> None:
    nvcc = "/usr/local/cuda/bin/nvcc"
    cu_src = "/root/bench_cublaslt.cu"
    cu_bin = "/root/bench_cublaslt"

    _print_env(nvcc)

    # 1. Compile the cuBLAS Lt harness for Blackwell (sm_100).
    print(f"$ building {cu_src}", flush=True)
    subprocess.check_call([
        nvcc, "-O3", "-std=c++17", "-arch=sm_100",
        "-lcublasLt", "-lcublas", cu_src, "-o", cu_bin,
    ])

    # 2. Run cuBLAS Lt harness captures CSV to stdout.
    print(f"$ running {cu_bin}", flush=True)
    cublas_out = subprocess.check_output([cu_bin], text=True)
    print("--- cuBLAS Lt CSV ---")
    print(cublas_out)

    # 3. Run torch._scaled_mm harness with cuBLAS Lt logging enabled so we
    # can see which algorithm (if any) torch dispatches to.
    print(f"$ running torch bench", flush=True)
    torch_env = os.environ.copy()
    torch_env["CUBLASLT_LOG_LEVEL"] = "5"
    torch_env["CUBLASLT_LOG_FILE"] = "/root/torch_cublaslt.log"
    torch_env["COMPILE_MODE"] = compile_mode
    torch_out = subprocess.check_output(
        ["/usr/local/bin/python", "-u", "/root/bench_torch.py"],
        text=True, env=torch_env,
    )
    print("--- torch CSV ---")
    print(torch_out)

    # 4. Parse logs into combined table.
    cu_rows = _parse_csv(cublas_out)
    th_rows = _parse_csv(torch_out)
    torch_algos = _parse_cublaslt_log("/root/torch_cublaslt.log")

    _print_perf_table(cu_rows, th_rows)
    _print_algo_table(cu_rows, torch_algos)


def _parse_csv(s: str) -> dict:
    # Strip `# comment` and blank lines so DictReader sees the M,N,K,... header
    # on the first row (the torch bench prints a `# torch ...` env line first).
    clean = "\n".join(l for l in s.splitlines() if l and not l.startswith("#"))
    rows = {}
    for r in csv.DictReader(io.StringIO(clean)):
        try:
            k = (int(r["M"]), int(r["N"]), int(r["K"]))
        except (KeyError, ValueError):
            continue
        rows[k] = r
    return rows


def _parse_cublaslt_log(path: str) -> dict:
    """Extract (M,N,K) → (algoId, tile, stages, splitK) from a Trace log."""
    if not Path(path).exists():
        return {}
    log = Path(path).read_text()
    pat = re.compile(
        r"Adesc=\[type=\S+ rows=(\d+) cols=(\d+)[^\]]*\]"
        r".*?Bdesc=\[type=\S+ rows=(\d+) cols=(\d+)[^\]]*\]"
        r".*?Cdesc=\[type=\S+ rows=(\d+) cols=(\d+)[^\]]*\]"
        r".*?algo=\[algoId=(\d+) tile=\S*?_(\S+) stages=\S*?_(\S+) "
        r"reductionScheme=\S+ numSplitsK=(\d+)\]"
    )
    out = {}
    for m in pat.finditer(log):
        # cuBLAS Lt is column-major: A=K×M, B=K×N, C=N×M.
        K = int(m.group(2)); M = int(m.group(6)); N = int(m.group(5))
        out.setdefault((M, N, K),
                       (m.group(7), m.group(8), m.group(9), int(m.group(10))))
    return out


def _print_env(nvcc: str) -> None:
    """Dump host + GPU env (torch, CUDA, driver, device) before the runs."""
    print("=== env ===")
    subprocess.run(["/usr/local/bin/python", "-c",
                    "import sys, torch; "
                    "print('python =', sys.version.split()[0]); "
                    "print('torch  =', torch.__version__); "
                    "print('cuda   =', torch.version.cuda); "
                    "print('device =', torch.cuda.get_device_name(0)); "
                    "print('sm     =', torch.cuda.get_device_capability(0))"])
    subprocess.run([nvcc, "--version"])
    subprocess.run(["nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv"])
    print()


def _f(d, k):
    try: return float(d.get(k, "FAIL"))
    except (TypeError, ValueError): return float("nan")


def _print_perf_table(cu, th):
    from tabulate import tabulate
    rows = []
    for shape in sorted(set(cu.keys()) | set(th.keys())):
        M, N, K = shape
        c, t = cu.get(shape, {}), th.get(shape, {})
        cd_us, cd_tf = _f(c, "default_us"), _f(c, "default_tflops")
        ct_us, ct_tf = _f(c, "tuned_us"),   _f(c, "tuned_tflops")
        te_us, te_tf = _f(t, "eager_us"),   _f(t, "eager_tflops")
        tc_us, tc_tf = _f(t, "compile_us"), _f(t, "compile_tflops")
        sp_eager = te_us / ct_us if ct_us > 0 else float("nan")
        sp_comp  = tc_us / ct_us if ct_us > 0 else float("nan")
        rows.append([
            f"{M}x{N}x{K}",
            f"{cd_us:.2f}", f"{ct_us:.2f}", f"{te_us:.2f}", f"{tc_us:.2f}",
            f"{cd_tf:.1f}", f"{ct_tf:.1f}", f"{te_tf:.1f}", f"{tc_tf:.1f}",
            f"{sp_eager:.2f}", f"{sp_comp:.2f}",
        ])
    headers = [
        "shape",
        "cu-def us", "cu-tun us", "th-eager us", "th-comp us",
        "cu-def TF", "cu-tun TF", "th-eag TF",  "th-comp TF",
        "eag/tun", "comp/tun",
    ]
    print("\n=== B200 NVFP4 GEMM, cuBLAS (default/tuned) vs torch (eager/compile) ===")
    print(tabulate(rows, headers=headers, tablefmt="github", stralign="right"))


def _print_algo_table(cu, torch_algos):
    from tabulate import tabulate
    rows = []
    for shape in sorted(set(cu.keys()) | set(torch_algos.keys())):
        M, N, K = shape
        c = cu.get(shape, {})
        t = torch_algos.get(shape)
        ct_algo  = c.get("tuned_algo",   "?")
        ct_tile  = c.get("tuned_tile",   "?")
        ct_stg   = c.get("tuned_stages", "?")
        ct_splk  = c.get("tuned_splitk", "?")
        if t is None:
            th_algo = th_tile = th_stg = th_splk = "-"
        else:
            th_algo, th_tile, th_stg, th_splk = t
        same = "YES" if str(ct_algo) == str(th_algo) else "no"
        rows.append([
            f"{M}x{N}x{K}",
            ct_algo, ct_tile, ct_stg, ct_splk,
            th_algo, th_tile, th_stg, th_splk,
            same,
        ])
    headers = [
        "shape",
        "cu algo id", "cu tile id", "cu stages id", "cu splitK",
        "th algo id", "th tile id", "th stages id", "th splitK",
        "same algo id?",
    ]
    print("\n=== Algorithm picks (cuBLAS tuned vs torch._scaled_mm) ===")
    print(tabulate(rows, headers=headers, tablefmt="github", stralign="right"))


@app.local_entrypoint()
def run(compile_mode: str = "default"):
    """Args: --compile-mode {max-autotune,default,reduce-overhead}."""
    bench_b200.remote(compile_mode=compile_mode)
