// Benchmark NVFP4 GEMM via cuBLASLt with per-shape heuristic tuning.
//
// For each shape (M, N, K), enumerate top-N cuBLASLt algorithms returned by
// cublasLtMatmulAlgoGetHeuristic, time each over a warmup+measurement loop,
// and report the best one's time + TFLOPS.
//
// Output is one CSV line per shape:
//   M,N,K,best_us,best_tflops,algo_id,tile_id,stages,splitK
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_100 bench_nvfp4_cublaslt.cu -lcublasLt -lcublas -o bench_nvfp4_cublaslt
//
// CUDA 13+ required for CUDA_R_4F_E2M1 + nvfp4 scale modes.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

#define CK(x) do { cudaError_t _ck_e = (x); if (_ck_e != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_ck_e)); std::exit(1); } } while (0)
#define BK(x) do { cublasStatus_t _bk_st = (x); if (_bk_st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS err %s:%d %d\n", __FILE__, __LINE__, (int)_bk_st); std::exit(1); } } while (0)

struct Shape { int M, N, K; };

// Output per shape: time + algo metadata for the FIRST heuristic candidate
// ("default" what you'd get if you trust the heuristic ranking) and for
// the BEST-of-top-N after re-timing each candidate ("tuned").
struct Result {
    Shape shape;
    double default_us, default_tflops;
    int    default_algo, default_tile, default_stages, default_split_k;
    double tuned_us, tuned_tflops;
    int    tuned_algo,   tuned_tile,   tuned_stages,   tuned_split_k;
};

// FP4 vec16 scale layout: one FP8 e4m3 scale per 16 contiguous elements.
// For an M×K matrix laid out row-major, we have M × (K/16) scales.
// cuBLAS expects scales in a swizzled layout aligned to 128B blocks; we use
// the "outer scale layout" convention requested via the descriptor.

static cublasLtMatmulHeuristicResult_t time_algo(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t desc,
    cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc,
    const void *alpha, const void *beta,
    const void *A, const void *B, const void *C, void *D,
    void *workspace, size_t workspace_size,
    cublasLtMatmulHeuristicResult_t *cand, int cand_count,
    int warmup, int iters, double *out_best_us, int *out_best_idx)
{
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    double best_ms = 1e30;
    int best = -1;

    for (int i = 0; i < cand_count; ++i) {
        // Warmup
        for (int w = 0; w < warmup; ++w) {
            cublasStatus_t s = cublasLtMatmul(
                handle, desc, alpha, A, Adesc, B, Bdesc,
                beta, C, Cdesc, D, Ddesc, &cand[i].algo,
                workspace, workspace_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) { /* skip broken algo */ break; }
        }
        CK(cudaDeviceSynchronize());

        cudaEventRecord(start);
        for (int it = 0; it < iters; ++it) {
            cublasStatus_t s = cublasLtMatmul(
                handle, desc, alpha, A, Adesc, B, Bdesc,
                beta, C, Cdesc, D, Ddesc, &cand[i].algo,
                workspace, workspace_size, 0);
            if (s != CUBLAS_STATUS_SUCCESS) { break; }
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        double per_call_ms = ms / iters;
        if (per_call_ms > 0 && per_call_ms < best_ms) {
            best_ms = per_call_ms;
            best = i;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (best >= 0) {
        *out_best_us = best_ms * 1e3;
        *out_best_idx = best;
        return cand[best];
    }
    cublasLtMatmulHeuristicResult_t empty{};
    *out_best_us = -1;
    *out_best_idx = -1;
    return empty;
}


static int32_t get_algo_attr(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoConfigAttributes_t attr) {
    int32_t v = 0;
    size_t written = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &v, sizeof(v), &written);
    return v;
}


static Result bench_shape(cublasLtHandle_t handle, Shape s, void *workspace, size_t workspace_size,
                          int warmup, int iters, int top_n)
{
    Result r{s, -1,-1, -1,-1,-1,-1, -1,-1, -1,-1,-1,-1};

    // Allocate FP4 packed buffers: 2 nibbles per byte → K/2 bytes per row.
    // A is M×K (row-major) → M rows × K/2 bytes.
    // B is K×N (col-major view = N×K row-major) → N×K/2 bytes.
    // D is M×N BF16.
    size_t bytes_A = (size_t)s.M * (s.K / 2);
    size_t bytes_B = (size_t)s.N * (s.K / 2);  // we store B as N×K for col-major
    size_t bytes_D = (size_t)s.M * s.N * sizeof(__nv_bfloat16);

    // Per-vec-16 FP8 scales: M × (K/16) for A, N × (K/16) for B.
    size_t scale_a_n = (size_t)s.M * (s.K / 16);
    size_t scale_b_n = (size_t)s.N * (s.K / 16);

    void *dA, *dB, *dD, *dScaleA, *dScaleB;
    CK(cudaMalloc(&dA, bytes_A));
    CK(cudaMalloc(&dB, bytes_B));
    CK(cudaMalloc(&dD, bytes_D));
    CK(cudaMalloc(&dScaleA, scale_a_n));
    CK(cudaMalloc(&dScaleB, scale_b_n));

    // Init with non-zero patterns so the heuristic isn't dominated by sparsity.
    CK(cudaMemset(dA, 0x55, bytes_A));
    CK(cudaMemset(dB, 0x77, bytes_B));
    // FP8 e4m3 scale = 1.0 has bit pattern 0x38. Use that as a flat init.
    CK(cudaMemset(dScaleA, 0x38, scale_a_n));
    CK(cudaMemset(dScaleB, 0x38, scale_b_n));
    CK(cudaMemset(dD, 0, bytes_D));

    // Descriptor: NVFP4 inputs, FP32 compute, BF16 output.
    // CUDA_R_4F_E2M1 is the FP4 e2m1 dtype id (CUDA 13).
    cublasLtMatmulDesc_t desc;
    BK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    // We pass A as transposed (FP4 GEMM convention: outer dim K accumulates).
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Per-vector NVFP4 scales: 16 elements per FP8 e4m3 scale.
    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
    BK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));

    // Layouts. A is FP4, B is FP4, C/D is BF16.
    cublasLtMatrixLayout_t Adesc, Bdesc, Ddesc;
    BK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, s.K, s.M, s.K));
    BK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, s.K, s.N, s.K));
    BK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, s.M, s.N, s.M));

    // Preferences.
    cublasLtMatmulPreference_t pref;
    BK(cublasLtMatmulPreferenceCreate(&pref));
    BK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    std::vector<cublasLtMatmulHeuristicResult_t> heur(top_n);
    int n_returned = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
        handle, desc, Adesc, Bdesc, Ddesc, Ddesc,
        pref, top_n, heur.data(), &n_returned);
    if (hs != CUBLAS_STATUS_SUCCESS || n_returned == 0) {
        fprintf(stderr, "[shape %dx%dx%d] no heuristic algos (status=%d returned=%d)\n",
                s.M, s.N, s.K, (int)hs, n_returned);
        goto cleanup;
    }
    heur.resize(n_returned);

    {
        float alpha = 1.0f, beta = 0.0f;

        // (1) Default: only the FIRST heuristic candidate.
        double default_us = -1;
        int    default_idx = -1;
        cublasLtMatmulHeuristicResult_t default_algo = time_algo(
            handle, desc, Adesc, Bdesc, Ddesc, Ddesc,
            &alpha, &beta, dA, dB, dD, dD,
            workspace, workspace_size,
            heur.data(), /*cand_count=*/1, warmup, iters,
            &default_us, &default_idx);

        // (2) Tuned: time all top-N, pick fastest.
        double tuned_us = -1;
        int    tuned_idx = -1;
        cublasLtMatmulHeuristicResult_t tuned_algo = time_algo(
            handle, desc, Adesc, Bdesc, Ddesc, Ddesc,
            &alpha, &beta, dA, dB, dD, dD,
            workspace, workspace_size,
            heur.data(), n_returned, warmup, iters,
            &tuned_us, &tuned_idx);

        if (default_idx < 0 && tuned_idx < 0) {
            fprintf(stderr, "[shape %dx%dx%d] all algos failed at runtime\n",
                    s.M, s.N, s.K);
            goto cleanup;
        }

        double flops = 2.0 * s.M * s.N * s.K;

        if (default_idx >= 0) {
            r.default_us = default_us;
            r.default_tflops = flops / (default_us * 1e-6) / 1e12;
            r.default_algo   = get_algo_attr(default_algo.algo, CUBLASLT_ALGO_CONFIG_ID);
            r.default_tile   = get_algo_attr(default_algo.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            r.default_stages = get_algo_attr(default_algo.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            r.default_split_k= get_algo_attr(default_algo.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        }
        if (tuned_idx >= 0) {
            r.tuned_us = tuned_us;
            r.tuned_tflops = flops / (tuned_us * 1e-6) / 1e12;
            r.tuned_algo    = get_algo_attr(tuned_algo.algo, CUBLASLT_ALGO_CONFIG_ID);
            r.tuned_tile    = get_algo_attr(tuned_algo.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
            r.tuned_stages  = get_algo_attr(tuned_algo.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
            r.tuned_split_k = get_algo_attr(tuned_algo.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
        }
    }

cleanup:
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD);
    cudaFree(dScaleA);
    cudaFree(dScaleB);

    return r;
}


int main() {
    Shape shapes[] = {
        {128, 128, 256},
        {256, 256, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 6144, 6144},
        {8192, 8192, 8192},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    // Env info on stderr so the CSV on stdout stays parseable.
    int rt_ver = 0, drv_ver = 0;
    cudaRuntimeGetVersion(&rt_ver);
    cudaDriverGetVersion(&drv_ver);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    size_t cublaslt_ver = cublasLtGetVersion();
    fprintf(stderr, "[cublaslt] version=%zu cuda_runtime=%d cuda_driver=%d "
                    "device=\"%s\" sm=%d.%d\n",
            cublaslt_ver, rt_ver, drv_ver, prop.name, prop.major, prop.minor);

    cublasLtHandle_t handle;
    BK(cublasLtCreate(&handle));

    // 256 MB workspace covers the largest tile choices.
    size_t workspace_size = 256ull * 1024 * 1024;
    void *workspace = nullptr;
    CK(cudaMalloc(&workspace, workspace_size));

    printf("M,N,K,"
           "default_us,default_tflops,default_algo,default_tile,default_stages,default_splitk,"
           "tuned_us,tuned_tflops,tuned_algo,tuned_tile,tuned_stages,tuned_splitk\n");
    for (int i = 0; i < n_shapes; ++i) {
        Result r = bench_shape(handle, shapes[i], workspace, workspace_size,
                               /*warmup=*/20, /*iters=*/100, /*top_n=*/10);
        printf("%d,%d,%d,"
               "%.3f,%.2f,%d,%d,%d,%d,"
               "%.3f,%.2f,%d,%d,%d,%d\n",
               r.shape.M, r.shape.N, r.shape.K,
               r.default_us, r.default_tflops,
               r.default_algo, r.default_tile, r.default_stages, r.default_split_k,
               r.tuned_us, r.tuned_tflops,
               r.tuned_algo, r.tuned_tile, r.tuned_stages, r.tuned_split_k);
        fflush(stdout);
    }

    cudaFree(workspace);
    cublasLtDestroy(handle);
    return 0;
}
