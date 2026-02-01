#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernel.cuh"

struct Options {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    std::string impl = "baseline";
    bool verify = true;
};

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--m") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            opt.m = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--n") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            opt.n = std::stoi(argv[++i]);
        } else if ((strcmp(argv[i], "--k") == 0 || strcmp(argv[i], "-k") == 0) && i + 1 < argc) {
            opt.k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
            opt.impl = argv[++i];
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            opt.verify = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./dgemm [--m int] [--n int] [--k int] [--impl baseline|naive|tiled|cublas] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
        }
    }
    return opt;
}

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + " : " + cudaGetErrorString(err));
    }
}

void check_cublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + " : cuBLAS error");
    }
}

double gflops(int m, int n, int k, double millis) {
    double flops = 2.0 * m * n * k;
    return flops / (millis * 1e6);
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    const int m = opt.m, n = opt.n, k = opt.k;
    const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
    const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);

    std::vector<float> h_a(m * k), h_b(k * n), h_c(m * n, 0.0f), h_ref(m * n, 0.0f);

    for (int i = 0; i < m * k; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * n; ++i) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes_a), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes_b), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_c, bytes_c), "cudaMalloc d_c");
    check_cuda(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice), "cudaMemcpy h_a to d_a");
    check_cuda(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice), "cudaMemcpy h_b to d_b");
    check_cuda(cudaMemset(d_c, 0, bytes_c), "cudaMemset d_c to 0");

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create stream");
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "record start event");
    check_cuda(cudaEventCreate(&stop), "create stop");

    float elapsed_ms = 0.0f;
    if (opt.impl == "baseline" || opt.impl == "naive" || opt.impl == "tiled") {
        check_cuda(cudaEventRecord(start, stream), "record start event");
        if (opt.impl == "baseline") {
            launch_naive_gemm(d_a, d_b, d_c, m, n, k, stream);
        } else if (opt.impl == "naive") {
            launch_naive_gemm(d_a, d_b, d_c, m, n, k, stream);
        } else if (opt.impl == "tiled") {
            launch_tiled_gemm(d_a, d_b, d_c, m, n, k, stream);
        }
        check_cuda(cudaEventRecord(stop, stream), "record stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "compute elapsed time");
        (void)elapsed_ms;  // remove once timing implemented
    } else if (opt.impl == "cublas") {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate");
        check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check_cuda(cudaEventRecord(start, stream), "record start");
        check_cublas(
            cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n,
                        m,
                        k,
                        &alpha,
                        d_b,
                        n,
                        d_a,
                        k,
                        &beta,
                        d_c,
                        n),
            "cublasSgemm");
        check_cuda(cudaEventRecord(stop, stream), "record stop");
        check_cuda(cudaEventSynchronize(stop), "sync stop");
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
        check_cublas(cublasDestroy(handle), "cublasDestroy");
    } else {
        throw std::invalid_argument("Unknown implementation: " + opt.impl);
    }

    
    check_cuda(cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "copy d_c->h_c");

    if (opt.verify) {
        cublasHandle_t handle;
        check_cublas(cublasCreate(&handle), "cublasCreate for verification");

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        check_cublas(
            cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    d_b, n,
                    d_a, k,
                    &beta,
                    d_c, n),
        "cublasSgemm for verification");

        check_cuda(cudaMemcpy(h_ref.data(), d_c, bytes_c, cudaMemcpyDeviceToHost), "copy d_c->h_ref");
        check_cublas(cublasDestroy(handle), "cublasDestroy for verification");

        float max_err = 0.0f;
        for (size_t i = 0; i < h_c.size(); ++i) {
            float err = std::fabs(h_c[i] - h_ref[i]);
            if (err > max_err) max_err = err;
        }
        std::cout << "MaxAbsError=" << std::scientific << max_err << std::endl;
    }
    if (elapsed_ms > 0.0f) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Impl=" << opt.impl << " M=" << m << " N=" << n << " K=" << k
                  << " Time(ms)=" << elapsed_ms << " GFLOP/s=" << gflops(m, n, k, elapsed_ms)
                  << std::endl;
    }

    check_cuda(cudaFree(d_a), "cudaFree d_a");
    check_cuda(cudaFree(d_b), "cudaFree d_b");
    check_cuda(cudaFree(d_c), "cudaFree d_c");
    check_cuda(cudaEventDestroy(start), "destroy start event");
    check_cuda(cudaEventDestroy(stop), "destroy stop event");
    check_cuda(cudaStreamDestroy(stream), "destroy stream");
    return 0;
}

