#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

inline dim3 make_grid(int m, int n) {
    return dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (m + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1);
}

__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

inline void launch_naive_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
    (void)d_a;
    (void)d_b;
    (void)d_c;
    (void)grid;
    (void)block;
    (void)stream;
}

inline void launch_tiled_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error launching gemm_tiled_kernel: %s\n", cudaGetErrorString(err));
    }
    (void)d_a;
    (void)d_b;
    (void)d_c;
    (void)grid;
    (void)block;
    (void)stream;
}

