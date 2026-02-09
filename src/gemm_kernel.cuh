#pragma once

#include <cuda_runtime.h>
#include <cstdio>

constexpr int DEFAULT_BLOCK_SIZE = 32;

inline dim3 make_grid(int m, int n, int block_size) {
    return dim3((n + block_size - 1) / block_size,
                (m + block_size - 1) / block_size,
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
        #pragma unroll 4
        for (int k = 0; k < K; ++k) {
            value = fmaf(A[row * K + k], B[k * N + col], value);
        }
        C[row * N + col] = value;
    }
}

template <int TileSize>
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M,
                                  int N,
                                  int K) {
    __shared__ float tile_A[TileSize][TileSize];
    __shared__ float tile_B[TileSize][TileSize];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    int tiles = (K + TileSize - 1) / TileSize;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * TileSize + threadIdx.x;
        int b_row = t * TileSize + threadIdx.y;
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
        #pragma unroll 8
        for (int k = 0; k < TileSize; ++k) {
            value = fmaf(tile_A[threadIdx.y][k], tile_B[k][threadIdx.x], value);
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

__global__ void gemm_tiled_1x2_kernel_16(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M,
                                        int N,
                                        int K) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][32];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col0 = blockIdx.x * 32 + threadIdx.x;
    int col1 = col0 + 16;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int tiles = (K + 15) / 16;
    for (int t = 0; t < tiles; ++t) {
        int a_col = t * 16 + threadIdx.x;
        int b_row = t * 16 + threadIdx.y;
        if (row < M && a_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < K && col0 < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col0];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (b_row < K && col1 < N) {
            tile_B[threadIdx.y][threadIdx.x + 16] = B[b_row * N + col1];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 16] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a = tile_A[threadIdx.y][k];
            acc0 = fmaf(a, tile_B[k][threadIdx.x], acc0);
            acc1 = fmaf(a, tile_B[k][threadIdx.x + 16], acc1);
        }
        __syncthreads();
    }
    if (row < M && col0 < N) {
        C[row * N + col0] = acc0;
    }
    if (row < M && col1 < N) {
        C[row * N + col1] = acc1;
    }
}

inline void launch_naive_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              cudaStream_t stream) {
    dim3 block(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE, 1);
    dim3 grid = make_grid(M, N, DEFAULT_BLOCK_SIZE);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
}

inline void launch_tiled_gemm(const float* d_a,
                              const float* d_b,
                              float* d_c,
                              int M,
                              int N,
                              int K,
                              int tile_size,
                              cudaStream_t stream) {
    if (tile_size == 16) {
        dim3 block(16, 16, 1);
        dim3 grid((N + 31) / 32, (M + 15) / 16, 1);
        gemm_tiled_1x2_kernel_16<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
    } else if (tile_size == 32) {
        dim3 block(32, 32, 1);
        dim3 grid = make_grid(M, N, 32);
        gemm_tiled_kernel<32><<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
    } else {
        fprintf(stderr, "Unsupported tile size: %d. Defaulting to 32.\n", tile_size);
        dim3 block(32, 32, 1);
        dim3 grid = make_grid(M, N, 32);
        gemm_tiled_kernel<32><<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error launching gemm_tiled_kernel: %s\n", cudaGetErrorString(err));
    }
}

