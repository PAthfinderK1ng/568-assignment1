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

__global__ void gemm_tiled_1x8_kernel_16(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M,
                                        int N,
                                        int K) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][128];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col0 = blockIdx.x * 128 + threadIdx.x;
    int col1 = col0 + 16;
    int col2 = col0 + 32;
    int col3 = col0 + 48;
    int col4 = col0 + 64;
    int col5 = col0 + 80;
    int col6 = col0 + 96;
    int col7 = col0 + 112;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;
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
        if (b_row < K && col2 < N) {
            tile_B[threadIdx.y][threadIdx.x + 32] = B[b_row * N + col2];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 32] = 0.0f;
        }
        if (b_row < K && col3 < N) {
            tile_B[threadIdx.y][threadIdx.x + 48] = B[b_row * N + col3];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 48] = 0.0f;
        }
        if (b_row < K && col4 < N) {
            tile_B[threadIdx.y][threadIdx.x + 64] = B[b_row * N + col4];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 64] = 0.0f;
        }
        if (b_row < K && col5 < N) {
            tile_B[threadIdx.y][threadIdx.x + 80] = B[b_row * N + col5];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 80] = 0.0f;
        }
        if (b_row < K && col6 < N) {
            tile_B[threadIdx.y][threadIdx.x + 96] = B[b_row * N + col6];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 96] = 0.0f;
        }
        if (b_row < K && col7 < N) {
            tile_B[threadIdx.y][threadIdx.x + 112] = B[b_row * N + col7];
        } else {
            tile_B[threadIdx.y][threadIdx.x + 112] = 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a = tile_A[threadIdx.y][k];
            acc0 = fmaf(a, tile_B[k][threadIdx.x], acc0);
            acc1 = fmaf(a, tile_B[k][threadIdx.x + 16], acc1);
            acc2 = fmaf(a, tile_B[k][threadIdx.x + 32], acc2);
            acc3 = fmaf(a, tile_B[k][threadIdx.x + 48], acc3);
            acc4 = fmaf(a, tile_B[k][threadIdx.x + 64], acc4);
            acc5 = fmaf(a, tile_B[k][threadIdx.x + 80], acc5);
            acc6 = fmaf(a, tile_B[k][threadIdx.x + 96], acc6);
            acc7 = fmaf(a, tile_B[k][threadIdx.x + 112], acc7);
        }
        __syncthreads();
    }
    if (row < M && col0 < N) {
        C[row * N + col0] = acc0;
    }
    if (row < M && col1 < N) {
        C[row * N + col1] = acc1;
    }
    if (row < M && col2 < N) {
        C[row * N + col2] = acc2;
    }
    if (row < M && col3 < N) {
        C[row * N + col3] = acc3;
    }
    if (row < M && col4 < N) {
        C[row * N + col4] = acc4;
    }
    if (row < M && col5 < N) {
        C[row * N + col5] = acc5;
    }
    if (row < M && col6 < N) {
        C[row * N + col6] = acc6;
    }
    if (row < M && col7 < N) {
        C[row * N + col7] = acc7;
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
        dim3 grid((N + 127) / 128, (M + 15) / 16, 1);
        gemm_tiled_1x8_kernel_16<<<grid, block, 0, stream>>>(d_a, d_b, d_c, M, N, K);
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

