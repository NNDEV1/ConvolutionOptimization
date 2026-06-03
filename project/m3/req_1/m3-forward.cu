#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include "gpu-new-forward.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_conv_fused_tensorcore(const float *mask, const float *input, float *output,
                                             int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    namespace wmma = nvcuda::wmma;

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int N = Batch * Height_out * Width_out;
    const int K_total = Channel * K * K;

    const int row_base = blockIdx.y * WMMA_M;
    const int col_base = blockIdx.x * WMMA_N;

    const int lane = threadIdx.x;

    __shared__ half tile_A[WMMA_M * WMMA_K];
    __shared__ half tile_B[WMMA_K * WMMA_N];
    __shared__ float tile_C[WMMA_M * WMMA_N];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tile_k = 0; tile_k < K_total; tile_k += WMMA_K) {
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += warpSize) {
            int i = idx / WMMA_K;
            int j = idx % WMMA_K;

            int row = row_base + i;
            int k_idx = tile_k + j;

            float a = 0.0f;
            if (row < Map_out && k_idx < K_total) {
                a = mask[(size_t)row * K_total + k_idx];
            }
            tile_A[idx] = __float2half(a);
        }

        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += warpSize) {
            int i = idx / WMMA_N;
            int j = idx % WMMA_N;

            int k_idx = tile_k + i;
            int col = col_base + j;

            float b_val = 0.0f;
            if (k_idx < K_total && col < N) {
                int tmp = col;
                int w_out = tmp % Width_out;
                tmp /= Width_out;
                int h_out = tmp % Height_out;
                int b = tmp / Height_out;

                int c = k_idx / (K * K);
                int rem = k_idx % (K * K);
                int p = rem / K;
                int q = rem % K;

                b_val = input[(size_t)b * Channel * Height * Width +
                              (size_t)c * Height * Width +
                              (size_t)(h_out + p) * Width + (w_out + q)];
            }
            tile_B[idx] = __float2half(b_val);
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, tile_A, WMMA_K);
        wmma::load_matrix_sync(b_frag, tile_B, WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(tile_C, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += warpSize) {
        int i = idx / WMMA_N;
        int j = idx % WMMA_N;

        int row = row_base + i;
        int col = col_base + j;

        if (row < Map_out && col < N) {
            int tmp = col;
            int w_out = tmp % Width_out;
            tmp /= Width_out;
            int h_out = tmp % Height_out;
            int b = tmp / Height_out;

            output[(size_t)b * Map_out * Height_out * Width_out +
                   (size_t)row * Height_out * Width_out +
                   (size_t)h_out * Width_out + w_out] = tile_C[idx];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    (void)host_output;

    const size_t out_bytes = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    const size_t in_bytes = (size_t)Batch * Channel * Height * Width * sizeof(float);
    const size_t mask_bytes = (size_t)Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**)device_output_ptr, out_bytes);
    cudaMalloc((void**)device_input_ptr, in_bytes);
    cudaMalloc((void**)device_mask_ptr, mask_bytes);

    cudaMemcpy(*device_input_ptr, host_input, in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_bytes, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int N = Batch * Height_out * Width_out;

    dim3 blockDim(32, 1, 1);
    dim3 gridDim((N + WMMA_N - 1) / WMMA_N,
                 (Map_out + WMMA_M - 1) / WMMA_M,
                 1);

    matmul_conv_fused_tensorcore<<<gridDim, blockDim>>>(device_mask, device_input, device_output,
                                                         Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t out_bytes = (size_t)Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
