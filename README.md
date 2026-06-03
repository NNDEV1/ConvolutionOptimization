# ConvolutionOptimization

CUDA implementations of 2D convolution for a fixed Fashion-MNIST CNN, built as a progression from a naive GPU kernel to a heavily tuned, GEMM-style forward pass. The rest of the network (pooling, fully connected layers, activations) comes from a small C++/Eigen stack based on [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp), adapted for the ECE 408 class project.

The main idea throughout is to treat convolution as a matrix multiply: unroll the input into columns, multiply by reshaped filters, then permute into the output layout. Later milestones fuse those steps and tune tiling, memory, and occupancy on the GPU.

## What it runs

The `project/` tree builds several inference binaries that load pretrained weights (`weights-86.bin`) and run the same network on Fashion-MNIST test images. Only the custom convolution layers swap implementations; everything else is shared.

Typical GPU targets (see `project/CMakeLists.txt`):

| Target | Convolution implementation |
|--------|----------------------------|
| `m1_cpu` | CPU reference (`cpu-new-forward`) |
| `m1_gpu` | Naive per-output-element CUDA kernel |
| `m2_unroll` | Separate unroll → tiled matmul → permute kernels |
| `m2_fused` | Single fused unroll-matmul-permute kernel |
| `m3` | Final optimized fused kernel (production path) |

Build from `project/` with CMake and a CUDA 11+ toolchain (SM 80/86 configured in CMake). Example:

```bash
cd project
mkdir -p build && cd build
cmake ..
cmake --build .
./m3 [batch_size] [--competition]
```

Data path and batch defaults are wired in `m1_gpu.cc` / `m1_cpu.cc` for the course environment; adjust `MNIST` paths locally if needed.

## Repository layout

```
project/
├── src/                    # Network, layers, MNIST loader, optimizers
│   └── layer/custom/       # Custom conv kernels (see below)
├── m3/                     # Incremental snapshots of the final kernel
├── ece408net.cc            # Network definition (two conv blocks + FC head)
├── m1_cpu.cc / m1_gpu.cc   # Inference drivers
└── CMakeLists.txt          # Milestone executables
```

## Kernel progression (`src/layer/custom`)

These files plug into `Conv_Custom` via `GPUInterface` (`gpu-new-forward.h`).

- **`new-forward.cu`** — Baseline GPU convolution: one thread per output element, nested loops over channels and the K×K stencil. Correctness-first starting point for milestone 1.

- **`unroll-new-forward.cu`** — Milestone 2 “unrolled” path: a dedicated `matrix_unrolling_kernel` builds the im2col matrix, `matmul.cu` runs a tiled shared-memory GEMM (`matrixMultiplyShared`), then a permute kernel maps the result into NCHW-style output. Three launches, but each stage is simple to reason about.

- **`kernel-fusion-forward.cu`** — Fused variant: one `matmul_conv_fused` kernel performs tiled multiply-accumulate while loading filter weights and input patches on the fly, avoiding separate unroll and permute passes over global memory.

- **`m3-forward.cu`** — Final optimized kernel used by the `m3` binary. Still a fused GEMM-style conv, but with a tuned tile geometry (`TILE_M` / `TILE_N` / `TILE_K`), `__restrict__` pointers, and **register coarsening** (`COARSE = 2`: each thread computes two output rows per tile). Filter and input tiles live in shared memory; the host wrapper handles allocation and H↔D copies.

Supporting code: `matmul.cu` / `matmul.h` (reference tiled GEMM), `gpu-utils.cu`, and CPU fallbacks in `cpu-new-forward.*`.

## Milestone 3 experiments (`project/m3/`)

`m3/` holds standalone snapshots of `m3-forward.cu` as individual optimizations were layered on. Each folder is a self-contained variant you can diff against the others or drop into `src/layer/custom/` when experimenting.

| Directory | Focus |
|-----------|--------|
| `op_0/` | `__constant__` memory for filter weights when they fit |
| `op_1/` | `__restrict__` on kernel pointers for alias analysis |
| `op_2/` | Cleaned fused matmul baseline (tiling + on-the-fly im2col) |
| `op_3/` | Tile-size / register **coarsening** sweep (`SWEEP_CONFIG`, `COARSEN_X`) |
| `op_5/` | FP16 tiles in shared memory (`__half`) with FP32 accumulation |
| `req_0/` | Multi-stream pipeline: pinned host memory, chunked batch, async unroll → matmul → permute |
| `req_1/` | Tensor Core path via WMMA (`matmul_conv_fused_tensorcore`) |

The canonical integrated result lives in **`src/layer/custom/m3-forward.cu`**, not in a single `m3/` subfolder.

## Network sketch

Two 7×7 conv layers (4 and 16 output maps) with ReLU and max pooling, then two fully connected layers and softmax—roughly 86×86 inputs after preprocessing, 10 classes out. GPU builds route both conv layers through `Conv_Custom` and the selected forward implementation.

## Third party

`project/third_party/` includes Eigen (header-only linear algebra for FC layers and utilities). See `third_party/readme.md` in that tree for upstream details.
