# MAISI VAE with Multi-GPU Tensor Splitting Parallelism

Generating high-resolution 3D CT volumes is costly, requiring substantial compute and memory due to large feature maps used in 3D CNNs. To address this, NVIDIA introduced the MAISI VAE, which compresses CT volumes into latent space for efficient generative modeling. Rather than relying on sliding-window inference (which causes boundary artifacts), they propose **Tensor Splitting Parallelism (TSP)** to divide feature maps into smaller segments, distributing them across multiple devices, and merging them to yield the layer's output. 

Concurrent processing across multiple devices significantly accelerates inference. It also reduces peak memory usage on a single device by allowing segments to be processed sequentially. However, the current open-source implementation of MAISI VAE does not support multi-device TSP. Instead, it relies on sequential single-device processing and frequent costly GPU-to-CPU data transfers for merging layer outputs. This creates bottlenecks for downstream applications through inefficient utilization of GPUs and slow inference.

![Overview of multi-GPU tensor splitting parallelism.](./assets/overview.png)
*Figure inspired by Guo et al. WACV 2025.*

## Usage

An improved version of `AutoencoderKLMaisi` implements multi-device TSP and several tweaks to reduce cost GPU-to-CPU communication bottlenecks. It is a drop-in replacement for [`AutoencoderKLMaisi`](https://github.com/Project-MONAI/MONAI/blob/main/monai/apps/generation/maisi/networks/autoencoderkl_maisi.py). You can replace all `AutoencoderKLMaisi` imports with the code provided in this repository [here](./src/autoencoderkl_maisi.py). Note that a new argument `num_devices` to control the number of devices for TSP is added (defaults to 1). Setting this to `None` uses all available GPUs for TSP.

An example is also provided [here](./src/extras/demo.ipynb). For further details, please follow their instructions on [NVIDIA-Medtech/NV-Generate-CTMR](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR). 

## Benchmark

The encoding and decoding performance of the improved MAISI was benchmarked against the original open-sourced implementation. The inference time (in seconds) and peak VRAM use (in GB) is measured for a 3D CT volume of $512 \times 512 \times 192$ voxels with $0.75 \times 0.75 \times 1.5$ spacing. In all cases, mixed-precision is used with memory saving enabled on up to 4 NVIDIA H200 GPUs. The improved MAISI VAE achieved significantly faster inference time in both single-device and multi-device TSP settings. Interestingly, the additional overhead of launching CUDA kernel on multiple devices often outweights the performance of single-device cuDNN optimization.

| **Method** | $\mathbf{N}_\textbf{splits}$ | $\mathbf{N}_\textbf{devices}$ | **Encoding** | **Decoding** |
| :-- | :--: | :--: | :--: | :--: |
| MAISI | 1 | 1 | 83.24s (19.24 GB) | 154.39s (49.89 GB) |
| MAISI, Improved | 1 | 1 | 46.73s (24.68 GB) | 104.04s (49.89 GB) |
| MAISI | 2 | 1 | 46.53s (27.48 GB) | 134.28s (44.1 GB) |
| MAISI, Improved | 2 | 1 | 1.52s (30.51 GB) | 69.32s (73.16 GB) |
| | 2 | 2 | 1.62s (30.48 GB) | 67.59s (73.24 GB) |
| MAISI | 4 | 1 | 52.38s (23.12 GB) | 72.61s (47.52 GB) |
| **MAISI, Improved** | **4** | **1** | **1.34s (30.72 GB)** | **2.32s (72.86 GB)** |
| | 4 | 4 | 1.40s (30.76 GB) | 2.45s (72.93 GB) |

## Resources

For further details, please refer to the below references:

- [NV-Generate-CTMR on GitHub](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR)
- [NV-Generate-CT on HuggingFace](https://huggingface.co/nvidia/NV-Generate-CT)
- [MAISI (v1) paper from WACV 2025](https://arxiv.org/pdf/2409.11169)
- [MONAI](https://monai.io/)

## Contact

For any questions, feel free to reach out via email: pranavk@umd.edu.