import argparse
import numpy as np
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.amp import autocast

from huggingface_hub import hf_hub_download
from .utils import *

# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
#     print("Count:", torch.cuda.device_count())

# Download MAISI VAE weights
model_path = hf_hub_download(
    repo_id="nvidia/NV-Generate-CT",
    filename="models/autoencoder_v1.pt",
)

# Download example image from CT-RATE
# (https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
img_path = hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    filename="dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_1.nii.gz",
    repo_type="dataset",
)

def step(
    fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    amp: bool = True,
    device: torch.device = "cuda",
    repetitions: int = 10,
) -> float:
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10
    timings = np.zeros((repetitions, 1))
    peaks = np.zeros((repetitions, 1))

    with torch.no_grad(), autocast(device, enabled=amp):
        for rep in range(repetitions):
            torch.cuda.reset_peak_memory_stats()
            starter.record()
            
            _ = fn(inputs)
            
            ender.record()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            torch.cuda.synchronize()
            time = starter.elapsed_time(ender) / 100
            timings[rep] = time
            peaks[rep] = peak_mem

    return round(np.mean(timings), 2), round(np.mean(peaks), 2)

def benchmark(
    model: nn.Module,
    img: torch.Tensor,
    amp: bool = True,
    device: torch.device = "cuda",
    repetitions: int = 10,
) -> None:
    img = img.to(device, non_blocking=True)
    model = model.to(device)
    
    # Warmup
    with torch.no_grad(), autocast(device, enabled=amp):
        z = model.encode_stage_2_inputs(img)

    # Encoding
    enc_mean_time, enc_mean_peak = step(model.encode_stage_2_inputs, img, amp, device, repetitions)

    # Decoding
    dec_mean_time, dec_mean_peak = step(model.decode_stage_2_outputs, z, amp, device, repetitions)
    
    print(f"    Encoding: ({enc_mean_time}s, {enc_mean_peak} GB), Decoding: ({dec_mean_time}s, {dec_mean_peak} GB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--num_splits", type=int, required=True)
    parser.add_argument("--num_devices", type=int, required=True)

    args = parser.parse_args()

    version = args.version
    device = "cuda"
    img_spacing = [0.75, 0.75, 1.5]
    sizes = [[128, 128, 128], [256, 256, 256], [512, 512, 192], [512, 512, 256]]
    img_size = sizes[args.img_size]
    num_splits = args.num_splits
    num_devices = args.num_devices
    
    print(
        f"MAISI v{version}: img_size={img_size}, num_splits={num_splits}"
    )

    img = load_image(img_path, img_size, img_spacing)
    model = load_vae_model(
        model_path, version, num_splits, num_devices, device
    )
    benchmark(model, img)