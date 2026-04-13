import torch
import torch.nn as nn

from monai.transforms import *

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi as AutoencoderKLMaisi_,
)
from ..autoencoderkl_maisi import AutoencoderKlMaisi


def get_config() -> dict:
    return {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "num_channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1.0e-6,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "norm_float16": True,
        "dim_split": 1,
        "save_mem": True,
    }


def load_vae_model(
    path: str,
    version: int = 1,
    num_splits: int = 1,
    num_devices: int = 1,
    device: torch.device = "cuda",
) -> nn.Module:
    if version == 1 and num_devices != 1:
        raise ValueError(
            "Original MAISI VAE does not support multiple devices.Either set `version=2` or `num_devices=1`"
        )

    config = get_config()
    config["num_splits"] = num_splits
    if version == 1:
        model = AutoencoderKLMaisi_(**config).to(device)
    elif version == 2:
        config["num_devices"] = num_devices
        model = AutoencoderKlMaisi(**config).to(device)
    else:
        ValueError("`version` must be 1 or 2. Got:", version)

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def load_image(
    path: str, img_size: list = [512, 512, 192], img_spacing: list = [0.75, 0.75, 1.5]
) -> torch.Tensor:
    transforms = Compose(
        [
            LoadImage(ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacing(pixdim=img_spacing, mode="trilinear"),
            ResizeWithPadOrCrop(spatial_size=img_size),
        ]
    )
    return transforms(path).unsqueeze(0)
