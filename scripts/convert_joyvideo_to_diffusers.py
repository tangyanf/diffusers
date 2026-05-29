import argparse
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from diffusers import (
    AutoencoderKLJoyVideo,
    JoyVideoPipeline,
    JoyVideoTransformer3DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


def get_transformer_config() -> Dict[str, Any]:
    config = {
        "hidden_size": 4096,
        "in_channels": 64,
        "num_attention_heads": 32,
        "num_layers": 40,
        "out_channels": 64,
        "patch_size": [1, 1, 1],
        "rope_dim_list": [16, 56, 56],
        "text_dim": 4096,
        "rope_type": "rope",
        "theta": 256,
    }
    return config


def convert_transformer(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if "model" in checkpoint:
        original_state_dict = checkpoint["model"]
    else:
        original_state_dict = checkpoint

    attn_suffixes = (
        "img_attn_qkv.",
        "img_attn_q_norm.",
        "img_attn_k_norm.",
        "img_attn_proj.",
        "txt_attn_qkv.",
        "txt_attn_q_norm.",
        "txt_attn_k_norm.",
        "txt_attn_proj.",
    )
    remapped = {}
    for key, value in original_state_dict.items():
        new_key = key
        if key.startswith("double_blocks."):
            for suffix in attn_suffixes:
                if "." + suffix in key and ".attn." + suffix not in key:
                    new_key = key.replace("." + suffix, ".attn." + suffix)
                    break
        # Convert img_in.weight from Linear [out, in] to Conv3d [out, in, 1, 1, 1]
        if key == "img_in.weight" and value.dim() == 2:
            value = value.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        remapped[new_key] = value

    config = get_transformer_config()
    with init_empty_weights():
        transformer = JoyVideoTransformer3DModel(**config)
    transformer.load_state_dict(remapped, strict=True, assign=True)
    return transformer


def convert_vae(vae_path: str):
    """Load VAE from a diffusers-format pretrained directory."""
    vae = AutoencoderKLJoyVideo.from_pretrained(vae_path)
    return vae


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer_ckpt_path",
        type=str,
        default=None,
        help="Path to original transformer checkpoint (.pth file)",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        help="Path to VAE pretrained directory (diffusers format)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path to Qwen2.5-VL text encoder",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to text_encoder_path if not set)",
    )
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where converted model should be saved",
    )
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    parser.add_argument("--flow_shift", type=float, default=5.159)
    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

if __name__ == "__main__":
    args = get_args()
    transformer = None
    vae = None
    dtype = DTYPE_MAPPING[args.dtype]

    if args.save_pipeline:
        assert args.transformer_ckpt_path is not None and args.vae_ckpt_path is not None
        assert args.text_encoder_path is not None

    if args.transformer_ckpt_path is not None:
        transformer = convert_transformer(args.transformer_ckpt_path)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.vae_ckpt_path is not None:
        vae = convert_vae(args.vae_ckpt_path)
        vae = vae.to(dtype=dtype)
        if not args.save_pipeline:
            vae.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.save_pipeline:
        tokenizer_path = args.tokenizer_path or args.text_encoder_path
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.text_encoder_path, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.flow_shift
        )

        pipe = JoyVideoPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        )
        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
