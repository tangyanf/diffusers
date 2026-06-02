"""Convert JoyImage checkpoints (T2I and DMD distillation) to diffusers format.

Usage – normal checkpoint::

    python scripts/convert_joyimage_to_diffusers.py \
        --transformer_ckpt_path /path/to/transformer.pth \
        --vae_ckpt_path /path/to/vae.pth \
        --text_encoder_path /path/to/Qwen3-VL \
        --save_pipeline \
        --output_path /path/to/output

Usage – DMD distillation checkpoint::

    python scripts/convert_joyimage_to_diffusers.py \
        --distill_ckpt_path /path/to/step_XXX.pth \
        --use_ema \
        --vae_ckpt_path /path/to/vae.pth \
        --text_encoder_path /path/to/Qwen3-VL \
        --save_pipeline \
        --output_path /path/to/output
"""

import argparse
import json
import os
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from diffusers import (
    AutoencoderKLWan,
    JoyImagePipeline,
    JoyImageTransformer3DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


# ---------------------------------------------------------------------------
# VAE key remapping (Wan format -> diffusers AutoencoderKLWan format)
# Copied from convert_joyimage_edit_to_diffusers.py
# ---------------------------------------------------------------------------


def convert_vae(vae_ckpt_path: str):
    old_state_dict = torch.load(vae_ckpt_path, weights_only=True)
    new_state_dict = {}

    middle_key_mapping = {
        "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
        "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
        "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
        "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
        "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
        "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
        "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
        "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
        "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
        "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
        "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",
        "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
        "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
        "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
        "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
        "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
        "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
        "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
        "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
        "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
        "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
        "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
        "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    }

    attention_mapping = {
        "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
        "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
        "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
        "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
        "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
        "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
        "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
        "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    }

    head_mapping = {
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.bias": "encoder.conv_out.bias",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.bias": "decoder.conv_out.bias",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }

    quant_mapping = {
        "conv1.weight": "quant_conv.weight",
        "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
    }

    for key, value in old_state_dict.items():
        if key in middle_key_mapping:
            new_state_dict[middle_key_mapping[key]] = value
        elif key in attention_mapping:
            new_state_dict[attention_mapping[key]] = value
        elif key in head_mapping:
            new_state_dict[head_mapping[key]] = value
        elif key in quant_mapping:
            new_state_dict[quant_mapping[key]] = value
        elif key == "encoder.conv1.weight":
            new_state_dict["encoder.conv_in.weight"] = value
        elif key == "encoder.conv1.bias":
            new_state_dict["encoder.conv_in.bias"] = value
        elif key == "decoder.conv1.weight":
            new_state_dict["decoder.conv_in.weight"] = value
        elif key == "decoder.conv1.bias":
            new_state_dict["decoder.conv_in.bias"] = value
        elif key.startswith("encoder.downsamples."):
            new_key = key.replace("encoder.downsamples.", "encoder.down_blocks.")
            if ".residual.0.gamma" in new_key:
                new_key = new_key.replace(".residual.0.gamma", ".norm1.gamma")
            elif ".residual.2.bias" in new_key:
                new_key = new_key.replace(".residual.2.bias", ".conv1.bias")
            elif ".residual.2.weight" in new_key:
                new_key = new_key.replace(".residual.2.weight", ".conv1.weight")
            elif ".residual.3.gamma" in new_key:
                new_key = new_key.replace(".residual.3.gamma", ".norm2.gamma")
            elif ".residual.6.bias" in new_key:
                new_key = new_key.replace(".residual.6.bias", ".conv2.bias")
            elif ".residual.6.weight" in new_key:
                new_key = new_key.replace(".residual.6.weight", ".conv2.weight")
            elif ".shortcut.bias" in new_key:
                new_key = new_key.replace(".shortcut.bias", ".conv_shortcut.bias")
            elif ".shortcut.weight" in new_key:
                new_key = new_key.replace(".shortcut.weight", ".conv_shortcut.weight")
            new_state_dict[new_key] = value
        elif key.startswith("decoder.upsamples."):
            parts = key.split(".")
            block_idx = int(parts[2])

            if "residual" in key:
                if block_idx in [0, 1, 2]:
                    new_block_idx, resnet_idx = 0, block_idx
                elif block_idx in [4, 5, 6]:
                    new_block_idx, resnet_idx = 1, block_idx - 4
                elif block_idx in [8, 9, 10]:
                    new_block_idx, resnet_idx = 2, block_idx - 8
                elif block_idx in [12, 13, 14]:
                    new_block_idx, resnet_idx = 3, block_idx - 12
                else:
                    new_state_dict[key] = value
                    continue

                if ".residual.0.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm1.gamma"
                elif ".residual.2.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.bias"
                elif ".residual.2.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv1.weight"
                elif ".residual.3.gamma" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.norm2.gamma"
                elif ".residual.6.bias" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.bias"
                elif ".residual.6.weight" in key:
                    new_key = f"decoder.up_blocks.{new_block_idx}.resnets.{resnet_idx}.conv2.weight"
                else:
                    new_key = key
                new_state_dict[new_key] = value

            elif ".shortcut." in key:
                if block_idx == 4:
                    new_key = key.replace(".shortcut.", ".resnets.0.conv_shortcut.")
                    new_key = new_key.replace("decoder.upsamples.4", "decoder.up_blocks.1")
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                    new_key = new_key.replace(".shortcut.", ".conv_shortcut.")
                new_state_dict[new_key] = value

            elif ".resample." in key or ".time_conv." in key:
                upsample_map = {3: 0, 7: 1, 11: 2}
                if block_idx in upsample_map:
                    new_key = key.replace(
                        f"decoder.upsamples.{block_idx}",
                        f"decoder.up_blocks.{upsample_map[block_idx]}.upsamplers.0",
                    )
                else:
                    new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
            else:
                new_key = key.replace("decoder.upsamples.", "decoder.up_blocks.")
                new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    with init_empty_weights():
        vae = AutoencoderKLWan()
    vae.load_state_dict(new_state_dict, strict=True, assign=True)
    return vae


# ---------------------------------------------------------------------------
# Transformer config & conversion
# ---------------------------------------------------------------------------


def get_transformer_config() -> Dict[str, Any]:
    return {
        "hidden_size": 4096,
        "in_channels": 16,
        "num_attention_heads": 32,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "rope_dim_list": [16, 56, 56],
        "text_dim": 4096,
        "rope_type": "rope",
        "theta": 10000,
    }


def convert_transformer(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, weights_only=True)
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
        remapped[new_key] = value

    config = get_transformer_config()
    with init_empty_weights():
        transformer = JoyImageTransformer3DModel(**config)
    transformer.load_state_dict(remapped, strict=True, assign=True)
    return transformer


def convert_distill_transformer(ckpt_path: str, use_ema: bool = False):
    """Convert a DMD distillation training checkpoint to JoyImageTransformer3DModel."""
    print(f"Loading distillation checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Expected a dict checkpoint, got {type(checkpoint)}. "
            "Please check that the path points to a valid training checkpoint."
        )

    if use_ema and "ema_model" in checkpoint:
        print("Using EMA model weights from checkpoint.")
        original_state_dict = checkpoint["ema_model"]
    elif "model" in checkpoint:
        if use_ema:
            print("WARNING: 'ema_model' key not found in checkpoint; falling back to 'model'.")
        original_state_dict = checkpoint["model"]
    else:
        original_state_dict = checkpoint

    cleaned = {}
    for k, v in original_state_dict.items():
        new_k = k
        for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    original_state_dict = cleaned

    step = checkpoint.get("step", "unknown")
    print(f"Checkpoint step: {step}, state dict has {len(original_state_dict)} keys")

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
        remapped[new_key] = value

    config = get_transformer_config()
    with init_empty_weights():
        transformer = JoyImageTransformer3DModel(**config)
    transformer.load_state_dict(remapped, strict=True, assign=True)
    return transformer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="Convert JoyImage checkpoints to diffusers format")
    parser.add_argument("--transformer_ckpt_path", type=str, default=None, help="Path to original transformer checkpoint")
    parser.add_argument("--vae_ckpt_path", type=str, default=None, help="Path to original VAE checkpoint")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to Qwen3-VL text encoder")
    parser.add_argument("--save_pipeline", action="store_true")
    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument("--dtype", default="bf16", help="Torch dtype to save the transformer in.")
    parser.add_argument("--flow_shift", type=float, default=7.0)
    parser.add_argument(
        "--distill_ckpt_path", type=str, default=None,
        help="Path to a DMD distillation training checkpoint (step_*.pth).",
    )
    parser.add_argument("--use_ema", action="store_true", help="Extract EMA model weights from distillation checkpoint.")
    parser.add_argument("--num_distill_timesteps", type=int, default=8, help="Number of inference timesteps for DMD sampling.")
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
        has_transformer = args.transformer_ckpt_path is not None or args.distill_ckpt_path is not None
        assert has_transformer and args.vae_ckpt_path is not None, \
            "save_pipeline requires --vae_ckpt_path and either --transformer_ckpt_path or --distill_ckpt_path"
        assert args.text_encoder_path is not None, "save_pipeline requires --text_encoder_path"

    if args.distill_ckpt_path is not None:
        transformer = convert_distill_transformer(args.distill_ckpt_path, use_ema=args.use_ema)
        transformer = transformer.to(dtype=dtype)
        if not args.save_pipeline:
            transformer.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
            print(f"Distillation transformer saved to: {args.output_path}")

    elif args.transformer_ckpt_path is not None:
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
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            args.text_encoder_path, torch_dtype=torch.bfloat16
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)

        flow_shift = args.flow_shift if args.flow_shift != 7.0 else 1.5
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=flow_shift)

        transformer = transformer.to("cuda")
        vae = vae.to("cuda")
        pipe = JoyImagePipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
        ).to("cuda")

        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")
        print(f"Full pipeline saved to: {args.output_path}")

        if args.distill_ckpt_path is not None:
            model_index_path = os.path.join(args.output_path, "model_index.json")
            with open(model_index_path) as f:
                model_index = json.load(f)
            model_index["_class_name"] = "JoyImageDMDPipeline"
            with open(model_index_path, "w") as f:
                json.dump(model_index, f, indent=2)
            print(f"Updated model_index.json: _class_name = 'JoyImageDMDPipeline'")
            print(
                f"\nDistillation model pipeline saved with flow_shift={flow_shift}."
                f"\nFor inference, use sampling_method='consistency_sampling' with "
                f"num_inference_steps={args.num_distill_timesteps}."
            )
