import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    Qwen2Tokenizer,
    Qwen3VLForConditionalGeneration,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLWan, JoyImageEditTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import JoyImagePipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
Examples:
    ```python
    >>> import torch
    >>> from diffusers import JoyImagePipeline

    >>> pipe = JoyImagePipeline.from_pretrained("path/to/joyimage", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> output = pipe(
    ...     prompt="A beautiful sunset over mountains.",
    ...     height=1024,
    ...     width=1024,
    ...     num_inference_steps=40,
    ...     guidance_scale=4.0,
    ...     generator=torch.manual_seed(0),
    ... )
    >>> output.images[0].save("joyimage_t2i.png")
    ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        if "timesteps" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in set(inspect.signature(scheduler.set_timesteps).parameters.keys()):
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class JoyImagePipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using the JoyImage architecture.

    Uses a Qwen3-VL text encoder, a dual-stream DiT transformer, and a WAN VAE.
    Supports both standard Euler flow-matching sampling and consistency (DMD)
    sampling via the ``sampling_method`` argument.

    Model offloading order: text_encoder -> transformer -> vae.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLWan,
        text_encoder: Qwen3VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: JoyImageEditTransformer3DModel,
        text_token_max_length: int = 512,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.text_token_max_length = text_token_max_length

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.prompt_template_encode = {
            "image": (
                "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            ),
        }
        self.prompt_template_encode_start_idx = {
            "image": 34,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_decoder_hidden_states(self, forward_fn, **kwargs):
        captured = {}

        def _hook(_module, _input, output):
            captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

        handle = self.text_encoder.model.language_model.layers[-1].register_forward_hook(_hook)
        try:
            forward_fn(**kwargs)
        finally:
            handle.remove()
        return captured["hidden_states"]

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        template_type: str = "image",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        template = self.prompt_template_encode[template_type]
        drop_idx = self.prompt_template_encode_start_idx[template_type]

        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.text_token_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        hidden_states = self._get_last_decoder_hidden_states(
            self.text_encoder,
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
        )

        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]

        max_seq_len = min(
            self.text_token_max_length,
            max(u.size(0) for u in split_hidden_states),
            max(u.size(0) for u in attn_mask_list),
        )
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        template_type: str = "image",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, template_type, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` has invalid keys.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError("`prompt` has to be of type `str` or `list`.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` is required.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` is required.")

    def normalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            latent = latent * self.vae.config.scaling_factor
        return latent

    def denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
            )
            latent = latent * latents_std + latents_mean
        else:
            latent = latent / self.vae.config.scaling_factor
        return latent

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels_latents,
            1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents

    # ------------------------------------------------------------------
    # Pipeline properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 40,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sampling_method: str = "euler",
        deterministic_sampling: bool = False,
        enable_denormalization: bool = True,
    ):
        r"""
        Generate images from text prompts.

        Args:
            prompt (`str` or `List[str]`):
                The prompt(s) to guide generation.
            height (`int`, *optional*, defaults to 1024):
                Height of the generated image in pixels.
            width (`int`, *optional*, defaults to 1024):
                Width of the generated image in pixels.
            num_inference_steps (`int`, *optional*, defaults to 40):
                Number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) to suppress undesired content.
            sampling_method (`str`, *optional*, defaults to ``"euler"``):
                Sampling strategy. ``"euler"`` for standard flow-matching ODE;
                ``"consistency_sampling"`` for DMD distillation models.
            deterministic_sampling (`bool`, *optional*, defaults to ``False``):
                When True and using ``"consistency_sampling"``, re-use the initial noise
                for each step instead of sampling fresh noise.

        Examples:

        Returns:
            [`~pipelines.joyimage.JoyImagePipelineOutput`] or `tuple`:
                Generated image(s).
        """
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                negative_prompt = [""] * batch_size

            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        target_dtype = prompt_embeds.dtype

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if sampling_method == "consistency_sampling":
                # -------------------------------------------------------
                # Consistency / DMD distillation sampling
                # Does NOT support classifier-free guidance.
                # -------------------------------------------------------
                if self.do_classifier_free_guidance:
                    raise ValueError(
                        "Classifier-free guidance is not supported with consistency_sampling. "
                        "Set guidance_scale <= 1."
                    )
                init_noise = latents.clone()
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    latent_model_input = latents
                    t_expand = t.repeat(latent_model_input.shape[0])

                    with torch.autocast(device_type="cuda", dtype=target_dtype):
                        flow_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            return_dict=False,
                        )[0]

                    # x0 prediction: x0 = x_t - flow_pred * sigma
                    sigma = (t_expand / 1000).view(-1, 1, 1, 1, 1)
                    image_pred = latent_model_input - flow_pred * sigma

                    if i == len(timesteps) - 1:
                        latents = image_pred
                        if progress_bar is not None:
                            progress_bar.update()
                        break

                    timestep_next = timesteps[i + 1].repeat(latent_model_input.shape[0])
                    sigma_next = (timestep_next / 1000).view(-1, 1, 1, 1, 1).to(latent_model_input.dtype)

                    if deterministic_sampling:
                        noise_next = init_noise
                    else:
                        noise_next = torch.randn(
                            latents.shape, generator=generator, dtype=target_dtype, device=device
                        )

                    # x_{t+1} = (1 - sigma_next) * x0 + sigma_next * noise
                    latents = (1 - sigma_next) * image_pred + noise_next * sigma_next

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    if progress_bar is not None:
                        progress_bar.update()
            else:
                # -------------------------------------------------------
                # Standard Euler flow-matching ODE sampling
                # -------------------------------------------------------
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    latent_model_input = (
                        torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    )
                    t_expand = t.repeat(latent_model_input.shape[0])

                    prompt_embeds_input = (
                        torch.cat([prompt_embeds, negative_prompt_embeds])
                        if self.do_classifier_free_guidance
                        else prompt_embeds
                    )

                    with torch.autocast(device_type="cuda", dtype=target_dtype):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds_input,
                            return_dict=False,
                        )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        if progress_bar is not None:
                            progress_bar.update()

        if output_type != "latent":
            if enable_denormalization:
                latents = self.denormalize_latents(latents)
            latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = latents

        # (B, C, 1, H, W) -> (B, C, H, W)
        if image.ndim == 5:
            image = image.squeeze(2)

        image = image.float()
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return JoyImagePipelineOutput(images=image)


class JoyImageDMDPipeline(JoyImagePipeline):
    """JoyImage DMD pipeline.

    DMD / consistency distillation models use the same pipeline with
    sampling_method="consistency_sampling" and guidance_scale<=1.
    This class exists as a named alias so that model_index.json can
    reference it directly, with DMD-appropriate defaults.
    """

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        kwargs.setdefault("sampling_method", "consistency_sampling")
        kwargs.setdefault("guidance_scale", 1.0)
        kwargs.setdefault("num_inference_steps", 8)
        return super().__call__(*args, **kwargs)
