import inspect
from typing import Callable, List, Optional, Union

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...models import AutoencoderKLJoyVideo, JoyVideoTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import JoyVideoPipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
Examples:
    ```python
    >>> import torch
    >>> from diffusers import JoyVideoPipeline
    >>> from diffusers.utils import export_to_video

    >>> pipe = JoyVideoPipeline.from_pretrained("path/to/joyvideo", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> prompt = "A cat walking on the beach at sunset."
    >>> output = pipe(
    ...     prompt=prompt,
    ...     height=480,
    ...     width=832,
    ...     num_frames=121,
    ...     num_inference_steps=35,
    ...     guidance_scale=3.5,
    ...     generator=torch.manual_seed(1024),
    ... ).frames[0]
    >>> export_to_video(output, "output.mp4", fps=24)
    ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
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


class JoyVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using JoyVideo.

    Model offloading order: text_encoder -> transformer -> vae.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLJoyVideo,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: JoyVideoTransformer3DModel,
        text_token_max_length: int = 768,
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
        if getattr(self, "vae", None) is not None:
            self.vae_scale_factor_temporal = getattr(self.vae, "scale_factor_temporal", 8)
            self.vae_scale_factor_spatial = getattr(self.vae, "scale_factor_spatial", 16)
        else:
            self.vae_scale_factor_temporal = 8
            self.vae_scale_factor_spatial = 16
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.prompt_template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_drop_idx = 91

    def _get_last_decoder_hidden_states(self, forward_fn, **kwargs):
        captured = {}

        def _hook(_module, _input, output):
            captured["hidden_states"] = output[0] if isinstance(output, tuple) else output

        handle = self.text_encoder.model.layers[-1].register_forward_hook(_hook)
        try:
            forward_fn(**kwargs)
        finally:
            handle.remove()
        return captured["hidden_states"]

    def _extract_masked_hidden(self, hidden_states, mask):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompt,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        drop_idx = self.prompt_template_drop_idx
        max_seq_len = self.text_token_max_length

        txt = [self.prompt_template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=max_seq_len + drop_idx,
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

        actual_max_len = min(
            max_seq_len,
            max(u.size(0) for u in split_hidden_states),
        )
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(actual_max_len - u.size(0), u.size(1))])
                if u.size(0) < actual_max_len
                else u[:actual_max_len]
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(actual_max_len - u.size(0))])
                if u.size(0) < actual_max_len
                else u[:actual_max_len]
                for u in attn_mask_list
            ]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt,
        device=None,
        num_videos_per_prompt: int = 1,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        max_sequence_length: int = 768,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_videos_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
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

    def normalize_latents(self, latent):
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        return latent

    def denormalize_latents(self, latent):
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(device=latent.device, dtype=latent.dtype)
            )
            latent = latent * latents_std + latents_mean
        return latent

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("Generator list length must match batch size.")

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 121,
        num_inference_steps: int = 35,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable, PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 768,
    ):
        r"""
        Generate a video conditioned on a text prompt.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s) used to suppress undesired content.
            height (`int`, defaults to 480):
                Height of the generated video in pixels.
            width (`int`, defaults to 832):
                Width of the generated video in pixels.
            num_frames (`int`, defaults to 121):
                Number of frames in the generated video.
            num_inference_steps (`int`, defaults to 35):
                Number of denoising steps.
            guidance_scale (`float`, defaults to 3.5):
                Classifier-free guidance scale.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos per prompt.
            generator (`torch.Generator`, *optional*):
                RNG generator(s) for deterministic sampling.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to ``"np"``):
                Output format.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a pipeline output object or a plain tuple.
            max_sequence_length (`int`, defaults to 768):
                Maximum text sequence length.

        Examples:

        Returns:
            [`~JoyVideoPipelineOutput`] or `tuple`.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            if negative_prompt is None and negative_prompt_embeds is None:
                negative_prompt = [""] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                max_sequence_length=max_sequence_length,
            )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = latents.to(transformer_dtype)
                t_expand = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_expand,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            latents = self.denormalize_latents(latents)
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return JoyVideoPipelineOutput(frames=video)
