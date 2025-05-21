import torch
from typing import Union,Optional,List,Callable,Any,Dict
from diffusers.pipelines.flux.pipeline_flux import calculate_shift,retrieve_timesteps
# torch disable grad
import numpy as np
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers import FluxPipeline,FluxTransformer2DModel,AutoencoderKL,FlowMatchEulerDiscreteScheduler,FluxImg2ImgPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import  CLIPTextModel,T5EncoderModel,CLIPTokenizer,T5TokenizerFast,CLIPTextConfig
from transformers.models.t5 import T5Config
from safetensors.torch import load_file
from diffusers.utils import load_image
from enum import Enum
from PIL import Image as PILImage
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class PipelineMode(Enum):
    ImageToImage = 0
    TextToImage = 1

class FluxEngine:
    def __init__(self,
        transformer="./models/flux/flux1-dev.safetensors",
        dtype=torch.bfloat16,
        ):
        self.dtype = dtype
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained('./models/flux/scheduler',torch_dtype=self.dtype)

        self.vae = AutoencoderKL.from_pretrained('./models/flux/vae', torch_dtype=self.dtype)
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.vae.requires_grad_(False)

        clip_config = CLIPTextConfig.from_pretrained('./models/flux/text_encoder', ignore_mismatched_sizes=True)
        self.text_encoder = CLIPTextModel(config=clip_config)
        self.text_encoder.load_state_dict(load_file("./models/flux/clip/clip_l.safetensors"), strict=False)
        self.text_encoder.cpu()
        self.text_encoder.to(dtype=self.dtype)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        T5_config = T5Config.from_pretrained('./models/flux/text_encoder_2',ignore_mismatched_sizes=True)
        self.text_encoder_2 = T5EncoderModel(config=T5_config)
        self.text_encoder_2.load_state_dict(load_file("./models/flux/t5/t5xxl_fp16.safetensors"), strict=False)
        self.text_encoder_2.cpu()
        self.text_encoder_2.to(dtype=self.dtype)
        self.text_encoder_2.eval()
        self.text_encoder_2.requires_grad_(False)

        DynamicSwapInstaller.install_model(self.text_encoder_2, device=gpu)
        
        self.tokenizer = CLIPTokenizer.from_pretrained('.\\models\\flux\\tokenizer',  torch_dtype=self.dtype)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained('.\\models\\flux\\tokenizer_2', torch_dtype=self.dtype)

        self.transformer = FluxTransformer2DModel.from_single_file(
            transformer,
            config = ".\\models\\flux",
            torch_dtype=self.dtype
            )
        self.transformer.cpu()
        self.transformer.eval()
        self.transformer.high_quality_fp32_output_for_inference = True
        self.transformer.requires_grad_(False)
        DynamicSwapInstaller.install_model(self.transformer, device=gpu)
        self.weight = None
        self.pipeline = None
        self.lora = None
        self.mode = None

    def __initImageToImagePipeline(self):
        self.vae.to(gpu)
        self.mode = PipelineMode.ImageToImage
        logger.info(self.mode)
        self.pipeline = FluxImg2ImgPipeline(
            scheduler = self.scheduler ,
            vae = self.vae,
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            text_encoder_2 = self.text_encoder_2,
            tokenizer_2 = self.tokenizer_2,
            transformer = self.transformer
        )

    def _initTxtToImagePipeline(self):
        self.__initTxtToImagePipeline()

    def __initTxtToImagePipeline(self):
        self.vae.cpu()
        self.mode = PipelineMode.TextToImage
        logger.info(self.mode)
        self.pipeline = FluxPipeline(
            scheduler = self.scheduler ,
            vae = self.vae,
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            text_encoder_2 = self.text_encoder_2,
            tokenizer_2 = self.tokenizer_2,
            transformer = self.transformer
        )

    def load_lora(self,path,weight=1):
        if not self.weight:
            self.weight = weight
            logger.info(f"LORA weight is {self.weight}")
        self.lora = path
        #self.pipeline.load_lora_weights(path)

    def unload_lora(self):
        if self.pipeline:
            self.pipeline.unload_lora_weights()
            self.lora = None

    def getImageFromText(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        strength = 0.8
        ):
        if self.mode != PipelineMode.TextToImage or self.pipeline == None:
            self.__initTxtToImagePipeline()
        if self.lora:
            self.pipeline.load_lora_weights(self.lora)
            self.lora = None
        images = self.run(
            prompt=prompt,
            height=height,
            width=width,num_inference_steps=num_inference_steps,
            joint_attention_kwargs={"scale":self.weight}
            ).images
        return images[0]

    def getImageFromImage(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: int = 28,
        image = None,
        strength = 0.6
        ):
        if self.mode != PipelineMode.ImageToImage or self.pipeline == None:
            self.__initImageToImagePipeline()
        if self.lora:
            self.pipeline.load_lora_weights(self.lora)
            self.lora = None
        inputImage = load_image(image)
        images = self.runi2i(
            image=inputImage,
            strength=strength,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            joint_attention_kwargs={"scale":self.weight}
            ).images
        return images[0]

    @torch.no_grad()
    def run(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        logger.info("runing")

        height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
  
        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._joint_attention_kwargs = joint_attention_kwargs
        self.pipeline._current_timestep = None
        self.pipeline._interrupt = False

        fake_diffusers_current_device(self.pipeline.text_encoder_2, gpu)
        load_model_as_complete(self.pipeline.text_encoder, target_device=gpu)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        #device = self.pipeline.text_encoder.device

        lora_scale = (
            self.pipeline.joint_attention_kwargs.get("scale", None) if self.pipeline.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.pipeline.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )


        unload_complete_models(self.pipeline.text_encoder,self.pipeline.text_encoder_2)
        move_model_to_device_with_memory_preservation(self.pipeline.transformer,target_device=gpu, preserved_memory_gb=4)

        # 4. Prepare latent variables
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4
        latents, latent_image_ids = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)
        self.pipeline._num_timesteps = len(timesteps)

        # handle guidance
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=self.dtype)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.pipeline.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.pipeline.transformer.encoder_hid_proj.num_ip_adapters

        if self.pipeline.joint_attention_kwargs is None:
            self.pipeline._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline.interrupt:
                    continue

                self.pipeline._current_timestep = t
                if image_embeds is not None:
                    self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.pipeline.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self.pipeline._current_timestep = None

        load_model_as_complete(self.pipeline.vae,target_device=gpu)

        if output_type == "latent":
            image = latents
        else:
            latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
            latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
            image = self.pipeline.vae.decode(latents, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.pipeline.maybe_free_model_hooks()
        #unload_complete_models()
        logger.info("image create finish,ready to save")
        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


    @torch.no_grad()
    def runi2i(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        logger.info("runing")

        height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt,
            prompt_2,
            strength,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._joint_attention_kwargs = joint_attention_kwargs
        self.pipeline._interrupt = False

        # 2. Preprocess image

        init_image = self.pipeline.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=self.dtype)

        fake_diffusers_current_device(self.pipeline.text_encoder_2, gpu)
        load_model_as_complete(self.pipeline.text_encoder, target_device=gpu)
        self.pipeline.vae.to(gpu)

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        lora_scale = (
            self.pipeline.joint_attention_kwargs.get("scale", None) if self.pipeline.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.pipeline.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        unload_complete_models(self.pipeline.text_encoder,self.pipeline.text_encoder_2)
        fake_diffusers_current_device(self.pipeline.vae, gpu)
        move_model_to_device_with_memory_preservation(self.pipeline.transformer,target_device=gpu, preserved_memory_gb=4)

        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.pipeline.vae_scale_factor // 2) * (int(width) // self.pipeline.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.pipeline.scheduler.config.get("base_image_seq_len", 256),
            self.pipeline.scheduler.config.get("max_image_seq_len", 4096),
            self.pipeline.scheduler.config.get("base_shift", 0.5),
            self.pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.pipeline.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.transformer.config.in_channels // 4

        latents, latent_image_ids = self.pipeline.prepare_latents(
            init_image,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        #unload_complete_models(self.pipeline.vae)

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)
        self.pipeline._num_timesteps = len(timesteps)

        # handle guidance
        if self.pipeline.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=self.dtype)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.pipeline.joint_attention_kwargs is None:
            self.pipeline._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline.interrupt:
                    continue

                if image_embeds is not None:
                    self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.pipeline.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self.pipeline._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.pipeline.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self.pipeline, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        load_model_as_complete(self.pipeline.vae, target_device=gpu)

        if output_type == "latent":
            image = latents

        else:
            latents = self.pipeline._unpack_latents(latents, height, width, self.pipeline.vae_scale_factor)
            latents = (latents / self.pipeline.vae.config.scaling_factor) + self.pipeline.vae.config.shift_factor
            image = self.pipeline.vae.decode(latents, return_dict=False)[0]
            image = self.pipeline.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.pipeline.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

if __name__ == '__main__':
    #from utils import testImageGen
    flux = FluxEngine()
    #flux.load_lora('outputs/roles/Doudou/lora.safetensors')
    image = flux.run(prompt="A cute rabbit that can talk, a bit shy but very friendly,short,fat",width=600,height=800,ip_adapter_image ="")
    image.save(f"text.png")
    #flux.unload_lora()
    #image = flux.getImageFromImage(image =image,prompt="A cute rabbit that can talk, a bit shy but very friendly,short,fat",width=600,height=800)
    #image.save(f"image_to_image.png")

