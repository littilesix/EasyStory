import os
import safetensors.torch as sf
import torch
import traceback
import einops
from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import numpy as np
from diffusers.utils import load_image as load_PIL
from typing import Union
import uuid
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
def load_image(image: Union[str, Image.Image]):
    # 使用Pillow库加载图像
    image = load_PIL(image)
    image = np.array(image)
    return image
    
class FramePackEngine:

    basename = f"videos/video_{uuid.uuid4()}"

    def __init__(self,outputs:str = None,name = None):
        if name:
            self.basename = name
        self.tid = outputs
        #logger.info("GPU Device Name:",torch.cuda.get_device_name(0))  # 显示 GPU 名称

        free_mem_gb = get_cuda_free_memory_gb(gpu)

        self.high_vram = free_mem_gb > 60

        logger.info(f'Free VRAM {free_mem_gb} GB')

        logger.info(f'High-VRAM Mode: {self.high_vram}')

        self.text_encoder = LlamaModel.from_pretrained("./models/video", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        self.text_encoder_2 = CLIPTextModel.from_pretrained("./models/video", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        self.tokenizer = LlamaTokenizerFast.from_pretrained("./models/video", subfolder='tokenizer')
        self.tokenizer_2 = CLIPTokenizer.from_pretrained("./models/video", subfolder='tokenizer_2')
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained("./models/video", subfolder='vae', torch_dtype=torch.float16).cpu()
        self.feature_extractor = SiglipImageProcessor.from_pretrained("./models/video", subfolder='feature_extractor')
        self.image_encoder = SiglipVisionModel.from_pretrained("./models/video", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
        #self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(f'{self.custom_cache_dir}\\models--lllyasviel--FramePackI2V_HY\\snapshots\\86cef4396041b6002c957852daac4c91aaa47c79', torch_dtype=torch.bfloat16).cpu()
        self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("./models/video",subfolder='transformer', torch_dtype=torch.bfloat16).cpu()

        self.text_encoder.eval() #切换到评估模式
        self.text_encoder_2.eval()
        self.image_encoder.eval()
        self.transformer.eval()

        if not self.high_vram:
        	#VAE（变分自编码器）
        	#切片（slicing） 和 平铺（tiling） 都是为了在显存不足时降低每次计算的内存占用，以便能够处理更大的模型或更大的输入数据。它们是优化内存使用的技术，特别适合显存不足的情况下使用。
            self.vae.enable_slicing()
            self.vae.enable_tiling()

        # 设置 transformer 模型在推理过程中使用高精度的 FP32 输出
        self.transformer.high_quality_fp32_output_for_inference = True
        logger.info('transformer.high_quality_fp32_output_for_inference = True')

        # 将 transformer 转换为 bfloat16 类型（减少显存占用），用于推理时
        self.transformer.to(dtype=torch.bfloat16)

        # 将其他模型（vae、image_encoder、text_encoder 等）转换为 float16 类型
        # 这样做的目的是减少显存占用，并且 float16 精度对推理来说通常足够
        #self.vae.to(dtype=torch.float16)
        #self.image_encoder.to(dtype=torch.float16)
        #self.text_encoder.to(dtype=torch.float16)
        #self.text_encoder_2.to(dtype=torch.float16)

        # 禁用模型的梯度计算，表示不在反向传播中计算梯度
        # 这样可以节省显存和计算资源，特别是当模型只用于推理时
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # 如果显存不足（high_vram 为 False），则使用 DynamicSwapInstaller 来优化内存管理
        # DynamicSwapInstaller 是一个更高效的内存优化方法，比 Huggingface 的 enable_sequential_offload 快 3 倍
        if not self.high_vram:
            # 安装 transformer 和 text_encoder 模型，使其可以根据需要动态交换到 GPU 或 CPU
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
        else:
            # 如果显存充足（high_vram 为 True），则将所有模型加载到 GPU 上
            self.text_encoder.to(gpu)
            self.text_encoder_2.to(gpu)
            self.image_encoder.to(gpu)
            self.vae.to(gpu)
            self.transformer.to(gpu)

        # 创建一个异步流，用于流式处理输出（可能用于展示或处理数据）
        #self.stream = AsyncStream()

        # 设置输出文件夹路径，并确保该文件夹存在
        self.outputs_folder = outputs or './outputs'
        os.makedirs(self.outputs_folder, exist_ok=True)

    @torch.no_grad()
    def generate(self,input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache):
        video_list = []
        temp = f'result_{uuid.uuid4()}'
        # 计算生成的视频总帧数（基于视频长度和每帧的窗口大小）
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)  # 假设 30fps，每帧的时间窗口为 latent_window_size * 4
        total_latent_sections = int(max(round(total_latent_sections), 1))  # 确保至少有一部分

        # 为任务生成一个唯一的 self.tid
        #self.tid = generate_timestamp()

        logger.info(f"{self.tid}：Task Starting")
        # 初始输出进度，标记为 'Starting ...'
        #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

        try:
            # 如果不使用高显存模式，清除已加载的模型以节省内存
            if not self.high_vram:
                unload_complete_models(self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, self.transformer)

            # 文本编码处理
            #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
            logger.info(f"{self.tid}:Text encoding ...")

            if not self.high_vram:
                # 假设我们只进行一次文本编码，所以这两步操作不会很耗时
                fake_diffusers_current_device(self.text_encoder, gpu)  # 移动模型到指定的GPU设备
                load_model_as_complete(self.text_encoder_2, target_device=gpu)

            # 对正向（prompt）和负向（n_prompt）文本进行编码
            llama_vec, clip_l_pooler = encode_prompt_conds(prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            # 如果 cfg 设置为 1，忽略负向提示，否则进行负向提示的编码
            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2)

            # 填充或裁剪文本编码，确保长度为 512
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # 图像预处理
            #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
            logger.info(f"{self.tid}:Image processing ...")

            # 获取输入图像的尺寸并进行调整
            #logger.info(type(input_image))
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            # 保存原始输入图像以便后续使用
            #Image.fromarray(input_image_np).save(os.path.join(self.outputs_folder, f'{self.basename or temp}_video.png'))

            # 将图像转化为 PyTorch tensor，并进行归一化处理
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE 编码
            #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
            logger.info(f"{self.tid}:VAE encoding ...")

            # 如果不使用高显存模式，则加载 VAE 模型
            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, self.vae)

            # 使用 CLIP 进行图像特征提取
            #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            logger.info(f"{self.tid}:CLIP Vision encoding ...")

            # 如果不使用高显存模式，加载图像编码器
            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)

            # 使用 CLIP 进行图像编码
            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # 转换数据类型，以确保一致性
            llama_vec = llama_vec.to(self.transformer.dtype)
            llama_vec_n = llama_vec_n.to(self.transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)

            # 进行采样过程
            #self.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
            logger.info(f"{self.tid}:Start sampling ...")

            # 初始化随机种子和帧数
            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3  # 每帧的数量

            # 初始化历史潜在空间
            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            # 设置 latent_paddings，这用于处理不同的时间段
            latent_paddings = reversed(range(total_latent_sections))

            # 对于 > 4 的情况，调整 latent_paddings 序列，使其效果更好
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

            # 循环生成每一部分的 latent
            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                # 如果用户要求终止任务，则退出
                #if self.stream.input_queue.top() == 'end':
                    #self.stream.output_queue.push(('end', None))
                    #return

                logger.info(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

                # 设置 latent 索引
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # 拼接 clean_latents
                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                # 移动模型到 GPU
                if not self.high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                # 初始化 teaCache（如果启用）
                if use_teacache:
                    self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    self.transformer.initialize_teacache(enable_teacache=False)
                # 调用 Hunyuan 模型进行采样生成
                generated_latents = sample_hunyuan(
                    transformer=self.transformer,  # transformer 网络
                    sampler='unipc',  # 采样器类型
                    width=width,  # 输出图像的宽度
                    height=height,  # 输出图像的高度
                    frames=num_frames,  # 生成的帧数
                    real_guidance_scale=cfg,  # 真实的引导尺度
                    distilled_guidance_scale=gs,  # 精炼的引导尺度
                    guidance_rescale=rs,  # 引导重新调整比例
                    num_inference_steps=steps,  # 推理步骤数
                    generator=rnd,  # 随机数生成器
                    prompt_embeds=llama_vec,  # 正向提示嵌入向量
                    prompt_embeds_mask=llama_attention_mask,  # 正向提示嵌入掩码
                    prompt_poolers=clip_l_pooler,  # 提示池化器
                    negative_prompt_embeds=llama_vec_n,  # 负向提示嵌入向量
                    negative_prompt_embeds_mask=llama_attention_mask_n,  # 负向提示嵌入掩码
                    negative_prompt_poolers=clip_l_pooler_n,  # 负向提示池化器
                    device=gpu,  # 运行设备（GPU）
                    dtype=torch.bfloat16,  # 数据类型（BFloat16）
                    image_embeddings=image_encoder_last_hidden_state,  # 图像编码器的最后隐藏状态
                    latent_indices=latent_indices,  # 潜在变量的索引
                    clean_latents=clean_latents,  # 清理后的潜在变量
                    clean_latent_indices=clean_latent_indices,  # 清理后的潜在变量索引
                    clean_latents_2x=clean_latents_2x,  # 双倍清理后的潜在变量
                    clean_latent_2x_indices=clean_latent_2x_indices,  # 双倍清理后的潜在变量索引
                    clean_latents_4x=clean_latents_4x,  # 四倍清理后的潜在变量
                    clean_latent_4x_indices=clean_latent_4x_indices,  # 四倍清理后的潜在变量索引
                    #callback=self.callback,  # 回调函数，用于实时更新生成进度
                )

                # 如果是最后一个部分，将开始潜在变量与生成的潜在变量拼接
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                # 更新生成的总潜在帧数
                total_generated_latent_frames += int(generated_latents.shape[2])

                # 将当前生成的潜在帧拼接到历史潜在帧中
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                # 如果使用低内存模式，进行内存保存操作
                if not self.high_vram:
                    offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=8)
                    load_model_as_complete(self.vae, target_device=gpu)

                # 处理历史潜在变量并生成图像
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                # 如果还没有像素数据，使用 VAE 解码器从潜在变量生成像素数据
                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, self.vae).cpu()
                else:
                    # 如果已经有像素数据，处理并合并新的像素数据
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    # 获取当前生成的像素帧
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], self.vae).cpu()
                    # 将当前生成的像素帧合并到历史像素数据中
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                # 如果使用低内存模式，卸载已完成的模型以节省内存
                if not self.high_vram:
                    unload_complete_models()

                # 设置输出文件名，并保存为 MP4 格式
                output_filename = os.path.join(self.outputs_folder, f'{self.basename or temp}_{total_generated_latent_frames}.mp4')

                video_list.append(output_filename)

                # 将像素数据保存为 MP4 文件
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30)

                logger.info(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                # 将生成的视频文件路径发送到输出队列
                #self.stream.output_queue.push(('file', output_filename))
                logger.info(f"{self.tid}:save video ...\n{output_filename}")

                # 如果是最后一个部分，结束生成过程
                if is_last_section:
                    break

            logger.info("video all clips finish")

        except:
            logger.info("break out...")
            # 捕获异常并打印详细的错误堆栈信息，帮助调试
            traceback.logger.info_exc()

            # 如果不是高内存模式，则卸载已加载的模型以释放内存
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder,  # 卸载文本编码器模型
                    self.text_encoder_2,  # 卸载第二个文本编码器模型
                    self.image_encoder,  # 卸载图像编码器模型
                    self.vae,  # 卸载变分自编码器（VAE）模型
                    self.transformer  # 卸载 Transformer 模型
                )

        # 返回，结束函数执行
        return video_list


    """
    def callback(self,d):

        preview = d['denoised']
        preview = vae_decode_fake(preview)

        # 将预览图像处理为正确的格式
        preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
        preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

        # 如果用户要求终止任务，则退出
        if self.stream.input_queue.top() == 'end':
            self.stream.output_queue.push(('end', None))
            raise KeyboardInterrupt('User ends the task.')

        # 更新进度条
        current_step = d['i'] + 1
        percentage = int(100.0 * current_step / self.steps)
        hint = f'Sampling {current_step}/{self.steps}'
        desc = f'Total generated frames: {int(max(0, self.total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (self.total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). The video is being extended now ...'
        #self.stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
        return
    """
    def getVideoClips(self,img,prompt,total_second=5,basename=None):
        if basename:
            self.basename = basename
        return self.generate(
            input_image=load_image(img),  # 输入图像，根据需要加载或处理
            prompt=prompt,  # 正向提示词嵌入，使用 LLAMA 模型的向量表示
            n_prompt="",  # 负向提示词嵌入，使用 LLAMA 模型的负向向量表示
            seed=31337,  # 随机种子，用于控制图像生成的随机性
            total_second_length=total_second,  # 总的生成时长，来自生成的潜在帧数，最多120秒
            latent_window_size=9,  # 潜在窗口大小，控制每次处理的帧数，最多33帧
            steps=25,  # 推理步数，决定生成过程中的迭代次数
            cfg=1,  # 真实引导比例，调节生成图像的指导强度1~32
            gs=10,  # 精炼引导比例，进一步调整生成的细节
            rs=0,  # 引导尺度，影响生成图像的风格和细节
            gpu_memory_preservation=6,  # 高内存模式标志，是否进行内存优化
            use_teacache=True  # 使用 teacache，保持为 True
            )

if __name__ == '__main__':
    video = FramePackEngine()
    video.getVideoClips("text.png","A chef rabbit is cooking with a soup spoon, stirring and tasting")