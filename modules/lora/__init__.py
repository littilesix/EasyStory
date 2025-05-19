import sys,os
import subprocess
from typing import List, Union

dataset = """[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 512
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{0}'
  class_tokens = '{1}'
  num_repeats = 10"""

class LoraConfig:
    def __init__(self, image_dir,class_tokens,captions: Union[List[str], str]):
        self.image_dir = image_dir
        self.class_tokens = class_tokens
        self.output_dir = image_dir
        self.captions = captions
        self.generate_caption_files()

    def generate_caption_files(self):
        # 获取output_dir下所有图片
        image_files = [
            f for f in os.listdir(self.output_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        ]

        if not image_files:
            raise ValueError(f"目录 {self.output_dir} 中没有找到任何图片文件！")

        image_files.sort()  # 排序，保证稳定性

        # 判断 captions 是 List 还是 单个 str
        if isinstance(self.captions, list):
            # 如果是list，数量要一致
            if len(image_files) != len(self.captions):
                raise ValueError(f"图片数量（{len(image_files)}）和 captions 数量（{len(self.captions)}）不一致！")

            captions_to_use = self.captions
        elif isinstance(self.captions, str):
            # 如果是单个字符串，扩展成一样数量的列表
            captions_to_use = [self.captions] * len(image_files)
        else:
            raise TypeError("captions 必须是 List[str] 或 str 类型")

        # 写入txt文件
        for img_file, caption in zip(image_files, captions_to_use):
            name_without_ext = os.path.splitext(img_file)[0]
            caption_path = os.path.join(self.output_dir, name_without_ext + ".txt")
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

        print(f"成功生成了 {len(image_files)} 个 caption 文件。")


class Lora:
    def __init__(self, config: LoraConfig):
        # 类的属性
        self.work = os.getcwd()
        self.base_model = "./models/flux/flux1-dev.safetensors"
        self.clip = "./models/flux/clip/clip_l.safetensors"
        self.t5 = "./models/flux/t5/t5xxl_fp16.safetensors"
        self.vae = "./models/flux/vae/ae.sft"
        self.exe = "./modules/lora/sd-scripts/flux_train_network.py"
        self.num_cpu_threads_per_process = 4
        self.dataset_config = f"{config.output_dir}/dataset.toml"
        self.output_dir = config.output_dir
        self.class_tokens = config.class_tokens
        self.image_dir = config.image_dir
        self.argvs = self.generate_script()
        with open(self.dataset_config,"w+",encoding="utf-8") as f:
          f.write(dataset.format(config.image_dir,config.class_tokens))

    def generate_script(self):
        # 使用类的属性直接生成命令
        script_list = [
            "env/scripts/accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", str(self.num_cpu_threads_per_process),
            self.exe,
            "--pretrained_model_name_or_path", self.base_model,
            "--clip_l", self.clip,""
            "--t5xxl", self.t5,
            "--ae", self.vae,
            "--cache_latents_to_disk",
            "--save_model_as", "safetensors",
            "--sdpa", "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--seed", "42",
            "--gradient_checkpointing",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--network_module", "networks.lora_flux",
            "--network_dim", "4",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
            "--lr_scheduler", "constant_with_warmup",
            "--max_grad_norm", "0.0",
            "--learning_rate", "8e-4",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--fp8_base",
            "--highvram",
            "--max_train_epochs", "16",
            "--save_every_n_epochs", "4",
            "--dataset_config", self.dataset_config,
            "--output_dir", self.output_dir,
            "--output_name", "lora",
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.1582",
            "--model_prediction_type", "raw",
            "--guidance_scale", "1",
            "--loss_type", "l2"
        ]

        # 返回最终生成的脚本列表
        return script_list

    def run(self):
        hf_home = os.environ.copy()
        hf_home["HF_HOME"] = "./models"

        #hf_home["PATH"] = f"{sys.path[0]};{hf_home['PATH']}"
        #print(hf_home)
        status = False
        try:
            # 注意：可以设定 working directory
            result = subprocess.run(
                self.argvs,
                #["./.env/Scripts/accelerate.exe"],
                cwd = self.work,
                env = hf_home,
                check=True,
                text=True,
            )
            status = True
            print("Training finished successfully.")
        except subprocess.CalledProcessError as e:
            print("Training failed.")
            print("Error output:", e)
        return status
# 调用示例
if __name__ == '__main__':
    import os
    lora = Lora(LoraConfig("./outputs/roles/Beans","Cotton_Candy_Rabbit","A cute rabbit that can talk, a bit shy but very friendly."))
    lora.run()