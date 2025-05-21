from tools import read_text,FakeModules
import shutil
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import sys,os
from huggingface_hub import hf_hub_download,HfApi
import time
import re
import zipfile
import ollama

os.makedirs("installer/cache",exist_ok=True)
os.makedirs("models",exist_ok=True)
print(f'platform: {sys.platform}')
from typing import Tuple, List, Optional,Dict,Union
from tqdm import tqdm

def unzip_to(zip_path, target_dir):
    if not os.path.exists(zip_path):
        print(f"not zip found in {zip_path}")
        return
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    if model_dir is None:  # use the pytorch hub_dir
        model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        #print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def getUrls(path):
    data = {}
    lines = read_text(path).splitlines()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        info = line.split(":")
        key = info[0].strip()
        if not key in data:
            data[key] = []
        data[key].append(":".join(info[1:]).strip())
    return data

# Define constants for data types
REPO = "repo"
URL = "url"
UNKNOWN = "unknown"

class Downer:
    def __init__(self, data: Dict[str, list]):
        self.data = data
        self.token = "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"
        try:
            self.repoApi = HfApi(token=self.token)
            self.repoApi.list_repo_files("black-forest-labs/FLUX.1-dev")
        except:
            self.repoApi = HfApi(endpoint="https://hf-mirror.com",token=self.token)

    @staticmethod
    def data_type(data: str) -> str:
        if re.match(r'^https?://', data):
            return URL
        elif re.match(r'^[\w\-\.]+/[\w\-\.]+(:([\w\-\./]+\.[\w]+)(,[\w\-\./]+\.[\w]+)*)?$', data):
            return REPO
        else:
            return UNKNOWN

    def down(
        self, key: str,
        save_path: str,
        name: Optional[str] = None,
        file_names:Optional[List[str]] = None,
        move_destination:Optional[str] = None
        ) -> bool:
        """Attempt to download the data using URL, repo ID, or repo subfolder."""
        status = False
        down_datas = self.data.get(key)
        file_name = name if name else key

        if not down_datas:
            print(f"❌ No data found for '{key}'")
            return False

        print(f"======={key} begin download ===========")
        for i, data in enumerate(down_datas):
            print(f"🔽 Downloading {key} [{i}]: {data}")
            data_type = Downer.data_type(data)

            if data_type == UNKNOWN:
                print(f"⚠️ Skipping invalid data: {data}")
                continue

            elif data_type == URL:
                try:
                    load_file_from_url(data, save_path, file_name=file_name)
                    status = True
                    break
                except Exception as e:
                    print(f"❌ URL [{i}] for '{key}' failed: {e}")
                    continue

            elif data_type == REPO:
                try:
                    repo_id,files = Downer.parse_repo_and_files(data)
                    self.smart_Hf_down(repo_id,save_dir=save_path,file_names=files if files else file_names,move_destination=move_destination)
                    status = True
                    break
                except Exception as e:
                    print(f"❌ repo_id [{i}] for '{key}' failed: {e}")
                    continue
        if status:
            print(f"✅ {key} successfully saved to {save_path}")
        else:
            print(f"❌ All download attempts failed for {key}. Please download manually.")
        print("=======================================")
        return status

    @staticmethod
    def parse_repo_and_files(data: str) -> Tuple[str, Optional[List[str]]]:
        """
        解析形如 'user/repo:file1.ext,file2.ext' 的字符串
        返回 repo_id 和文件列表
        """
        pattern = r'^([\w\-]+/[\w\-\.]+)(?::([\w\-\./]+\.[\w]+(?:,[\w\-\./]+\.[\w]+)*))?$'
        match = re.fullmatch(pattern, data)
        if not match:
            print(f"❌ invalid repo data: '{data}'，should be 'user/repo[:file1.ext,file2.ext,...]'")

        repo_id = match.group(1)
        files_str = match.group(2)

        files = files_str.split(',') if files_str else None
        return repo_id, files

    def smart_Hf_down(
        self,
        repo_id: str,
        save_dir: str, 
        file_names:Optional[List[str]] = None, 
        endpoints=["https://huggingface.co","https://hf-mirror.com"],
        move_destination:Optional[List[str]] = None
        ):
        os.makedirs(save_dir, exist_ok=True)
        all_files = self.get_repo_files(repo_id)
        print(f'This repo contains:{all_files}')
        if file_names:
            matched_files = [f for f in file_names if f in all_files]
        else:
            matched_files = all_files
        if not matched_files:
            raise ValueError(f"❌ not match file: {file_names or 'all'}")

        print(f"📦 Total files to download: {len(matched_files)}")
        success_count = 0
        failed_files = []

        print(matched_files)
        # tqdm 进度条
        for file in tqdm(matched_files, desc="📥 Downloading files", unit="file"):
            if "/" in file:
                subfolder,file = file.split("/")
            else:
                subfolder,file = None,file
                #os.makedirs(os.path.join(save_dir, file.split("/")[0]),exist_ok=True)
            for endpoint in endpoints:
                try:
                    des_file = os.path.join(move_destination,file) if move_destination else None
                    save_file = os.path.join(save_dir,subfolder,file) if subfolder else os.path.join(save_dir,file)
                    #print(des_file,save_file)
                    if (des_file and os.path.exists(des_file)) or os.path.exists(save_file):
                        print(f"✅ {des_file or save_file} already exists")
                        pass
                    else:
                        path = hf_hub_download(repo_id,file,endpoint=endpoint,subfolder=subfolder,token=self.token,local_dir=save_dir,force_download=True)
                        if move_destination:
                            os.makedirs(move_destination,exist_ok=True)
                            shutil.move(path,des_file)
                            if subfolder:
                                shutil.rmtree(os.path.join(save_dir, subfolder))
                    success_count += 1
                    break
                except Exception as e:
                    #print(f'\n{e}')
                    continue
            else:
                failed_files.append(file)
                tqdm.write(f"❌ Failed to download: {file}")

        # 结果统计
        print(f"\n✅ Success: {success_count}/{len(matched_files)}")
        if failed_files:
            print(f"❌ Failed ({len(failed_files)}):")
            for f in failed_files:
                print(f" - {f}")
            raise ValueError(f"not all files downloaded")

    def get_repo_files(self,repo_id):
        return self.repoApi.list_repo_files(repo_id)

    def snapshot_download():
        self.repoApi.snapshot_download()

if __name__ == '__main__':


    fish_speech_files =  [
        "tokenizer.tiktoken",
        "config.json",
        "model.pth",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
      ]

    flux_files =  [
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        'text_encoder_2/config.json',
        'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json',
        'tokenizer_2/special_tokens_map.json', 'tokenizer_2/spiece.model', 'tokenizer_2/tokenizer.json', 'tokenizer_2/tokenizer_config.json', 
        "flux1-dev.safetensors","model_index.json",
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors'
      ]

    video_files = [
        'text_encoder/config.json', 'text_encoder/model-00001-of-00004.safetensors', 'text_encoder/model-00002-of-00004.safetensors', 'text_encoder/model-00003-of-00004.safetensors', 'text_encoder/model-00004-of-00004.safetensors', 'text_encoder/model.safetensors.index.json', 
        'text_encoder_2/config.json','text_encoder_2/model.safetensors',
        'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 
        'tokenizer_2/merges.txt', 'tokenizer_2/special_tokens_map.json', 'tokenizer_2/tokenizer_config.json', 'tokenizer_2/vocab.json', 
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors'
        ]

    models = Downer(getUrls("installer/models_data_for_win.txt"))

    ollama_bin = shutil.which("ollama")
    ffmpeg_bin = shutil.which("ffmpeg")

    if ollama_bin or os.path.exists("libs/ollama/ollama.exe"):
        print(f"✅ollama is already exists in {ollama_bin or 'libs/ollama/ollama.exe'}，skip the installer")
    else:
        if models.down("ollama",save_path = "installer/cache",name="ollama-windows-amd64.zip"):
            unzip_to("installer/cache/ollama-windows-amd64.zip","libs/ollama")

    from engines.ollama import OllamaEngine
    engine = OllamaEngine()
    if engine.isOpened:
        engine.close()
    else:
        if engine.client:
            print("waiting to pull deepseek-r1:14b until the \"success\"")
            pbar = None

            for progress in engine.client.pull("deepseek-r1:14b", stream=True):
                if progress.total and progress.completed is not None:
                    if pbar is None:
                        pbar = tqdm(total=progress.total, unit='B', unit_scale=True, desc=progress.status)
                    pbar.update(progress.completed - pbar.n)
                    pbar.set_description(progress.status)
                else:
                    # 状态可能是非文件进度阶段（比如 pulling manifest）
                    print(f"{progress.status}...")
            if pbar:
                pbar.close()
            print("✅ download model deepseek-r1:14b \"success\"")
            engine.close()
        else:
            print("❌download model deepseek-r1:14b \"fail\"")

    if ffmpeg_bin or os.path.exists("libs/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"):
        print(f"✅ffmpeg is already exists in {ffmpeg_bin or 'libs/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe'}，skip the installer")
    else:
        if models.down("ffmpeg",save_path = "installer/cache",name="ffmpeg.zip"):
            unzip_to("installer/cache/ffmpeg.zip","libs")

    if models.down("lora",save_path = "installer/cache"):
        unzip_to("installer/cache/lora.zip","models/hub")

    models.down("font",save_path = "libs/fonts")
    models.down("voice",save_path = "libs/voices")

    models.down("fish_speech", save_path = "models/fish-speech-1.5",file_names=fish_speech_files)

    models.down("flux_text_encoder1", save_path = "models/flux",move_destination="models/flux/clip")
    models.down("flux_text_encoder2", save_path = "models/flux",move_destination="models/flux/t5")
    models.down("flux_vae", save_path = "models/flux",move_destination="models/flux/vae")
    models.down("flux", save_path = "models/flux",file_names = flux_files)

    models.down("hunyuan",save_path = "models/video",file_names=video_files)
    models.down("video_image_encoder",save_path = "models/video")
    models.down("video_transformer",save_path = "models/video/transformer")

    models.down("scaler",save_path = "models/scale")