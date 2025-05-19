import os
import platform
import requests
import zipfile
import tarfile
from tqdm import tqdm
import subprocess
import shutil

class FFmpegDownloader:
    def __init__(self, destination):
        self.destination = destination

    def is_ffmpeg_available(self):
        """检查系统环境变量中是否已存在 ffmpeg，或指定目录下的 ffmpeg"""
        # 1. 检查系统环境变量中是否已存在 ffmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"ffmpeg is already available in system PATH: {ffmpeg_path}")
            return ffmpeg_path

        ffmpeg_dirs = [os.path.join(self.destination, file) for file in os.listdir(self.destination) if "ffmpeg" in file]

        ffmpeg_dirs = [file for file in ffmpeg_dirs if os.path.isdir(file)]

        if ffmpeg_dirs.__len__() == 0: return None

        # 3. 如果没有直接的 'ffmpeg'，检查是否有类似 'bin' 文件夹，并尝试找到可执行文件
        for root, dirs, files in os.walk(ffmpeg_dirs[0]):
            for file in files:
                if platform.system() == "Windows" and file.lower() == "ffmpeg.exe":
                    print(f"Found executable ffmpeg: {os.path.join(root, file)}")
                    return os.path.join(root, file)
                elif platform.system() != "Windows" and file.lower() == "ffmpeg":
                    print(f"Found executable ffmpeg: {os.path.join(root, file)}")
                    return os.path.join(root, file)

        # 4. 如果都没有找到，返回 None
        print("ffmpeg not found.")
        return None

    def set_ffmpeg_url(self):
        """根据操作系统设置 ffmpeg 的下载地址"""
        system = platform.system()
        arch = platform.architecture()[0]

        if system == "Windows":
            self.ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        elif system == "Linux":
            if arch == "64bit":
                self.ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            else:
                self.ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz"
        elif system == "Darwin":  # macOS
            self.ffmpeg_url = "https://evermeet.cx/ffmpeg/ffmpeg-119416-g1dbc5675c1.7z"
        else:
            print("Unsupported operating system.")
            return None

    def download(self):
        """下载并提取 ffmpeg 文件"""
        self.ffmpeg_path = self.is_ffmpeg_available()
        if self.ffmpeg_path:
            print(f"ffmpeg is already available at: {self.ffmpeg_path}")
            return self.ffmpeg_path

        # 设置下载的 ffmpeg url
        self.set_ffmpeg_url()

        if not self.ffmpeg_url:
            print("Unsupported operating system. Cannot download ffmpeg.")
            return None
        
        print(f"Downloading ffmpeg from {self.ffmpeg_url}...")

        # 下载文件并显示进度
        response = requests.get(self.ffmpeg_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        # 创建目标文件夹
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)

        # 文件名
        filename = os.path.join(self.destination, self.ffmpeg_url.split("/")[-1])

        # 使用 tqdm 来显示进度条
        with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc="Downloading") as pbar:
            with open(filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
        
        print("Download completed!")

        # 解压文件
        self.extract(filename)

        # 返回ffmpeg的路径
        ffmpeg_bin = 'ffmpeg' if platform.system() == "Windows" else './ffmpeg'
        ffmpeg_path = os.path.join(self.destination, ffmpeg_bin)
        print(f"ffmpeg extracted to: {ffmpeg_path}")

        return ffmpeg_path

    def extract(self, filename):
        """解压下载的文件"""
        if filename.endswith(".zip"):
            print("Extracting zip file...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(self.destination)
            print("Extraction completed!")
        elif filename.endswith(".tar.xz"):
            print("Extracting tar.xz file...")
            with tarfile.open(filename, "r:xz") as tar_ref:
                tar_ref.extractall(self.destination)
            print("Extraction completed!")
        else:
            print("Unsupported file format for extraction.")

    def run(self):
        """下载并提取ffmpeg"""
        return self.download()

if __name__ == '__main__':
    downloader = FFmpegDownloader(destination="./libs")
    ffmpeg_path = downloader.run()
    print(ffmpeg_path)

