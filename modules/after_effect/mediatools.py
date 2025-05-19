import subprocess
import platform
import os
import logging
from tools import StreamToLogger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def LogSubprocess(result):
    if result.stdout:
        logger.info("[stdout]\n" + result.stdout)
    if result.stderr:
        logger.error("[stderr]\n" + result.stderr)

def convertToWav(m4a_file,output_dir=None,ffmpeg="ffmpeg"):
    # 检查输入文件是否存在
    if not os.path.exists(m4a_file):
        logger.info(f"Error: The file {m4a_file} does not exist.")
        return None

    # 设置输出目录和输出文件路径
    if output_dir is None:
        output_dir = os.path.dirname(m4a_file)  # 如果没有指定目录，就使用输入文件所在的目录
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(m4a_file))[0] + '.wav')

    # 调用ffmpeg进行转换
    try:
        # 使用subprocess调用ffmpeg来转换音频
        command = [
            ffmpeg,
            '-i', m4a_file,         # 输入文件
            '-acodec', 'pcm_s16le', # 使用pcm_s16le编码格式（常见的WAV格式编码）
            '-ar', '24000',         # 设置采样率为44100Hz（WAV文件的常用设置）
            '-ac', '2' ,'-y',
            output_file          # 设置音频通道数为2（立体声）
        ]

        # 运行命令并等待完成
        result = subprocess.run(
            command,
            #cwd = work_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
            )
        LogSubprocess(result)
        logger.info(f"Conversion successful! WAV file saved to {output_file}")
        return output_file

    except subprocess.CalledProcessError as e:
        logger.exception(f"Error during conversion: {e}")
        return None

def merge_audio_video(video_file, audio_file, output_file,ffmpeg="ffmpeg"):
    """
    合并音频和视频文件
    
    :param video_file: 输入的视频文件路径
    :param audio_file: 输入的音频文件路径
    :param output_file: 输出的合并后文件路径
    """
    try:
        # 判断系统平台是否需要授予执行权限
        if platform.system() in ['Darwin', 'Linux']:
            os.chmod(ffmpeg, 0o755)  # 授予执行权限

        # 调用subprocess来执行ffmpeg命令合并音视频
        command = [
            ffmpeg,
            '-i', video_file,   # 输入视频文件
            '-i', audio_file,   # 输入音频文件
            '-c:v', 'copy',     # 复制视频编码
            '-c:a', 'aac',      # 使用aac编码音频
            '-strict', 'experimental',  # 使ffmpeg允许实验性编码
            "-y",#覆盖
            output_file         # 输出文件路径
        ]
        
        result = subprocess.run(
            command,
            #cwd = work_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
            )
        LogSubprocess(result)
        logger.info(f"audio and video merged: {output_file}")
    except subprocess.CalledProcessError as e:
        logger.exception(f"something errors happened when merge audio and video：{e}")
    except Exception as e:
        logger.exception(f"error{e}")

def merge_videos(video_files, output_file,ffmpeg="ffmpeg"):
    """
    合并多个视频文件
    
    :param video_files: 输入的多个视频文件路径列表
    :param output_file: 输出的合并后文件路径
    """
    try:
        # 判断系统平台是否需要授予执行权限
        if platform.system() in ['Darwin', 'Linux']:
            os.chmod(ffmpeg, 0o755)  # 授予执行权限

        # 创建一个临时文件用于存储ffmpeg的输入文件列表
        with open("inputs.txt", "w") as file:
            for video_file in video_files:
                file.write(f"file '{video_file}'\n")
        
        # 调用subprocess执行ffmpeg命令来合并视频
        command = [
            ffmpeg,
            '-f', 'concat',         # 使用concat协议
            '-safe', '0',           # 允许使用绝对路径
            '-i', 'inputs.txt',     # 输入文件列表
            '-c:v', 'copy',         # 复制视频编码
            '-c:a', 'aac', "-y" ,       # 使用aac编码音频
            output_file             # 输出文件路径
        ]

        result = subprocess.run(
            command, 
            #cwd = work_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True)
        LogSubprocess(result)
        logger.info(f"videos merged success,here the path: {output_file}")

        # 删除临时文件
        os.remove("inputs.txt")
        
    except subprocess.CalledProcessError as e:
        logger.exception(f"something errors happened:merge video{e}")
    except Exception as e:
        logger.exception(f"errors{e}")

if __name__ == '__main__':
    from download import FFmpegDownloader
    ff = FFmpegDownloader("libs").run()
    convertToWav(
        r"D:\Projects\easyStory\气势磅礴音乐.m4a",
        "",
         ff
        )