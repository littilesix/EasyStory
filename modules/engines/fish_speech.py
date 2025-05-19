import sys
import soundfile as sf
from storydata import StoryData
from typing import Union
import subprocess
import os
import librosa
import math
import torch
try:
    from fish_speech_lib.inference import FishSpeech
except:
    from tools import FakeModules
    from fish_speech_lib.inference import FishSpeech

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from backend import AudioBackend

from tools import proportional_integer_allocation

class FishSpeechEngine:

    def __init__(self, data:StoryData = None,voice:str = None):
        self.data = data
        self.voice = voice
        if self.data and self.data.has("shots"):
            self.shots = data.shots

        compile = False
        device = "cuda"
        checkpoint_path = "./models/fish-speech-1.5"

        if torch.backends.mps.is_available():
            device = "mps"
        elif not torch.cuda.is_available():
            device = "cpu"

        self.tts = FishSpeech(
            device = device,
            half = False,
            compile_model = False,
            llama_checkpoint_path  = "models/fish-speech-1.5",  # Пути по умолчанию
            decoder_checkpoint_path = "models/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            streaming = False,
        )

    @property
    def references(self):
        return self.get_reference_by_name(self.voice)

    def gen_all_shots_audio(self):
        os.makedirs(self.data.cache_dir,exist_ok=True)
        for shot in self.shots:
            #保存多条数据一致
            path = f"{self.data.cache_dir}/Shot_{shot.id}.wav"
            if os.path.exists(path):
                shot.audio = path
                continue
            self.genShotWav(shot)
        self.data.save(prefix="audiosave")
        self.tts.engine.llama_queue.put(None)

    def genShotWav(self,shot):
        path = f"{self.data.cache_dir}/Shot_{shot.id}.wav"
        self.genWavFile(shot.subtitle,path)
        shot.audio = path
        audo_info = sf._SoundFileInfo(path,verbose=False)
        shot.audio_time = audo_info.duration
        shot.time = math.ceil(shot.audio_time) + 1 #给视频画面留出一秒余量时间
        split_lengths = [len(scene.split_text) for scene in shot.scenes]
        int_durations = proportional_integer_allocation(weights=split_lengths,total_count=shot.time)
        for duration,scene in zip(int_durations,shot.scenes):
            scene.time = duration

    def genWavFile(self,text,path,after=False,play=False,noise_strength=0.2,top_db=20):
        reference_audio,reference_audio_text = self.references
        rate, data = self.tts(
            text=text,
            reference_audio=reference_audio,
            reference_audio_text=reference_audio_text,
            #max_new_tokens=2000
            )
        if after:
            # 简单降噪：预加重（可选）
            data = librosa.effects.preemphasis(data, coef=noise_strength)
            data, _ = librosa.effects.trim(data, top_db=top_db)

        logger.info(path)
        sf.write(path, data, rate)

        if play:
            AudioeGener.play_audio(path)

    @staticmethod
    def play_audio(path):
        try:
            if sys.platform == "darwin":
                command = ["afplay", path] 
            elif sys.platform == "win32":
                the_bin = ["curl","",""]
            else:
                the_bin = "sudo apt-get install sox&&sudo apt-get install sox libsox-fmt-all&&play"
            subprocess.run([the_bin, path], check=True)
        except subprocess.CalledProcessError as e:
            logger.info(f"Error playing audio: {e}")
        except FileNotFoundError:
            logger.info("afplay command not found. Make sure you are using macOS.")

    def get_reference_by_name(self,name):
        audio = f"./libs/voices/{name}.wav"
        text = f"./libs/voices/{name}.text"
        if not os.path.exists(audio):
            raise FileNotFoundError(f"voice’s audio data is missing {audio}")
        if not os.path.exists(text):
            raise FileNotFoundError(f"voice’s text data is missing {text}")
        with open(text,"r",encoding="utf-8") as f:
            text = f.read()
        return audio,text


if __name__ == '__main__':
    gen = FishSpeechEngine(voice="default")
    gen.genWavFile("这个声音跟我的声音像吗","test.wav")

