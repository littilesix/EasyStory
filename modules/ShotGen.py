import sys,os
from tools import clean
from storydata import StoryData
from PIL import Image
from glob import glob
import re
from tools import FFmpegDownloader
from after_effect import AfterEffectProcessor,merge_videos,merge_audio_video
import cv2
from backend import ImageBackend,VideoBackend
from engines import FluxEngine,FramePackEngine
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pattern = r"Scene_(\d+)_(\d+).mp4"

def get_sorted_video_files(video_files,pattern= pattern,index=2):
    # 定义一个正则表达式来匹配文件名中的索引部分
    #pattern = r"Shot_(\d+)_(\d+).mp4"  # 这里匹配 "Shot_1_index1.mp4" 之类的文件
    # 对文件进行排序
    try:
        sorted_files = sorted(video_files, key=lambda x: int(re.search(pattern, x).group(index)))
    except:
        return []
    return sorted_files

class ShotGener:
    def __init__(
            self,
            data:StoryData,
            imgBackend:ImageBackend = None,
            videoBackend:VideoBackend =None,
            plugin_list=None,
            width=600,
            height=800,
            ):

        self.imgBackend = imgBackend if imgBackend else ImageBackend(FluxEngine)
        self.videoBackend = videoBackend if videoBackend else VideoBackend(FramePackEngine)
        self.width = width
        self.height = height
        self.data = data
        self.roles = data.roles
        self.shots = data.shots
        self.scenes = data.scenes
        self.plugin_list = plugin_list

        if not self.shots:
            logger.info("shots is empty")
            return

        self.images_all_exists = [os.path.exists(f"{self.data.cache_dir}/Shot_{shot.id}.png") for shot in self.shots]

        if False in self.images_all_exists:
            self.imgBackend.register()
            for shot in self.shots:self.ShotImageProcess(shot)
            clean(self.imgBackend,"_instance")
            clean(self,"imgBackend")
        else:
            for shot in self.shots: shot.img = f"{self.data.cache_dir}/Shot_{shot.id}.png"

        self.scenes_clips = [glob(f"{self.data.cache_dir}/Shot_{self.shots[scene.shot_id].id}_Scene_{scene.id}_*.mp4") for scene in self.scenes]

        self.scenes_clips = [get_sorted_video_files(clips) for clips in self.scenes_clips]

        for clips,scene in zip(self.scenes_clips,self.scenes):

            logger.info(f"scene{scene.id} on shot{self.shots[scene.shot_id].id}'s cache: {clips}")

        self.videos_all_exists = [len(clips)>0 and int(re.search(pattern, clips[-1]).group(2))%9==1 for clips in self.scenes_clips]

        logger.info(f"shot's clips cache status:{self.videos_all_exists}")

        if False in self.videos_all_exists:
            self.videoBackend.register(outputs=self.data.cache_dir)
            for scene in self.scenes:
                self.SceneVideoProcess(scene)
            clean(self.videoBackend,"_instance")
            clean(self,"videoBackend")
        else:
            for scene in self.scenes:
                current_shot = self.shots[scene.shot_id]
                current_scene = current_shot.scenes[scene.scene_id]
                scene.clips = self.scenes_clips[scene.id]
                current_scene.video = scene.clips[-1]

        self.ffmpeg_path = FFmpegDownloader(destination="libs").run()

        for shot in self.shots:
            self.MergeScenesToShotVideo(shot)

        if plugin_list:
            logger.info("processing AfterEffect")
            for shot in self.shots:
                self.sendShotVideoToAfter(shot)

        for shot in self.shots:
            shot.merge_video = f"{self.data.cache_dir}/Shot-{shot.id}-merge.mp4"
            merge_audio_video(shot.raw_video, shot.audio, shot.merge_video, ffmpeg=self.ffmpeg_path)


        self.data.video = f'{self.data.dir}/{self.data.id.replace("/","-")}.mp4'
        merge_videos([shot.merge_video for shot in self.shots], self.data.video, ffmpeg=self.ffmpeg_path)

        self.data.save(prefix='afterAndmerge')

    def sendShotVideoToAfter(self,shot):
        path = f"{self.data.cache_dir}/Shot-{shot.id}-raw.mp4"
        if os.path.exists(path):
            shot.raw_video = path
            return
        result = AfterEffectProcessor(shot,self.data,self.plugin_list).run()
        if result:
            shot.raw_video = result
        else:
            shot.raw_video = shot.scene_video

    def MergeScenesToShotVideo(self,shot):
        path = f"{self.data.cache_dir}/Shot-{shot.id}-scene.mp4"
        merge_videos([scene.video for scene in shot.scenes], path,ffmpeg=self.ffmpeg_path)
        shot.scene_video = path

    def ShotImageProcess(self,shot):
        logger.info(f"==========Shot--{shot.id}--=========runing")
        shot.info()
        if self.images_all_exists[shot.id]:
            shot.img = f"{self.data.cache_dir}/Shot_{shot.id}.png"
        else:
            shot.img = self.genShotImage(shot)

    def SceneVideoProcess(self,scene):
        current_shot = self.shots[scene.shot_id]
        current_scene = current_shot.scenes[scene.scene_id]
        basename = f"Shot_{current_shot.id}_Scene_{scene.id}"
        if self.videos_all_exists[scene.id]:
            scene.clips = self.scenes_clips[scene.id]
            current_scene.video = scene.clips[-1]
            return
        if scene.scene_id == 0:
            scene.img = current_shot.img
        else:
            last_videos = self.scenes[scene.id-1].clips
            save_path = f"{self.data.cache_dir}/{basename}.png"
            self.getLastFrame(last_videos[-1],save_path)
            scene.img = save_path
        scene.clips = self.videoBackend.getVideoClips(
            scene.img,
            current_scene.video_prompt,
            total_second = current_scene.time,
            basename = basename
            )
        current_scene.video = scene.clips[-1]

    def genShotImage(self,shot):
        path = f"{self.data.cache_dir}/Shot_{shot.id}.png"
        if shot.roles_number == 0:
            image = self.imgBackend.getImageFromText(prompt=shot.image_prompt,width=self.width,height=self.height)
        if shot.roles_number > 0: 
            first_role_lora = self.roles[shot.roles[0]].lora
            self.imgBackend.load_lora(first_role_lora)
            logger.info(f"load lora {first_role_lora}")
            img1 = self.imgBackend.getImageFromText(prompt=shot.image_prompt,width=self.width,height=self.height)
            self.imgBackend.unload_lora()
            image = img1
        if shot.roles_number > 1:
            second_role_lora = self.roles[shot.roles[1]].lora
            self.imgBackend.load_lora(second_role_lora)
            logger.info(f"load lora {second_role_lora}")
            img2 = self.imgBackend.getImageFromText(prompt=shot.image_prompt,width=self.width,height=self.height)
            self.imgBackend.unload_lora()
            mergeImage = self.merge_images(img1, img2)
            mergeImage.save(f"{self.data.cache_dir}/merge_{shot.id}.png")
            image = self.imgBackend.getImageFromImage(image=mergeImage,prompt=shot.merge_prompt,width=self.width,height=self.height)
        image.save(path)
        return path

    def getLastFrame(self, video_path, save_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"can't open the path {video_path}")

        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"when get last frame,read video fail,check the video")

        # 定位到最后一帧（帧编号从0开始）
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("cant't read last frame")

        # 保存最后一帧为图像文件
        cv2.imwrite(save_path, frame)
        logger.info(f"save the last frame to: {save_path}")


    @staticmethod
    def merge_images(img1,img2):
        # 打开两张 PNG 图片
        # 确保两张图大小相同
        assert img1.size == img2.size, "sizes must match"

        width, height = img1.size
        half_width = width // 2

        # 从第一张图裁剪左半部分
        left_part = img1.crop((0, 0, half_width, height))

        # 从第二张图裁剪右半部分
        right_part = img2.crop((half_width, 0, width, height))

        # 创建新图像
        new_img = Image.new("RGB", (width, height))

        # 拼接两个部分
        new_img.paste(left_part, (0, 0))
        new_img.paste(right_part, (half_width, 0))

        # 保存新图
        return new_img
        
if __name__ == '__main__':
    from engines import *
    from backend import *
    from storydata import StoryData
    data = StoryData.from_json("test")
    audioGen =  AudioBackend(FishSpeechEngine).register(data,voice="default")
    audioGen.gen_all_shots_audio()
    del audioGen
    from RoleGen import RoleGener
    role = RoleGener(data)
    from after_effect import Subtitle,VideoScaler
    from backend import FrameProcessPlugin
    #result = AfterEffectProcessor(data.shots[0],data,plugin_list=[FrameProcessPlugin(Subtitle)]).run()
    gen = ShotGener(data,plugin_list=[FrameProcessPlugin(Subtitle),FrameProcessPlugin(VideoScaler)])
    #gen = ShotGener(data,plugin_list=[FrameProcessPlugin(Subtitle)])