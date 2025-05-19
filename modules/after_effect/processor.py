import os
import numpy as np
from storydata import StoryData
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import cv2

class Frame:
    def __init__(self, index:int, image:np.ndarray, data:StoryData, status:bool):
        self.index = index
        self.image = image
        self.data = data
        self.bool = bool

class AfterEffectProcessor:
    def __init__(
        self,
        shot:StoryData,
        data:StoryData,
        plugin_list = None
        ):
    
        self.plugin_list = plugin_list
        if not plugin_list:
            return
        self.cap = cv2.VideoCapture(shot.scene_video)
        self.data = data
        self.shot = shot
        self.id = self.data.id
        self.output_path = f"{self.data.cache_dir}/Shot-{shot.id}-raw.mp4"
        self.frames = []

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video file")
        
        self.prepareVideoInfo()

        self.prepareEmptyVideoFrame()

        self.registerAllPlugins()
        #for frame in self.frames:print(frame.data)
        #pass AfterEffectProcessor to plugin，some plugin need to change video_width or other
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.video_width , self.video_height))

    def run(self):
        if not self.plugin_list:
            return None
        for frame in self.frames:
            frame.status, frame.image = self.cap.read()
            if not frame.status:
                logger.info("read frame fail,nothing effect to video")
                return None
            for plugin in self.plugin_list:
                try:
                    frame.image = plugin.rend(frame)
                except Exception as e:
                    logger.exception(f"{plugin._impl_cls.__name__} rend frame fail,nothing effect to video\n:{e}")
                    print(f"{plugin._impl_cls.__name__} rend frame fail,nothing effect to video\n:{e}")
                    continue
            self.out.write(frame.image)

        self.cap.release()
        self.out.release()
        if os.path.exists(self.output_path):
            return self.output_path
        else:
            return None

    def prepareVideoInfo(self):
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) #帧率
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.total_frames / self.fps

    def registerAllPlugins(self):
        if self.plugin_list:
            for plugin in self.plugin_list:
                plugin.register(self)

    def prepareEmptyVideoFrame(self):
        for i in range(self.total_frames):
            self.frames.append(Frame(i,None,StoryData(),False))


