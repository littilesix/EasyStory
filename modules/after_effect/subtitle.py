import cv2
from PIL import Image, ImageDraw,ImageFont 
import re
import numpy as np
import jieba
from .processor import AfterEffectProcessor,Frame
from storydata import StoryData
from tools import proportional_integer_allocation
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

punctuation = r"""。，、；：“”‘ ’《》【】（）！？－——……——～『』「」〔〕〖〗〘〙〚〛｛｝［］〈〉﹏～·!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~｡｢｣、〃》｟｠〝〞〟〜"""
punctuation_escape = re.escape(punctuation)


def replace_punctuation_with_other(line, strip_all=True, repl=""):
    if strip_all:
        # 去除所有非中英文与数字字符
        rule = re.compile(u"[^a-zA-Z0-9\u4e00-\u9fa5]")
        line = rule.sub('', line)
    else:
        pattern = f"[{punctuation_escape}]"
        line = re.sub(pattern, repl, line)
    return line.strip()

class Subtitle:
    title_size = 40
    author_size = 30
    subtitle_size = 40
    stroke_width = 2
    title_color = (0,0,0)
    author_color = (255,255,255)
    stroke_color = (255,255,51)
    subtitle_color = (255,255,255)
    subtitle_stroke_color = (255,255,255)
    font = ("libs/fonts/default.ttf",40)
    isAlreadyTransColor = False

    def __init__(self,processor:AfterEffectProcessor):
        self.transColor()
        self.processor = processor
        self.text = self.processor.shot.subtitle
        self.fps = self.processor.fps
        self.video_width = self.processor.video_width
        self.video_height = self.processor.video_height
        self.video_duration = self.processor.video_duration
        self.font = ImageFont.truetype(Subtitle.font[0], Subtitle.font[1])

        temp_img = Image.new('RGB', (self.video_width, self.video_height))
        draw = ImageDraw.Draw(temp_img)
        max_width = int(self.video_width * 0.9)

        bbox = draw.textbbox((0, 0), "测试文字", font=self.font)

        text_height = bbox[3] - bbox[1]

        self.start_y = 0.8*self.video_height+text_height

        lines = Subtitle.split_text_to_lines(self.text, draw, self.font, max_width)

        text_lengths = [len(text) for text,width in lines]

        int_frame_counts =  proportional_integer_allocation(text_lengths,self.processor.total_frames)

        frame_id = 0
        for frame_counts,line in zip(int_frame_counts,lines):
            for i in range(frame_counts):
                self.processor.frames[frame_id].data.text = line[0]
                self.processor.frames[frame_id].data.text_width = line[1]
                frame_id+=1
                
        assert frame_id == self.processor.total_frames, "Frame count mismatch!"

    @classmethod
    def transColor(cls):
        if cls.isAlreadyTransColor:
            return
        for key, value in cls.__dict__.items():
            if "color" in key:
                if isinstance(value, tuple) and len(value) == 3:
                    setattr(cls, key, Subtitle.from_RGB2BGR(value))
                elif isinstance(value, str):
                    setattr(cls, key, Subtitle.convert_color(value))
        cls.isAlreadyTransColor = True

    def rend(self,frame:Frame):
        return SubtitleFrame(frame.image,frame.data.text,frame.data.text_width,self).draw()

    @staticmethod
    def split_text_to_lines(text, draw, font, max_width):
        lines = []

        # 1. 拆句（按标点符号断句）
        sentences = re.split(r"[{}]".format(r".,;!\?:\\，。！？、\n\r……——（）《》\(\)\\{\}"), text)
        sentences = [s.strip() for s in sentences if s.strip()]

        logger.info(sentences)

        for sentence in sentences:
            line = ""
            line_width = 0

            # 2. 混合分词逻辑（中英文分开处理）
            # 中文连续串用结巴分词，非中文部分直接保留
            # 例："今天 is a good day" → ["今天", "is", "a", "good", "day"]
            segments = []
            pattern = re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+|[^\w\s]+|\s+')
            for frag in pattern.findall(sentence):
                if re.fullmatch(r'[\u4e00-\u9fff]+', frag):
                    # 对中文连续部分用结巴分词
                    segments.extend(jieba.cut(frag))
                else:
                    # 其他部分直接加入
                    segments.append(frag)

            # 3. 逐词构建每一行
            for token in segments:
                test_line = line + token
                test_width = draw.textlength(test_line, font=font)

                if test_width <= max_width:
                    line = test_line
                    line_width = test_width
                else:
                    if line.strip():
                        lines.append((line.strip(), line_width))
                    line = token
                    line_width = draw.textlength(token, font=font)

            if line.strip():
                lines.append((line.strip(), line_width))

        return lines

    @staticmethod
    def convert_color(hex_color):
        # 从16进制颜色值转换为RGB
        red = int(hex_color[1:3], 16)
        green = int(hex_color[3:5], 16)
        blue = int(hex_color[5:7], 16)
        bgr_array = (blue, green, red)
        return bgr_array

    @staticmethod
    def from_RGB2BGR(color):
        return (color[2],color[1],color[0])

class SubtitleFrame:
    def __init__(self,image:np.ndarray,text:str,text_width:int,video:Subtitle):
        self.image = image
        self.text = text
        self.video = video
        self.text_width = text_width

    def draw(self):
        pil_image = Image.fromarray(self.image)
        drawer = ImageDraw.Draw(pil_image)
        #text_width = drawer.textlength(self.text, font=self.video.font)
        self.x = (self.video.video_width - self.text_width) // 2
        drawer.text(
            (self.x, self.video.start_y),
            self.text,
            font=self.video.font,
            fill=self.video.title_color,
            stroke_fill=self.video.stroke_color,
            stroke_width=self.video.stroke_width
        )
        return np.array(pil_image)