#confing=utf-8
import sys,os
from tools import clean,cleanAll,FakeModules,setup_logging #,StreamToLogger
from storydata import StoryData
from engines import *
from backend import *
from RoleGen import RoleGener
from StoryboardGen import StoryBoardGener
from ShotGen import ShotGener
from after_effect import *
import logging
from style import Style
os.makedirs("outputs/roles",exist_ok=True)
setup_logging()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#sys.stdout = StreamToLogger(logger, logging.INFO)
#sys.stderr = StreamToLogger(logger, logging.ERROR)

def extract_custom_style(lines: list[str]) -> type[Style] | None:
    """
    从 run_setting.txt 的内容中提取自定义样式定义块（以 base 开头，以 #style end 结尾）。
    返回一个动态构造的 CustomStyle 类，继承自 Style。
    """
    inside_style = False
    style_dict = {}

    for line in lines:
        stripped = line.strip()
        if not inside_style:
            if stripped.startswith("base:"):
                key, value = stripped.split(":", 1)
                style_dict[key.strip()] = value.strip()
                inside_style = True
        elif stripped.lower() == "style end":
            break
        elif ":" in stripped:
            key, value = stripped.split(":", 1)
            style_dict[key.strip()] = value.strip()

    if not style_dict or "base" not in style_dict:
        return None  # 或 raise ValueError("Missing 'base' in style block")

    return type("CustomStyle", (Style,), style_dict)

class RenewConfig(StoryData):
    def __init__(self):
        #self.renewData = StoryData()
        super().__init__()
        self.shots = False
        self.audios = False
        self.roles = False
        self.images = False
        self.videos = False
        self.afters = False
        self.shot_list = []

    def __setattr__(self,k,v):
        valid_list = ["shots","audios","roles","images","videos","afters","shot_list"]
        if k in valid_list:
            super().__setattr__(k,v)
        else:
            raise AttributeError(f"valid key is {valid_list},not include {k}")

class DefaultStyle(Style):
    base = "professional 3d model"
    render = "octane render,highly detailed,volumetric,dramatic lighting"
    pixar = "pixar animation"
    dream = "dreamWorks"

class StoryTaskConfig:
    def __init__(
        self,
        chatBackend:ChatBackend = None,
        videoBackend:VideoBackend = None,
        imageBackend:ImageBackend = None,
        audioBackend:AudioBackend = None,
        video_with:int = 600,
        video_height:int = 800,
        plugin_list:list = None,
        voice:str = "default",
        style:type = DefaultStyle,
        renew:RenewConfig = None
        ):

        self.chatBackend = chatBackend  if chatBackend else ChatBackend(OllamaEngine)

        self.videoBackend = videoBackend if videoBackend else VideoBackend(FramePackEngine)

        self.imageBackend = imageBackend if imageBackend else ImageBackend(FluxEngine)

        self.audioBackend = audioBackend if audioBackend else AudioBackend(FishSpeechEngine)

        self.video_with = video_with

        self.video_height = video_height

        self.plugin_list = plugin_list
        
        self.voice = voice

        self.style = style

        self.renew = renew if renew else RenewConfig()

class StoryTask:
    def __init__(
        self,
        task:str=None,
        config = None
        ):

        if not config:
            config = StoryTaskConfig()

        redo_list = self.cache_process(task,config.renew) if task else []

        if redo_list:
            config.renew.shots = True

        logger.info(f"{task+' ' if task else ''}runing StoryboardGener")
        
        self.storyboard = StoryBoardGener(
            task=task,
            refresh_shot=config.renew.shots,
            refresh_list=redo_list,
            backend=config.chatBackend,
            style=config.style
            )

        if not self.storyboard.isOk:
            logger.info("StoryBoardGener init fail")
            return

        self.data = self.storyboard.data

        logger.info(f"task is '{self.data.id}'")

        self.roles = self.data.roles

        logger.info(f"{self.data.id} runing AudioGener")
        
        config.audioBackend.register(self.data,voice=config.voice)
        config.audioBackend.gen_all_shots_audio()
        clean(config.audioBackend,"_instance")
        #clean(config, "audioBackend")
        
        if config.renew.roles and task:
            for role in self.data.roles:
                StoryTask.remove_cache(f"outputs/roles/{role.id}",start="img",end = "png")
                StoryTask.remove_cache(f"outputs/roles/{role.id}",start="lora",end = "safetensors")

        logger.info(f"{self.data.id} runing RoleGen")
        self.roleGen = RoleGener(self.data,backend = config.imageBackend,style=config.style)

        logger.info(f"{self.data.id} runing ShotGener")
        shotGen = ShotGener(
            data = self.data,
            imgBackend = config.imageBackend,
            videoBackend = config.videoBackend,
            plugin_list= config.plugin_list,
            width=config.video_with,
            height=config.video_height,
            )

    @staticmethod
    def remove_cache(cache_dir,end:str,start:str="Shot_"):
        files =[os.path.join(cache_dir,file) for file in os.listdir(cache_dir) if file.startswith(start) and file.endswith(f"{end}")]
        for file in files:
            os.remove(file),logger.info(f"remove cache {file}")

    def cache_process(self,task,renew):
        redo_list_update = []
        cache_dir = f'outputs/{task}/cache'
        if os.path.exists(cache_dir):
            if renew.shots:
                logger.info("process shot refine regenerate")
                renew.audios = True
            if renew.audios:
                logger.info("process audio file delete ing...")
                StoryTask.remove_cache(cache_dir,end = ".wav")
                renew.videos = True
            if renew.images:
                logger.info("process picture file delete ing...")
                StoryTask.remove_cache(cache_dir,end = ".png")
                StoryTask.remove_cache(cache_dir,start="merge",end = ".png")
                renew.videos = True
            if renew.videos:
                logger.info("process video file delete ing...")
                StoryTask.remove_cache(cache_dir,end = ".mp4")
                renew.afters = True
            if renew.afters:
                logger.info("process video with after_effect file delete ing...")
                StoryTask.remove_cache(cache_dir,start="Shot-",end = ".mp4")
            if renew.shot_list:
                for shot_id in renew.shot_list:
                    if isinstance(shot_id,int):
                        redo_list_update.append(shot_id)
                        need_to_delete_list = [os.path.join(cache_dir,file) for file in os.listdir(cache_dir) if file.startswith(f"Shot_{shot_id}") or file.startswith(f"Shot-{shot_id}")]
                        for file in need_to_delete_list:
                            logger.info(f"Shot which index {shot_id} process cache {file} delete ing...")
                            os.remove(file)
        return redo_list_update

    @classmethod
    def fromText(cls,path="run_setting.txt"):
        if not os.path.exists(path):
            raise FileNotFoundError("missing run_setting.txt")
        with open(path, "r", encoding="utf-8") as f:
            setting = f.read()
        lines = [line for line in setting.splitlines() if not line.strip().startswith("#")]

        setting = "\n".join(lines)
        # 提取自定义样式类
        CustomStyle = extract_custom_style(lines)
        
        if CustomStyle:
            style = CustomStyle
        else:
            style = DefaultStyle
        
        print("\n=== Custom Style ===")
        print(CustomStyle.__name__)
        print(style("this is a test").style)

        import re

        def extract_setting(key, default=None):
            match = re.search(rf"{key}:(.*)", setting)
            return match.group(1).strip() if match else default

        # renew 相关设置
        renew = RenewConfig()
        renew.shots = extract_setting("renewShots", "no").lower() == "yes"
        renew.audios = extract_setting("renewAudios", "no").lower() == "yes"
        renew.roles = extract_setting("renewRoles", "no").lower() == "yes"
        renew.images = extract_setting("renewimages", "no").lower() == "yes"
        renew.videos = extract_setting("renewvideos", "no").lower() == "yes"
        renew.afters = extract_setting("renewAfters", "no").lower().strip() == "yes"

        import environments
        environments.LANG = extract_setting("lang")
        environments.SCENE = extract_setting("scene")
        environments.ACTION = extract_setting("action")

        print("\n=== Script Setting ===")
        print(f"LANG:{environments.LANG}")
        print(f"SCENE:{environments.SCENE}")
        print(f"ACTION:{environments.ACTION}")

        shot_list_str = extract_setting("renewShotByIndex", "")
        if shot_list_str:
            try:
                renew.shot_list = [int(i.strip()) for i in shot_list_str.split(",") if i.strip().isdigit()]
            except:
                logger.warning("Failed to parse renewShotByIndex")

        print("\n=== Renew Config ===")
        print(f"shots: {renew.shots}")
        print(f"audios: {renew.audios}")
        print(f"roles: {renew.roles}")
        print(f"images: {renew.images}")
        print(f"videos: {renew.videos}")
        print(f"afters: {renew.afters}")
        print(f"shot_list: {renew.shot_list}")

        # 后端配置
        chat_backend_str = extract_setting("ChatBackend", "OllamaEngine")
        image_backend_str = extract_setting("ImageBackend", "FluxEngine")
        video_backend_str = extract_setting("VideoBackend", "FramePackEngine")
        audio_backend_str = extract_setting("AudioBackend", "FishSpeechEngine")
        after_effect_str = extract_setting("AffterEffectPlugs", "")

        chat_backend = ChatBackend(eval(chat_backend_str))
        image_backend = ImageBackend(eval(image_backend_str))
        video_backend = VideoBackend(eval(video_backend_str))
        audio_backend = AudioBackend(eval(audio_backend_str))

        plugin_list = []
        plugin_names =[]
        for class_name in [p.strip() for p in after_effect_str.split(",") if p.strip()]:
            try:
                plugin_class = eval(class_name)
                plugin_backend = FrameProcessPlugin(plugin_class)
                plugin_list.append(plugin_backend)
                plugin_names.append(class_name)
            except Exception as e:
                logger.warning(f"Failed to load plugin class '{class_name}': {e}")

        print("\n=== Backend Setting ===")
        print(f"ChatBackend: {chat_backend_str}")
        print(f"ImageBackend: {image_backend_str}")
        print(f"VideoBackend: {video_backend_str}")
        print(f"AudioBackend: {audio_backend_str}")
        print(f"PluginList: {plugin_names}")

        # 构建 StoryTaskConfig 对象
        config = StoryTaskConfig(
            chatBackend=chat_backend,
            imageBackend=image_backend,
            videoBackend=video_backend,
            audioBackend=audio_backend,
            plugin_list=plugin_list,
            style=style,
            renew=renew
        )

        task = extract_setting("task", None)
        print(f"\n=== task ===\ntask: {task}")

        print(f"\n=== all setting readed ===")

        return cls(task=task, config=config)

if __name__ == '__main__':
    StoryTask.fromText()
    #renew = RenewConfig()
    #renew.audios = True
    #config = StoryTaskConfig(renew=renew)
    #task = StoryTask("test",config =config)
    #task = StoryTask("2025-05-08/172311",renew_audios=True)