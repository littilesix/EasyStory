import json,re,os
from tools import clean
from storydata import StoryData
from backend import ChatBackend
from engines.ollama import OllamaEngine
from datetime import datetime
import logging
from environments import *
import time
from style import Pro3DModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#model="qwen3:30b-a3b-q4_K_M"
class StoryBoardGener:
    def __init__(
            self,
            task: str = None, 
            refresh_shot: bool = False, 
            refresh_list=None, 
            backend=None,
            style=Pro3DModel
            ):

        self.isOk = False
        self.cache_mode = False
        self.backend = backend if backend else ChatBackend(OllamaEngine())
        self.refresh_list = refresh_list or []
        self.refresh_shot = refresh_shot
        self.style = style

        # Step 1: 判断是否使用缓存模式
        if task and os.path.exists(f"./outputs/{task}") and os.path.exists(f"./outputs/{task}/StoryBoard.json"):
            self.cache_mode = True

        # Step 2: 后端注册逻辑
        if not self.cache_mode or self.refresh_shot:
            self.backend.register()
            if not self.backend.isOpened:
                self.backend.close()
                raise ChildProcessError("text init fail,shut down the proxy or change backend model name")

        # Step 3: 加载或创建数据对象
        self.data = self._load_or_create_data(task)

        #os.makedirs(self.data.cache_dir, exist_ok=True)

        # Step 4: 刷新镜头
        if self.refresh_shot:
            self._refresh_shots()
            self.save()
            self.backend.close()
            self.isOk = True
            return

        # Step 5: 如果使用缓存直接返回
        if self.cache_mode:
            self.isOk = True
            return

        # Step 6: 生成新的故事板
        if not self.get():
            logger.info("when get storyboard , but it is fail")
            return

        self.save()
        self.backend.close()
        clean(self, "backend")
        self.isOk = True

    def _load_or_create_data(self, task):
        if self.cache_mode:
            try:
                data = StoryData.from_json(task)
                os.makedirs(data.cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning("read project cache fail, regenerate project")
                logger.exception(e)
                data = StoryData()
                data.id = task
                self.refresh_shot = False
        else:
            data = StoryData()
            data.id = datetime.now().strftime("%Y-%m-%d/%H%M%S")
            self.refresh_shot = False

        data.dir = f"./outputs/{data.id}"
        data.cache_dir = f"./outputs/{data.id}/cache"
        return data

    def _refresh_shots(self):
        for shot in self.data.shots:
            if not self.refresh_list or shot.id in self.refresh_list:
                self.refine_shot(shot)

    def getNum(self,line):
        pattern = re.compile(r'\d+')
        match = pattern.search(line)
        if match:
            return int(match.group())


    def get_once(self):
        #DATA = StoryData()
        #DATA.ROLESINFO = getRolesInfoGlobal()
        self.md = self.backend.stream(read_storyboard_prompt())
        self.parseToData(self.md)
        for shot in self.data.shots:
            self.refine_shot(shot)

    def get(self):
        for i in range(10):
            try:
                self.get_once()
                return True
            except Exception as e:
                logger.warning(f"[{i+1}/10] generate StoryData failed.")
                logger.exception(e)
                time.sleep(1)  # optional: avoid rapid retry loop
        logger.error("All 10 attempts failed to generate StoryData.")
        return False

    def getRolesInInfo(self,*texts):
        def getRoleInInfo(role):
            return [role.id in text or role.name in text or role.eng in text for text in texts]
        return [role.id for role in self.data.roles if True in getRoleInInfo(role)]

    def save(self):
        #os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        self.data.save()
        if not self.cache_mode:
            self.md_file = f"{self.data.dir}/StoryBoard.md"
            with open(self.md_file, "w+", encoding="utf-8") as f:
                f.write(self.md)

    def refine_shot_subtitle(self,shot):
        prompt = read_refine_shot_subtitle().format(
            shot.id,
            shot.text,
            LASTSHOTSUBTITLE + self.data.shots[shot.id-1].subtitle if shot.id !=0 else FIRSTSHOTWITHOUTCONTEXT  ,
            self.data.roles_info
            )

        result = self.backend.stream(prompt)
        lines = result.strip().split("\n")
        shot.scenes = []
        index = 0
        for line in lines:
            if "subtitle_text" in line:
                shot.subtitle = line.split("subtitle_text:")[1].strip()
            elif "first_frame_prompt" in line:
                shot.first_frame_prompt = line.split("first_frame_prompt:")[1].strip()
                shot.roles = self.getRolesInInfo(shot.text,shot.subtitle,shot.first_frame_prompt)
                shot.roles_number = len(shot.roles)
                roles = [self.data.roles[role] for role in shot.roles]
                roles_tail_string = f'{shot.roles_number}roles' if shot.roles_number>1 else ""
                if shot.roles_number == 0:
                    shot.desc_prompt = ""
                elif shot.roles_number == 1:
                    shot.desc_prompt = f"{roles[0].id}({roles[0].trans})"
                else:
                    shot.desc_prompt = f"{roles[0].id}({roles[0].trans}),{roles[0].spec} at left,{roles[1].id}({roles[1].trans}),{roles[0].spec} at right"
                shot.merge_prompt = self.style(shot.desc_prompt)+roles_tail_string+shot.first_frame_prompt
                shot.image_prompt = shot.merge_prompt.style
            elif "split_text" in line:
                scene = StoryData()
                shot.scenes.append(scene)
                scene.id = index
                scene.split_text = line.split("split_text:")[1].strip()
            elif "scene_text" in line:
                scene = shot.scenes[index]
                scene.scene_text = line.split("scene_text:")[1].strip()
            elif "video_trends_prompt" in line:
                scene = shot.scenes[index]
                scene.video_trends_prompt = line.split("video_trends_prompt:")[1].strip()
                index+=1

    def prepare_scenes_data_in_one_shot(self,shot):
        start_index = len(self.data.scenes) if self.data.scenes else 0
        for scene in shot.scenes:
            scene_info = StoryData()
            scene_info.id = start_index+scene.id
            scene_info.shot_id = shot.id
            scene_info.scene_id = scene.id
            scene_info.isEnd = True if scene.id == len(shot.scenes)-1 else False
            self.data.scenes.append(scene_info)
            scene.roles = self.getRolesInInfo(scene.scene_text,scene.video_trends_prompt)
            scene.roles_number = len(scene.roles)
            roles = [self.data.roles[role] for role in scene.roles]
            scene.video_prompt = self.style(f'{"" if scene.roles_number==0 else ",".join([f"{role.eng} is {role.spec}" for role in roles])}')+scene.video_trends_prompt

    def refine_shot(self,shot):
        self.refine_shot_subtitle(shot)
        self.prepare_scenes_data_in_one_shot(shot)
        shot.info()

    def parseToData(self,text):
        lines = text.strip().split("\n")
        self.data.roles = StoryData()
        self.data.shots = []
        self.data.scenes = []
        roles = self.data.roles
        current_id = None
        current_index = 0
        for line in lines:
            if "background" in line:
                self.data.backgroud = line.split("background:")[1].strip()
            elif "roleEng" in line:
                eng = line.split("roleEng:")[1].strip()
                current_id = eng.replace(" ","_")
                roles[current_id] = StoryData()
                roles[current_id].eng = eng
                roles[current_id].id = current_id
            elif "roleName" in line:
                roles[current_id].name = line.split("roleName:")[1].strip()
            elif "roleDesc" in line:
                roles[current_id].desc = line.split("roleDesc:")[1].strip()
            elif "roleTrans" in line:
                roles[current_id].trans = line.split("roleTrans:")[1].strip()
            elif "roleSpecies" in line:
                roles[current_id].spec = line.split("roleSpecies:")[1].strip()
            elif "shotText" in line:
                shot = StoryData()
                shot.id = current_index
                shot.time = 4
                shot.text = line.split("shotText:")[1].strip()
                shot.subtitle = ""
                self.data.shots.append(shot)
                current_index+=1
        self.data.roles_info = "\n".join([f"Role\nid:{role.id}\nname:{role.name}\nenglishname:{role.eng}" for role in self.data.roles])


if __name__ == '__main__':
    #from datetime import datetime
    #tid = datetime.now().strftime("%Y-%m-%d/%H%M%S")
    #story = StoryBoardGener("2025-05-05/210124",refresh_shot=True)
    #from style import Pro3DModel

    backend = ChatBackend(OllamaEngine)
    backend.model = "qwen3:30b-a3b-q4_K_M"
    story = StoryBoardGener(backend=backend)
    #style = Pro3DModel
    #roles_tail_string = "two roles"
    #first = "Flame suddenly appears from behind, his tail shooting fire to create a wall of flames blocking the way"
    #desc_prompt = "Lucky(Brave and curious boy with a blue cap and red vest),boy at left,Spark(Orange fox with flame-like tail and red eyes),boy at right"
    #merge_prompt = style(desc_prompt)+roles_tail_string+first
    #print(merge_prompt.style)