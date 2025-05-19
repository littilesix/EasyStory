import sys,os
import numpy as np
from PIL import Image
from tools import clean,cleanAll
from storydata import StoryData
from lora import LoraConfig,Lora
from backend import ImageBackend
from engines import FluxEngine
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from style import Pro3DModel

class RoleGener:
    def __init__(self,data:StoryData,backend:ImageBackend = None,style=Pro3DModel):

        os.makedirs(f"{os.getcwd()}/outputs/roles",exist_ok=True)
        
        if backend:
            self.backend = backend
        else:
            self.backend = ImageBackend(FluxEngine)

        self.data = data
        self.roles = data.roles
        self.angles = [45 , 90, 135, 180, 225, 270]
        self.style = style

        for role in self.roles:
            path = f"./outputs/roles/{role.id}"
            os.makedirs(path, exist_ok=True)
            role.path = path
            role.prompt= self.genPrompt(role)

        if not self.isAllRolesFileExists(file="img.png"):
            #如果缓存全部存在就跳过,免于初始化ImageGener
            self.backend.register()
        self.genImgAll()
        clean(self.backend,"_instance")
        clean(self, "backend")
        """
        if not self.isAllRolesFileExists(file="model.glb"):
            self.init_3d()
        self.gen3DAll()
        cleanAll(self,"pipeline_texgen","pipeline_shapegen")

        if not self.isAllRolesFileExists("angle_270.png")
            for role in self.roles:
                self.genAngleImages(role["mesh"],role["path"])
        """
        if self.isAllRolesFileExists(file="lora.safetensors"):
            for role in self.roles:role.lora = f"{role.path}/lora.safetensors"
            data.save()
            return

        for role in self.roles:
            predit_path = f"{role.path}/lora.safetensors"
            if role.cache.lora_safetensors:
                role.lora = predit_path
            else:
                role_lora_prompt = f"{role.eng} is {role.spec},{role.trans}"
                self.lora = Lora(LoraConfig(role.path, role.id, role_lora_prompt))
                self.lora.run()
                clean(self,attr="lora")
                role.lora = predit_path

        data.save()

    def genImgAll(self):
        for role in self.roles:
            predit_path = f"{role.path}/img.png"
            if role.cache.img_png:
                role.img = predit_path
            else:
                image = self.backend.getImageFromText(prompt=role.prompt,width=512,height=512)
                image.save(predit_path)
                role.img = predit_path

    def isAllRolesFileExists(self, file):
        all_exist = True
        for role in self.roles:
            path = os.path.join(role.path, file)
            if not role.has("cache"):
                role.cache = StoryData()
            if not os.path.exists(path):
                role.cache[file] = False
                all_exist = False
            else:
                role.cache[file] = True
        if all_exist:
            logger.info(f"{file} is all exists")
        return all_exist

    def genPrompt(self,role):
        predit_path = f"{role.path}/desc.txt"
        if not os.path.exists(predit_path):
            with open(predit_path,"w+",encoding="utf-8") as f:
                f.write(f"{role.id}\n{role.trans}\n{self.style.base}")
        return self.style(f"a {role.spec},{role.trans}").styleFrontView

"""
    def gen3DAll(self):
        #self.init_3d()
        for role in self.roles:
            predit_path = f"{role.path}/model.glb"
            if role.cache.model_glb:
                role.mesh = trimesh.load_mesh(predit_path)
            else:
                mesh = self.gen3D(role.img)
                mesh.export(predit_path)
                role.mesh = mesh


    def init_3d(self):
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        model_path = 'tencent/Hunyuan3D-2'
        self.pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        self.pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, 
        subfolder="hunyuan3d-dit-v2-0"
        )

    def gen3D(self,image):
        import torch
        from hy3dgen.rembg import BackgroundRemover
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)

        mesh = self.pipeline_shapegen(
            image=image,
            num_inference_steps=30,
            octree_resolution=256,
            num_chunks=8000,
            generator=torch.manual_seed(12345),
            output_type='trimesh'
        )[0]

        mesh = self.pipeline_texgen(mesh, image=image)
        return mesh

    def genAngleImages(self,mesh:trimesh.Trimesh,role_path:str):
        from io import BytesIO
        scene = mesh.scene()
        angleImages = []
        for angle in self.angles:
            radians = np.radians(angle)
            scene.set_camera(
                angles=(0, radians, 0),
                #distance=2.5,
                center=mesh.centroid
            )
            img_bytes  = scene.save_image(resolution=(512, 512),)
            #img = Image.open(BytesIO(img_bytes))  # 设置分辨率
            path = f'{role_path}/angle_{angle}.png'
            angleImages.append(path)
            with open(path,'wb') as f:
                f.write(img_bytes)
        return angleImages
"""
if __name__ == '__main__':
    test = StoryData.from_json("2025-05-11/125639")
    gen = RoleGener(test)
    #from ShotGen import ShotGener
    #shots = ShotGener(test)