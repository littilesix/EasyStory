# EasyStory 使用说明文档

EasyStory 是一个自动生成故事内容的工具链，支持从文本描述生成分镜图、角色生成、语音配音、视频合成等，适合用于动画制作、视频内容创作、AI剧本生成等场景。它支持灵活的后端配置、风格定义以及内容更新控制。

🎬 一键构建剧情生成视频
🧠 自动保持人物设定一致性
🎭 多角色、多镜头脚本支持，全程运行在本地大模型上

## 🔧 自动安装
 最低需要16G VRAM 
 点击**install.bat**

## 🔧手动安装

```bat
conda create -n easystory python=3.10.6
cd easystory
pip install -r installer/requirements.txt
python installer/get-models.py
```

## 🚀快速使用

点击 run.bat

你可以预先在 `run_setting.txt` 中配置风格、语音、重生成选项等。

------

## 🧰 风格配置示例

所有风格关键词必须使用英文：

```
base:professional 3d model
pixar:pixar animation
dream:dreamWorks
```

可以自定义为：

```
base:Ink Wash Painting
qibaishi:Qi Baishi
```

`base` 是必须保留并定义的基础风格项。

------

## 📂 目录结构

```c#
EasyStory
├── env/                    # python环境安装位置
├── installer/         		# 安装缓存和脚本
├── libs/                 	# 命令行工具和资产文件
├── models/                 # 所有模型
├── modules/                # 项目代码模块,可以直接索引的目录
    ├── after_effect/       # 字幕、高清放大，及其他插件
    ├── diffusers_helper/   # framepack带的diffusers优化
    ├── engines/            # 后端引擎模块（图像/语音/视频）
    ├── tools/        		# 工具集（日志、清理、流重定向等）
    ├── lora/             	# lora相关模块
    ├── backend.py     		# 后端引擎的协议
    ├── environments.py    	# 重要的环境变量设置，语言场景任务等
    ├── RoleGen.py     		# 角色生成流程
    ├── ShotGen.py     		# 分镜图片视频资源生成流程
    ├── StoryboardGen.py    # 故事模板生成流程
    ├── style.py   			# 风格类的基类
    ├── storydata.py    	# 一个数据类，更直观的成员索引，序列化json和dict
    ├── StoryTask.py    	# 任务和流程、缓存管理
├── outputs/           		# 故事项目文件及缓存，角色项目文件及缓存，以及最后的输出
├── scripts/           		# 自定义的后台和插件，运行脚本等
├── logs/           		# 日志
├── env.bat              	# 打开运行环境
├── install.bat        		# 快速安装
├── run.bat             	# 快速运行
├── run_setting.bat         # 快速运行配置
```

------

## 💻代码示例

```python
from StoryTask import StoryTask,RenewConfig,StoryTaskConfig
from style import Style
from after_effect import *
from engines import *
from backend import *
from after_effect import *
import cv2
from environments import *

def convert_to_grayscale(img):
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {input_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

class myStyle(Style):
    base = "professional 3d model"  #必须保留的成员
    render = "octane render,highly detailed,volumetric,dramatic lighting"
    pixar = "pixar animation"
    dream = "dreamWorks"
    
class myCustomPlugin():  
    #FrameProcessPlugin协议必须实现的方法和成员，可以查看backend.py
    def __init__(self,processor:AfterEffectProcessor):
        self.processor = processor
        self.width = processor.video_width
        self.prepareDatas()

    def prepareDatas(self)
    	for frame in self.processor.frames:
            frame.data.anything = "anything" #frame.data是StoryData数据类
        
    def rend(self,frame:Frame):
        print(frame.data.anything) #获取数据
        return convert_to_grayscale(frame.image) #一定要返回处理过的帧
        
renew = RenewConfig()

renew.images = True  #重写生成的分镜图片。

renew.shot_list = [0,1]#重跑0、1分镜

plugin_list = [FrameProcessPlugin(myCustomPlugin),FrameProcessPlugin(subtitle)]

#进阶用法  以下为可选设置，各个engine提供的配置项

LANG = JP  #来自environments

OllamaEngine.model = "deepseek-v3"  #指定模型，而不是是列表的第一个

voice = "唐国强"  #libs/voices里面

Subtitle.subtitle_color = (255,235,255) #调整字幕颜色

#更多设置自行探索

config = StoryTaskConfig(style=myStyle,renew = renew, voice=voice, plugin_list=plugin_list)

config.renew.videos = True

task = StoryTask(none,config)

```

------

## 📜 开源协议

项目推荐使用 MIT 协议
