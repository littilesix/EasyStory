# EasyStory User Manual

**EasyStory** is a toolchain for automatically generating story content. It supports storyboard generation from text descriptions, character creation, voice-over, video synthesis, and more. It is suitable for animation production, video content creation, and AI script generation. It allows flexible backend configuration, style definition, and content update control.

🎬 One-click story + scene generation

🧠 Keeps character settings consistent across scenes

🎭 Supports multi-character, multi-shot scripting – all powered by local LLMs

------

## 🔧 Automatic Installation

16G VRAM Need

Click **install.bat**

------

## 🔧 Manual Installation

```bat
conda create -n easystory python=3.10.6
cd easystory
pip install -r installer/requirements.txt
python installer/get-models.py
```

------

## 🚀 Quick Start

Click `run.bat`

You can also configure tasks in `run_setting.txt` before running.

## 🧰 Style Configuration Example

All style keywords **must be in English**:

```
base:professional 3d model  
pixar:pixar animation  
dream:dreamWorks  
```

You can customize them like this:

```
base:Ink Wash Painting  
qibaishi:Qi Baishi  
```

The `base` entry is **mandatory** and must be defined.

------

## 📂 Directory Structure

```c#
EasyStory
├── env/                    # Python environment location
├── installer/              # Installation cache and scripts
├── libs/                   # CLI tools and asset files
├── models/                 # All models
├── modules/                # Project code modules, can be directly imported
    ├── after_effect/       # Subtitle generation, upscaling, and other plugins
    ├── diffusers_helper/   # Optimizations for diffusers in framepack
    ├── engines/            # Backend engines (image, audio, video)
    ├── tools/              # Utility tools (logging, cleanup, stream redirection, etc.)
    ├── lora/               # LoRA-related modules
    ├── backend.py          # Protocol for backend engines
    ├── environments.py     # Important environment variables (language, scenario, task)
    ├── RoleGen.py          # Character generation process
    ├── ShotGen.py          # Shot image/video generation process
    ├── StoryboardGen.py    # Storyboard template generation process
    ├── style.py            # Base class for styles
    ├── storydata.py        # A data class for intuitive member access, JSON and dict serialization
    ├── StoryTask.py        # Task process and cache management
├── outputs/                # Story project files, character files, and final output
├── scripts/                # custom backends and plugins,run script.already in the python search path.
├── logs/                   # Logs
├── env.bat                 # Launch environment
├── install.bat             # Quick installation
├── run.bat                 # Quick start script
├── run_setting.bat         # Quick start script settings
```

------

## 💻 Code Example

```python
from StoryTask import StoryTask, RenewConfig, StoryTaskConfig
from style import Style
from after_effect import *
from engines import *
from backend import *
import cv2
from environments import *

def convert_to_grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

class myStyle(Style):
    base = "professional 3d model"  # Required member
    render = "octane render, highly detailed, volumetric, dramatic lighting"
    pixar = "pixar animation"
    dream = "dreamWorks"
    
class myCustomPlugin():
    # Must implement FrameProcessPlugin protocol, see backend.py
    def __init__(self, processor: AfterEffectProcessor):
        self.processor = processor
        self.width = processor.video_width
        self.prepareDatas()

    def prepareDatas(self):
        for frame in self.processor.frames:
            frame.data.anything = "anything"  # frame.data is a StoryData instance
        
    def rend(self, frame: Frame):
        print(frame.data.anything)  # Accessing data
        return convert_to_grayscale(frame.image)  # Must return processed frame

renew = RenewConfig()
renew.images = True  # Regenerate storyboard images
renew.shot_list = [0, 1]  # Regenerate storyboard shots 0 and 1

plugin_list = [FrameProcessPlugin(myCustomPlugin), FrameProcessPlugin(subtitle)]

#Andvance
# Optional configurations for engine
LANG = "JP"  # From environments.py
ACTION = "make a video about cat"
OllamaEngine.model = "deepseek-v3"  # Specify model instead of using default
voice = "TangGuoqiang"  # Voice name from libs/voices
Subtitle.subtitle_color = (255, 235, 255)  # Customize subtitle color

#and more ......

config = StoryTaskConfig(style=myStyle, renew=renew, voice=voice, plugin_list=plugin_list)

config.renew.videos = True

task = StoryTask(None, config)
```

------

## 📜 Open Source License

MIT License for this project.
