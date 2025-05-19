import logging
from datetime import datetime
import json,os
from typing import Any, Dict, List, Union
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def extract_multiple_subdirectories(file_path, num_subdirectories):
   subdirectories = []
   for _ in range(num_subdirectories):
      file_path, subdirectory = os.path.split(file_path)
      subdirectories.insert(0, subdirectory)
   return subdirectories

class StoryData:
    def __init__(self, d: Dict[str, Any] = {}):
        self._init_data(d)

    def _init_data(self, d: Dict[str, Any]):
        for k, v in d.items():
            self.set(k, v)

    @property
    def isEmpty(self):
        return not self.__dict__

    def parse_data(self,value, key=None):
        if isinstance(value, (str,int,bool,float,StoryData)):
            if key: super().__setattr__(key, value)
            return value
        else:
            try:
                if key:super().__setattr__(key, str(value))
                logger.debug(f"parse {value.__class__} to str")
                return str(value)
            except Exception as e:
                if key:super().__setattr__(key,None)
                logger.exception(f"try to parse {value.__class__} fail,exception happen:\n{e}\nStoryData only accept str,int,list,dict,bool or StoryData data")
                return None

    def set(self, key: str, value: Any):
        """Set attribute with recursive conversion for dict/list."""
        if isinstance(value, dict):
            super().__setattr__(key, StoryData(value))
        elif isinstance(value, list):
            super().__setattr__(key, [StoryData(i) if isinstance(i, dict) else self.parse_data(i,key=None) for i in value])
        else:
            self.parse_data(key=key, value=value)


    def has(self, key: str) -> bool:
        return hasattr(self, key)

    def __getattr__(self, attr: str) -> Any:
        if "LAZY_" in attr:
            return "{LAZY_IGNORE}"
        raise AttributeError(f"Cannot access '{attr}' before assignment.")

    def __setattr__(self, k: str, v: Any):
        logger.debug(f"set '{v}' to '{k}'")
        self.set(k, v)

    def __getitem__(self, key: Union[str, int]) -> Any:
        if isinstance(key, str):
            key = key.replace(".","_")
        if isinstance(key, int):
            # 使用 members 顺序来支持索引访问
            key = self.members[key]
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        key = key.replace(".","_")
        setattr(self, key, value)

    def __repr__(self):
        summary = ', '.join(f"{k}={getattr(self, k)!r}" for k in self.members[:3])
        if len(self.members) > 3:
            summary += ', ...'
        return f"<StoryData({summary})>"

    def copy(self):
        return copy.deepcopy(self)

    @property
    def members(self) -> List[str]:
        return [k for k in self.__dict__ if not k.startswith('_')]

    def info(self):
        logger.info(self)

    def to_dict(self) -> dict:
        result = {}
        for key in self.members:
            value = getattr(self, key)
            if isinstance(value, StoryData):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, StoryData) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    @classmethod
    def from_json(cls, path: str) -> 'StoryData':
        #Compatible task ID reading
        if not os.path.isabs(path):
            if not (path.startswith("./outputs") or path.startswith("outputs")):
                path = os.path.join("outputs", path)
            path = os.path.join(os.getcwd(), path)
            
        if os.path.isdir(path):
            path = os.path.join(path, "StoryBoard.json")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data = cls(data)

        paths = extract_multiple_subdirectories(path,3)

        data.id = paths[1] if "outputs" in paths else "/".join(paths[0:2])

        return data

    def save(self,path=None,prefix=None):
        #Compatible task ID saving
        if path:
            if not os.path.isabs(path):
                if not (path.startswith("./outputs") or path.startswith("outputs")):
                    path = os.path.join("outputs", path)
                path = os.path.join(os.getcwd(), path)
            if os.path.isdir(path):
                path = os.path.join(path, "StoryBoard.json") if self.has("id") else os.path.join(path, "StoryData_notRoot.json")
        elif self.has("id"):
            path = f"./outputs/{self.id}/StoryBoard.json"
        else:
            raise IOError("it is not a root StoryData which has an ID")

        path_dir = os.path.dirname(path)

        os.makedirs(path_dir,exist_ok=True)

        with open(path, "w+", encoding="utf-8") as json_file:
            json.dump(self.to_dict(),json_file, ensure_ascii=False, indent=4)

        if self.has("id"):
            backup = datetime.now().strftime("%Y%m%d%H%M%S")
            backup_path = f"outputs/{self.id}/StoryBoard{'_'+prefix if prefix else ''}_{backup}.json"
            with open(backup_path, "w+", encoding="utf-8") as json_file:
                json.dump(self.to_dict(),json_file, ensure_ascii=False, indent=4)
                
        logger.debug(f"StoryData saved to {path}")


    def __len__(self):
        return len(self.members)

if __name__ == '__main__':
    data = StoryData.from_json("2025-05-09/130315")