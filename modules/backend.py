import inspect
from storydata import StoryData
from after_effect.processor import Frame,AfterEffectProcessor

backend_valid = ["validate", "register", "_impl_cls", "_instance","validateMethodParams"]

class Backend:
    required_methods = []
    required_attributes = []

    def __init__(self,impl_cls):
        if issubclass(impl_cls,Backend) or impl_cls == Backend:
            raise ValueError(f"{impl_cls.__name__} is a Backend wrapper already")
            return
        self.validate(impl_cls)
        self._impl_cls = impl_cls

    def register(self,*args, **kwargs):
        self._instance = self._impl_cls(*args, **kwargs)
        return self._instance

    def __call__(self,*args, **kwargs):
        self._instance = self._impl_cls(*args, **kwargs)
        return self._instance

    def validate(self, impl_cls):
        cls_name = impl_cls.__name__
        cls_dict = impl_cls.__dict__
        problems = []

        for attr in self.required_attributes:
            if not hasattr(impl_cls, attr):
                problems.append(f"missing attribute: {attr}")


        for method in self.required_methods:
            isList = True if isinstance(method, list) else False
            key = method[0] if isList else method
            func = cls_dict.get(key)
            if not (func and inspect.isfunction(func)):
                problems.append(f"missing method: {key}")
            else:
                if isList:
                    self.validateMethodParams(func,problems,method[1:])

        if problems:
            raise TypeError(
                f"Class '{cls_name}' does not conform to {self.__class__.__name__} requirements:\n"
                + "\n".join(problems)
                + "\nNote: 使用 Backend wrapper 时必须实现指定的方法、成员与构造参数，以确保接口一致性。"
            )

    def validateMethodParams(self,method,problems,valid_list):

        init_sig = inspect.signature(method)
        init_params = init_sig.parameters

        for req in valid_list:
            if isinstance(req, tuple):
                arg_name, expected_type = req
            else:
                arg_name, expected_type = req, None
            if arg_name not in init_params:
                problems.append(f"missing {method.__name__} method parameter: {arg_name}")
            else:
                param = init_params[arg_name]
                ann = param.annotation
                default = param.default
                actual_type = None
                if ann is not inspect._empty:
                        # 优先检查注解
                    actual_type = ann
                elif default is not inspect._empty:
                    # 如果没有注解，则根据默认值推断类型
                    actual_type = type(default)
                if expected_type and actual_type != expected_type:
                    problems.append(f"{method.__name__}  param '{arg_name}' has type {actual_type}, expected {expected_type}")
                elif expected_type is not None and actual_type is None:
                    problems.append(f"{method.__name__}'s  {arg_name}' has no annotation or default to infer type")

    def __setattr__(self, name, value):
        #print(name,value,name.startswith("_") or name.startswith("__") or name.startswith("required_") or name in ["validate","register"])
        if (name.startswith("required_")
            or name in backend_valid
            or name.startswith("_") and name.endswith("_")   # 匹配 _xxx_ 形式的属性名
            or name.startswith("__") and name.endswith("__")): 
            object.__setattr__(self, name, value)
        else:
            setattr(self._impl_cls, name, value)

    def __getattribute__(self, name):
        #print(name,name.startswith("_") or name.startswith("__") or name.startswith("required_") or name in ["validate","register"])
        if (name.startswith("required_")
            or name in backend_valid
            or name.startswith("_") and name.endswith("_")   # 匹配 _xxx_ 形式的属性名
            or name.startswith("__") and name.endswith("__")): 
            return object.__getattribute__(self, name)
        else:
            try:
                return getattr(self._instance, name)
            except AttributeError:
                return getattr(self._impl_cls, name)

class AudioBackend(Backend):
    required_methods = ['gen_all_shots_audio',["__init__",("data",StoryData),("voice",str)]]

class ImageBackend(Backend):
    required_methods = [
            'load_lora',
            ['getImageFromText',"prompt","width","height"], 
            'unload_lora',
            ['getImageFromImage',"image","prompt","width","height"]
        ]

class VideoBackend(Backend):
    required_methods = [['getVideoClips',"img","prompt","total_second","basename"],["__init__",("outputs",str)]]

class ChatBackend(Backend):
    required_methods = ['stream',"close"]
    required_attributes = ['isOpened',"model"]

class FrameProcessPlugin(Backend):
    required_methods = [['rend',("frame",Frame)],['__init__',("processor",AfterEffectProcessor)]]

if __name__ == '__main__':

    class testBackend(Backend):
        required_methods = [['stream',("conetent",str)]]

    class test:
        def __int__(self):
            print("init")

        def stream(self,conetent:str):
            print("say hello")
    test = testBackend(test)
