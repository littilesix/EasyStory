import sys
import types

fake_pyrootutils = types.ModuleType("pyrootutils")
fake_pyrootutils.setup_root = lambda *args, **kwargs: None  # 替代函数为空操作
sys.modules["pyrootutils"] = fake_pyrootutils


import torchvision.transforms._functional_tensor as _ft


# 创建一个 ModuleType 对象
ft_module = types.ModuleType("torchvision.transforms.functional_tensor")

# 将 _functional_tensor 的所有属性复制过来
for attr in dir(_ft):
    setattr(ft_module, attr, getattr(_ft, attr))

# 注册为全局模块路径
sys.modules["torchvision.transforms.functional_tensor"] = ft_module