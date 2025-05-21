import pkgutil
import importlib
import inspect

__all__ = []

# 遍历当前包下的所有模块
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    # 遍历模块内的成员，寻找符合命名规则的类（比如以 Engine 结尾）
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.endswith("Engine") and obj.__module__ == module.__name__:
            globals()[name] = obj
            __all__.append(name)