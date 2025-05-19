class Style(str):
    base = ""
    frontView = "plain background,front view,full body"

    def __init_subclass__(cls):
        super().__init_subclass__()
        if "base" not in cls.__dict__:
            raise NotImplementedError(f"Class {cls.__name__} must override 'base'.")
    # 自动加载所有 str 类型属性（排除 base 和 frontView），并拼接
    def _load_all_styles(self, args=None):
        cls = self.__class__
        components = list(args or [])
        for attr in dir(cls):
            if attr.startswith("_") or attr in {"base", "frontView"}:
                continue
            value = getattr(cls, attr)
            if isinstance(value, str):
                components.append(value)
        return self.load(*components)

    @property
    def styleFrontView(self):
        return self._load_all_styles(args=[self.__class__.base, self.__class__.frontView, self])

    @property
    def style(self):
        return self._load_all_styles(args=[self.__class__.base, self])

    def load(self, *argvs):
        return self.__class__(",".join(argvs))

    def __add__(self, other: str):
        return self.__class__((str(self) + ("," if self else "") + other).replace(".", ""))

    def __repr__(self):
        return f"Style({super().__str__()})"

class Pro3DModel(Style):
    base = "professional 3d model"
    render = "octane render,highly detailed,volumetric,dramatic lighting"
    pixar = "pixar animation"
    dream = "dreamWorks"


if __name__ == "__main__":
    p = Pro3DModel("a warrior")
    print("styleFrontView:", p.styleFrontView)
    print("style:", p.style)