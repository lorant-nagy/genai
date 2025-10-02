REGISTRY = {}

def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls

def build(cls_name: str, params: dict):
    Cls = REGISTRY[cls_name]  # KeyError if missing (fine)
    return Cls(**params)