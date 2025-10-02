import yaml

class Cfg:
    def __init__(self, data):
        for k, v in (data or {}).items():
            if isinstance(v, dict):
                v = Cfg(v)
            elif isinstance(v, list):
                v = [Cfg(i) if isinstance(i, dict) else i for i in v]
            setattr(self, k, v)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Cfg):
                v = v.to_dict()
            elif isinstance(v, list):
                v = [i.to_dict() if isinstance(i, Cfg) else i for i in v]
            result[k] = v
        return result

def load_cfg(path: str) -> Cfg:
    with open(path, "r") as f:
        return Cfg(yaml.safe_load(f))

def infer_generic(config, params):
    for key in config.generic.to_dict():
        if hasattr(params, key):
            setattr(params, key, getattr(config.generic, key))