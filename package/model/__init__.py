from .hpnet import HPNet
from .gp_baseline import GPBaseline

__factory = {
    'hpnet': HPNet,
    'gp_baseline': GPBaseline,
}


def create_model(cfg):
    return __factory[cfg.name](cfg)