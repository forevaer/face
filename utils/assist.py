import os
import torch
from config import config


def load_model_dict(model: torch.nn.Module):
    """
    自动加载训练数据
    """
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path))
    return model
