import os
import torch
import torch.nn as nn
from .config import Config
from typing import Optional

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

class BaseModel(nn.Module):
    def __init__(self, config: Config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, Config):
            raise ValueError(f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class")
        self.config = config
    
    @classmethod
    def from_pretranied(cls, pretrained_path):
        if os.path.isfile(pretrained_path):
            raise Exception(f"Provided path ({pretrained_path}) should be a directory, not a file")

        config = cls.config_class.from_json_file(os.path.join(pretrained_path, CONFIG_NAME))
        state_dict = torch.load(os.path.join(pretrained_path, WEIGHTS_NAME), map_location="cpu")
        model = cls(config)
        model.load_state_dict(state_dict)
        return model

    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class.
        """
        raise NotImplementedError(f"Make sure `_init_weights` is implemented for {self.__class__}")
    
    def save_pretrained(
        self,
        save_directory: str,
        state_dict: Optional[dict] = None
    ):
        if os.path.isfile(save_directory):
            raise Exception(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        
        model_to_save = unwrap_model(self)
        model_to_save.config.save_pretrained(save_directory)
        if state_dict is None:
            state_dict = model_to_save.state_dict()
        torch.save(state_dict, os.path.join(save_directory, WEIGHTS_NAME))


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model