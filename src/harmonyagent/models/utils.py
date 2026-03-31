from harmonyagent.domain_model import Model
from harmonyagent.models.test_models import TapeModel, TapeModelConfig
from harmonyagent.models.vllm_raw import VllmRawModel, VllmRawModelConfig

MODEL_CLASS_MAPPING = {
    "vllmraw": VllmRawModel,
    "tape": TapeModel,
}

MODEL_CLASS_CONFIG_MAPPING = {
    "vllmraw": VllmRawModelConfig,
    "tape": TapeModelConfig,
}


def get_model_class(model_class: str) -> Model:
    return MODEL_CLASS_MAPPING[model_class]


def get_model_class_config(model_class: str) -> type:
    return MODEL_CLASS_CONFIG_MAPPING[model_class]
