from typing import Any

from omegaconf import OmegaConf

DATASETS_CONFIG = "configs/datasets.yaml"


def load_conf(
    config_path: str,
    semantic_config_path: str | None = None,
    overrides: list[str] | None = None,
) -> Any:
    conf = OmegaConf.merge(OmegaConf.load(DATASETS_CONFIG), OmegaConf.load(config_path))
    if semantic_config_path is not None:
        conf = OmegaConf.merge(conf, {"SEMANTIC_MLLM": OmegaConf.load(semantic_config_path)})

    OmegaConf.set_struct(conf, True)  # freeze structure: unknown keys raise
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))
    return conf
