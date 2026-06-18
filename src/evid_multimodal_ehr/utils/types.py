# Standard Library
from typing import Any, TypedDict


class HParamsDict(TypedDict):
    seed: int
    logger: bool
    devices: str
    batch_size: int
    lr: float
    max_epochs: int
    comments: str

    # Dataset arguments
    outcome: str
    n_class: int
    data_path: str
    structured_d_in_ls: Any
    class_weight: Any
    cv_split: int

    # Model Arguments
    model: str
    pretrained_model: str
    prototype_dim: int
    n_blocks: int
    structured_d_hidden: int
    notes_d_hidden: int
    alpha1: int | float
    alpha2: int | float
    dropout: float
