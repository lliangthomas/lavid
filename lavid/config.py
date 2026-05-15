from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    lr: float
    batch_size: int
    num_epochs: int
    decay_epochs: List[int]
    weight_decay: float
    devices: int
    imagenet_groups: Optional[List[str]] = None
    imagenet_cls2group: Optional[Dict[str, str]] = None


@dataclass
class TrainConfig:
    data_dir: str
    dataset: str
    model: str 
    scheduler: str = "multi_step"
    precision: str = "bf16-mixed"
    seed: Optional[int] = None
    pretrained: bool = False

    teacher: Optional[str] = None
    teacher_layer: Optional[int] = None
    teacher_path: Optional[str] = None
    kd: Optional[str] = None
    feature_type: str = ""
    logit_stand: bool = False
    gamma: float = 1.0
    beta: float = 0.0

    wandb_name: Optional[str] = None
    imagenet_group: Optional[str] = None
    crd_feat_dim: int = 128
    s_dim: int = 512
    t_dim: int = 2560
    n_data: int = 0
    class_names: Optional[List[str]] = field(default=None, init=False)
    qa_total_options: Optional[Union[int, Dict[str, int]]] = field(default=None, init=False)
    concat_options: int = field(default=0, init=False)
