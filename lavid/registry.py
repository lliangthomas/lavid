from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KDSpec:
    name: Optional[str]
    model_cls: str
    data_kind: str
    mlkd_augment: bool = False
    traditional_inner: Optional[str] = None


KD_REGISTRY: dict[Optional[str], KDSpec] = {
    None: KDSpec(None, "BaselineModel", "plain"),
    "mllm_mse": KDSpec("mllm_mse", "KDMSEModel", "mllm_mse"),
    "mllm_crd": KDSpec("mllm_crd", "KDCRDModel", "mllm_crd"),
    "makd": KDSpec("makd", "KDMaKDModel", "makd"),
    "lavid": KDSpec("lavid", "KDLaViDModel", "lavid"),
    "lavid_imagenet": KDSpec("lavid_imagenet", "KDLaViDImageNetModel", "lavid"),
    "kd": KDSpec("kd", "KDTraditionalModel", "plain", traditional_inner="kd"),
    "dkd": KDSpec("dkd", "KDTraditionalModel", "plain", traditional_inner="dkd"),
    "mlkd": KDSpec(
        "mlkd", "KDTraditionalModel", "plain", mlkd_augment=True, traditional_inner="mlkd"
    ),
    "rkd": KDSpec("rkd", "KDTraditionalModel", "plain", traditional_inner="rkd"),
    "lavid_mlkd": KDSpec(
        "lavid_mlkd",
        "KDLaViDTraditionalModel",
        "lavid",
        mlkd_augment=True,
        traditional_inner="mlkd",
    ),
}


def get_kd_spec(name: Optional[str]) -> KDSpec:
    if name not in KD_REGISTRY:
        raise ValueError(f"Unknown kd method: {name!r}")
    return KD_REGISTRY[name]


def kd_model_class(name: Optional[str]):
    from lavid.models import distill_models

    return getattr(distill_models, get_kd_spec(name).model_cls)


STUDENT_FEATURE_DIM: dict[str, int] = {
    "resnet18": 512,
    "mobilenetv2": 1280,
    "shufflenetv2": 1024,
    "vit-b16": 768,
}

TEACHER_FEATURE_DIM: dict[str, int] = {
    "llava1.5": 4096,
}
