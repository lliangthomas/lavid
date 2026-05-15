import json
from typing import Tuple

import torch
from torch.utils.data import random_split

from lavid import (
    DatasetSpec,
    TrainConfig,
)
from lavid.data import (
    Custom102FlowersLoader,
    CustomCaltechLoader,
    CustomCUB200Loader,
    CustomFGVCAircraftLoader,
    CustomImageNetLoader,
    CustomOxfordPetsLoader,
    CustomStanfordCarsLoader,
    CustomWaterbirdLoader,
)


def get_class_names(dataset_name: str) -> list[str]:
    fname = (
        "caltech101_classes_duplicate_face"
        if dataset_name == "caltech101"
        else f"{dataset_name}_classes"
    )
    with open(f"lavid/data/classes/{fname}.json", "r") as f:
        return json.load(f)


_DATASET_NUM_CLASSES = {
    "caltech101": 101,
    "stanfordcars": 196,
    "oxfordpets": 37,
    "cub200": 200,
    "waterbird": 2,
    "fgvc_aircraft": 102,
    "flowers": 102,
    "imagenet": 1000,
}


def _build_splits(name: str, data_dir: str, imagenet_group):
    if name == "caltech101":
        ds = CustomCaltechLoader(root_dir=data_dir)
        train_split, _, test_split = random_split(
            ds, [4310, 0, 4367], generator=torch.Generator().manual_seed(27)
        )
        return train_split, test_split, test_split, None, None
    if name == "stanfordcars":
        return (
            CustomStanfordCarsLoader(root_dir=data_dir, split="train"),
            CustomStanfordCarsLoader(root_dir=data_dir, split="test"),
            CustomStanfordCarsLoader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "oxfordpets":
        return (
            CustomOxfordPetsLoader(root_dir=data_dir, split="train"),
            CustomOxfordPetsLoader(root_dir=data_dir, split="test"),
            CustomOxfordPetsLoader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "cub200":
        return (
            CustomCUB200Loader(root_dir=data_dir, split="train"),
            CustomCUB200Loader(root_dir=data_dir, split="test"),
            CustomCUB200Loader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "waterbird":
        return (
            CustomWaterbirdLoader(root_dir=data_dir, split="train"),
            CustomWaterbirdLoader(root_dir=data_dir, split="val"),
            CustomWaterbirdLoader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "fgvc_aircraft":
        return (
            CustomFGVCAircraftLoader(root_dir=data_dir, split="trainval"),
            CustomFGVCAircraftLoader(root_dir=data_dir, split="test"),
            CustomFGVCAircraftLoader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "flowers":
        return (
            Custom102FlowersLoader(root_dir=data_dir, split="trainval"),
            Custom102FlowersLoader(root_dir=data_dir, split="test"),
            Custom102FlowersLoader(root_dir=data_dir, split="test"),
            None,
            None,
        )
    if name == "imagenet":
        train_split = CustomImageNetLoader(root_dir=data_dir, split="train", select_group=imagenet_group)
        val_split = CustomImageNetLoader(root_dir=data_dir, split="val", select_group=imagenet_group)
        test_split = CustomImageNetLoader(root_dir=data_dir, split="val", select_group=imagenet_group)
        return train_split, val_split, test_split, train_split.groups, train_split.cls2group
    raise ValueError(f"Invalid dataset: {name}")


def _dataset_hyperparams(name: str) -> dict:
    if name == "imagenet":
        devices = min(torch.cuda.device_count(), 8)
        return dict(
            lr=0.2,
            batch_size=512 // devices,
            num_epochs=100,
            decay_epochs=[30, 60, 90],
            weight_decay=1e-4,
            devices=devices,
        )
    
    return dict(
        lr=0.01,
        batch_size=16,
        num_epochs=240,
        decay_epochs=[150, 180, 210],
        weight_decay=5e-4,
        devices=1,
    )


def _model_overrides(model: str, hp: dict) -> dict:
    if model == "vit-b16":
        devices = min(torch.cuda.device_count(), 8)
        hp = dict(hp)
        hp.update(devices=devices, batch_size=512 // devices, num_epochs=500)
    elif model == "clip":
        hp = dict(hp)
        hp.update(lr=1e-5, batch_size=32, num_epochs=75, decay_epochs=[])
    return hp


def get_dataset(cfg: TrainConfig) -> Tuple[object, object, object, DatasetSpec]:
    name = cfg.dataset
    if name not in _DATASET_NUM_CLASSES:
        raise ValueError(f"Invalid dataset: {name}")

    train_split, val_split, test_split, groups, cls2group = _build_splits(
        name, cfg.data_dir, cfg.imagenet_group
    )

    hp = _model_overrides(cfg.model, _dataset_hyperparams(name))

    spec = DatasetSpec(
        name=name,
        num_classes=_DATASET_NUM_CLASSES[name],
        lr=hp["lr"],
        batch_size=hp["batch_size"],
        num_epochs=hp["num_epochs"],
        decay_epochs=list(hp["decay_epochs"]),
        weight_decay=hp["weight_decay"],
        devices=hp["devices"],
        imagenet_groups=groups,
        imagenet_cls2group=cls2group,
    )
    return train_split, val_split, test_split, spec


def load_lavid_logits_shape(
    dataset_name: str,
    teacher: str,
    feature_type: str,
    groups=None,
):
    if groups is not None:
        out = {}
        for group in groups:
            with open(
                f"questions/{dataset_name}/{group}/{teacher}_{feature_type}.json", "r"
            ) as f:
                logits = json.load(f)
            example = list(logits.values())[0]
            qa_shape = [len(example[i]) for i in range(len(example))]
            out[group] = sum(qa_shape)
        return out

    with open(
        f"questions/{dataset_name}/logits/{teacher}_{feature_type}.json", "r"
    ) as f:
        logits = json.load(f)
    example = list(logits.values())[0]
    if isinstance(example[0], list):
        qa_shape = [len(example[i]) for i in range(len(example))]
        return sum(qa_shape)
    return len(example)


def compute_lavid_spec(cfg: TrainConfig, ds_spec: DatasetSpec):
    groups = ds_spec.imagenet_groups if ds_spec.name == "imagenet" else None
    cfg.qa_total_options = load_lavid_logits_shape(ds_spec.name, cfg.teacher, cfg.feature_type, groups=groups)
    cfg.concat_options = cfg.qa_total_options if cfg.kd == "makd" and isinstance(cfg.qa_total_options, int) else 0


