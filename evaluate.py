import dataclasses
import json
import os
from typing import Annotated, Optional

import typer
import torch
from torch.utils.data import DataLoader

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from lavid import TrainConfig, get_kd_spec, kd_model_class, get_best_checkpoint
from lavid.data import BaseDataset
from lavid.models import CLIPFinetuneModel
from utils import (
    compute_lavid_spec,
    get_class_names,
    get_dataset,
)


def _from_dict(cls, d):
    init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
    return cls(**{k: v for k, v in d.items() if k in init_fields})


def _find_config(weights_path):
    base = weights_path if os.path.isdir(weights_path) else os.path.dirname(weights_path)
    p = os.path.join(base, "config.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"config.json not found at {p}")
    return p


def run_evaluate(cfg: TrainConfig, test_loader, model):
    model.eval()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=WandbLogger(project="default", name=cfg.wandb_name),
    )
    trainer.test(model, dataloaders=test_loader)


def load_and_test(ckpt, cfg, ds_spec, test_loader, logger=None):
    ckpt = get_best_checkpoint(ckpt)
    if cfg.model == "clip":
        model = CLIPFinetuneModel.load_from_checkpoint(ckpt, cfg=cfg, ds_spec=ds_spec)
        run_evaluate(cfg, test_loader, model)
    else:
        model_class = kd_model_class(cfg.kd)
        model = model_class.load_from_checkpoint(ckpt, cfg=cfg, ds_spec=ds_spec)
        if ds_spec.name == "waterbird":
            waterbird_test_acc(cfg, test_loader, model)
        else:
            run_evaluate(cfg, test_loader, model)


def waterbird_test_acc(cfg: TrainConfig, test_loader, model):
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=WandbLogger(project="default", name=cfg.wandb_name),
    )
    trainer.test(model, dataloaders=test_loader)

    mapping = {"0_0": 0, "0_1": 1, "1_0": 2, "1_1": 3}
    group_names = [
        "landbird_land",
        "landbird_water",
        "waterbird_land",
        "waterbird_water",
    ]

    all_preds = torch.cat([probs.argmax(dim=-1) for probs, _ in model.test_outputs])
    all_labels = torch.cat([labels for _, labels in model.test_outputs])
    all_group_ids = torch.tensor(
        [mapping[g] for batch in model.waterbird_test_groups for g in batch]
    )

    correct_mask = (all_preds == all_labels).float()
    correct = torch.zeros(4).scatter_add(0, all_group_ids, correct_mask)
    overall = torch.bincount(all_group_ids, minlength=4).float()

    group_accs = correct / overall
    log_dict = {f"test_acc_{group_names[i]}": group_accs[i].item() for i in range(4)}
    log_dict["test_acc_overall"] = (correct.sum() / overall.sum()).item()
    wandb.log(log_dict)


app = typer.Typer()


@app.command()
def main(
    ckpt: Annotated[str, typer.Option()],
    data_dir: Annotated[Optional[str], typer.Option()] = None,
):
    with open(_find_config(ckpt)) as f:
        saved = json.load(f)

    train_cfg = _from_dict(TrainConfig, saved)

    if data_dir is not None:
        train_cfg.data_dir = data_dir

    evaluate(ckpt, train_cfg)


def evaluate(ckpt: str, cfg: TrainConfig):
    _, _, test_split, ds_spec = get_dataset(cfg)

    test_dataset = BaseDataset(
        test_split,
        dataset_name=ds_spec.name,
        num_classes=ds_spec.num_classes,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=ds_spec.batch_size, pin_memory=True, num_workers=8
    )

    if cfg.model == "clip":
        cfg.class_names = get_class_names(ds_spec.name)
    if cfg.kd and get_kd_spec(cfg.kd).data_kind in ("lavid", "makd"):
        compute_lavid_spec(cfg, ds_spec)

    load_and_test(ckpt, cfg, ds_spec, test_loader)

    wandb.finish()


if __name__ == "__main__":
    app()
