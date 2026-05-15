import dataclasses
import json
import os
from datetime import datetime
from typing import Annotated, Optional

import typer

from torch.utils.data import DataLoader

import wandb
from wandb.util import generate_id

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.fabric.utilities.seed import seed_everything

from lavid import (
    TrainConfig,
    get_best_checkpoint,
    get_kd_spec,
    kd_model_class,
    STUDENT_FEATURE_DIM,
    TEACHER_FEATURE_DIM,
)
from lavid.data import BaseDataset
from lavid.models import CLIPFinetuneModel
from evaluate import load_and_test
from utils import (
    compute_lavid_spec,
    get_class_names,
    get_dataset,
    load_lavid_logits_shape,
)


app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[str, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    model: Annotated[str, typer.Option()],
    precision: Annotated[str, typer.Option()] = "bf16-mixed",
    teacher: Annotated[Optional[str], typer.Option()] = None,
    teacher_layer: Annotated[Optional[int], typer.Option()] = None,
    trad_teacher_path: Annotated[Optional[str], typer.Option()] = None,
    kd: Annotated[Optional[str], typer.Option()] = None,
    feature_type: Annotated[str, typer.Option()] = "",
    seed: Annotated[Optional[int], typer.Option()] = None,
    wandb_name: Annotated[Optional[str], typer.Option()] = None,
    imagenet_group: Annotated[Optional[str], typer.Option()] = None,
    pretrained: Annotated[bool, typer.Option("--pretrained", is_flag=True)] = False,
    logit_stand: Annotated[bool, typer.Option("--logit-stand", is_flag=True)] = False,
    gamma: Annotated[float, typer.Option("-r", "--gamma")] = 1.0,
    beta: Annotated[float, typer.Option("-b", "--beta")] = 0.0,
    crd_feat_dim: Annotated[int, typer.Option()] = 128,
):
    train_cfg = TrainConfig(
        data_dir=data_dir, dataset=dataset, model=model,
        precision=precision, teacher=teacher, teacher_layer=teacher_layer,
        teacher_path=trad_teacher_path, kd=kd, feature_type=feature_type,
        seed=seed, wandb_name=wandb_name,
        imagenet_group=imagenet_group,
        pretrained=pretrained,
        logit_stand=logit_stand, gamma=gamma, beta=beta,
        crd_feat_dim=crd_feat_dim,
    )
    train(train_cfg)


def train(cfg: TrainConfig):
    wandb_logger = WandbLogger(
        name=cfg.wandb_name, log_model=False
    )

    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{cfg.wandb_name}-{timestamp}-{generate_id()}"

    os.makedirs(output_dir, exist_ok=True)

    train_split, val_split, test_split, ds_spec = get_dataset(cfg)

    if cfg.model == "clip":
        cfg.class_names = get_class_names(ds_spec.name)

        if cfg.kd and "lavid" in cfg.kd:
            cfg.qa_total_options = load_lavid_logits_shape(
                ds_spec.name, cfg.teacher, cfg.feature_type
            )

    if ds_spec.name == "imagenet" and cfg.kd == "lavid":
        cfg.kd = "lavid_imagenet"

    train_dataset = BaseDataset(
        train_split,
        dataset_name=ds_spec.name,
        num_classes=ds_spec.num_classes,
        split="train",
        teacher=cfg.teacher,
        teacher_layer=cfg.teacher_layer,
        kd=cfg.kd,
        k=4096,
        feature_type=cfg.feature_type,
    )
    val_dataset = BaseDataset(
        val_split,
        dataset_name=ds_spec.name,
        num_classes=ds_spec.num_classes,
        split="val",
    )
    test_dataset = BaseDataset(
        test_split,
        dataset_name=ds_spec.name,
        num_classes=ds_spec.num_classes,
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=ds_spec.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ds_spec.batch_size, pin_memory=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=ds_spec.batch_size, pin_memory=True, num_workers=8
    )

    if cfg.model == "clip":
        model = CLIPFinetuneModel(cfg, ds_spec)
    else:
        cfg.s_dim = STUDENT_FEATURE_DIM.get(cfg.model, -1)
        cfg.t_dim = TEACHER_FEATURE_DIM.get(cfg.teacher, 2560)
        cfg.n_data = len(train_dataset)

        if get_kd_spec(cfg.kd).data_kind in ("lavid", "makd"):
            compute_lavid_spec(cfg, ds_spec)

        if cfg.teacher_path:
            cfg.teacher_path = get_best_checkpoint(cfg.teacher_path)

        model = kd_model_class(cfg.kd)(cfg, ds_spec)

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    wandb_logger.experiment.config.update(dataclasses.asdict(cfg))

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=ds_spec.num_epochs,
        accelerator="gpu",
        devices=ds_spec.devices,
        strategy="ddp_find_unused_parameters_true" if ds_spec.devices > 1 else "auto",
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        default_root_dir=output_dir,
        check_val_every_n_epoch=5,
        precision=cfg.precision,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.global_rank == 0:
        ckpt = get_best_checkpoint(output_dir)
        load_and_test(ckpt, cfg, ds_spec, test_loader, wandb_logger)

    wandb.finish()


if __name__ == "__main__":
    app()
