import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from lavid.models.distill.makd import MaKD
from lavid.models.distill.crd.criterion import CRDLoss

from types import SimpleNamespace
from lavid.config import DatasetSpec, TrainConfig
from lavid.utils import load_teacher

from lavid.models.resnet import resnet18, resnet50
from lavid.models.mobilenetv2 import mobilenet_v2
from lavid.models.shufflenetv2 import shufflenet_v2_x1_0
from torchvision.models import vit_b_16

from lavid.models.distill.dkd import DKD
from lavid.models.distill.mlkd import MLKD
from lavid.models.distill.rkd import RKD
from lavid.models.distill.kd import KD


def _build_kd_module(model_s, model_t, kd_name, logit_stand):
    if kd_name == "dkd":
        cfg = SimpleNamespace(T=4, ALPHA=1.0, BETA=2.0, WARMUP=1, CE_WEIGHT=1.0)
        return DKD(model_s, model_t, cfg), cfg
    if kd_name == "mlkd":
        cfg = SimpleNamespace(T=1, ALPHA=0.5, CE_WEIGHT=0.5, LOGIT_STAND=logit_stand)
        return MLKD(model_s, model_t, cfg), cfg
    if kd_name == "rkd":
        return RKD(model_s, model_t), None
    if kd_name == "kd":
        return KD(model_s, model_t), None
    raise ValueError(f"Unknown KD method: {kd_name}")

def vit_extractor(model, x):
    x = model._process_input(x)

    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    features = model.encoder(x)

    feature = features[:, 0]
    logits = model.heads(feature)

    return feature, logits


class BaselineModel(pl.LightningModule):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__()
        self.cfg = cfg
        self.ds_spec = ds_spec
        self.criterion = nn.CrossEntropyLoss()
        self.test_outputs = []
        self.waterbird_test_groups = []

        out_dim = ds_spec.num_classes + cfg.concat_options
        if cfg.model == "resnet18":
            self.model1 = resnet18(num_classes=out_dim)
        elif cfg.model == "mobilenetv2":
            self.model1 = mobilenet_v2(num_classes=out_dim)
        elif cfg.model == "shufflenetv2":
            self.model1 = shufflenet_v2_x1_0(num_classes=out_dim)
        elif cfg.model == "resnet50":
            if cfg.pretrained:
                self.model1 = resnet50(pretrained=cfg.pretrained, num_classes=1000)
                self.model1.fc = nn.Linear(self.model1.fc.in_features, out_dim)
            else:
                self.model1 = resnet50(num_classes=out_dim)
        elif cfg.model == "vit-b16":
            self.model1 = vit_b_16(weights='IMAGENET1K_V1')
            self.model1.heads = nn.Linear(self.model1.hidden_dim, out_dim)

    def _log_metric(self, name, value, on_step=True):
        self.log(name, value, on_step=on_step, on_epoch=True, prog_bar=True, sync_dist=True)

    def _log_train(self, loss, acc, **extra):
        for name, val in extra.items():
            self._log_metric(name, val)
        self._log_metric('train_loss', loss)
        self._log_metric('train_acc', acc, on_step=False)
        self.log('learning_rate', self.optimizer.param_groups[0]['lr'], prog_bar=True)

    def forward(self, x):
        logit_s = self.model1(x)
        return logit_s

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.ds_spec.name == "waterbird":
            x, y, group = batch
        else:
            x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.ds_spec.name == "waterbird":
            x, y, group = batch
            self.waterbird_test_groups.append(group)
        else:
            x, y = batch

        logits = self(x)

        probs = F.softmax(logits, dim=1)
        self.test_outputs.append([probs, y])

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.ds_spec.lr,
            momentum=0.9,
            weight_decay=self.ds_spec.weight_decay,
        )

        if self.cfg.scheduler == "multi_step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.ds_spec.decay_epochs,
                gamma=0.1
            )
        elif self.cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.ds_spec.num_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler}")

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class KDMSEModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.model_t = nn.Linear(cfg.t_dim, cfg.s_dim)
        self.mse_criterion = nn.MSELoss()

    def forward(self, x, feat_t=None):
        feat_s, logit_s = self.model1(x, is_feat=True)
        if feat_t is None:
            return logit_s
        feat_t = self.model_t(feat_t)
        return logit_s, feat_s, feat_t

    def training_step(self, batch, batch_idx):
        x, feat_t, y = batch
        logit_s, feat_s, feat_t = self(x, feat_t)

        cls_loss = self.criterion(logit_s, y)

        feat_s = feat_s[-1]

        if feat_t.dtype != feat_s.dtype:
            feat_t = feat_t.to(feat_s.dtype)

        feat_s = F.normalize(feat_s, p=2.0)
        feat_t = F.normalize(feat_t, p=2.0)

        mse_loss = self.mse_criterion(feat_s, feat_t)
        loss = self.cfg.gamma * cls_loss + self.cfg.beta * mse_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, mse_loss=mse_loss)
        return loss


class KDCRDModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.model_t = None
        self.encoder = None
        self.crd_criterion = CRDLoss(cfg)

    def forward(self, x, feat_t=None):
        feat_s, logit_s = self.model1(x, is_feat=True)
        if feat_t is None:
            return logit_s
        logit_t = self.model_t(feat_t) if self.model_t else None
        if self.encoder:
            feat_t = self.encoder(feat_t)
        return logit_s, feat_t, feat_s, logit_t

    def training_step(self, batch, batch_idx):
        x, feat_t, y, idx, contrast_idx = batch
        logit_s, feat_t, feat_s, logit_t = self(x, feat_t)

        feat_s = feat_s[-1]

        if feat_t.dtype != feat_s.dtype:
            feat_t = feat_t.to(feat_s.dtype)

        cls_loss = self.criterion(logit_s, y)
        crd_loss = self.crd_criterion(feat_s, feat_t, idx, contrast_idx)

        loss = self.cfg.gamma * cls_loss + self.cfg.beta * crd_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, crd_loss=crd_loss)
        return loss


class KDMaKDModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.makd_criterion = MaKD(ds_spec.num_classes)

    def forward(self, x, feat_t=None):
        logit_s = self.model1(x)
        if feat_t is None:
            return logit_s[:, :self.ds_spec.num_classes]
        return logit_s

    def training_step(self, batch, batch_idx):
        x, feat_t, y = batch
        logit_s = self(x, feat_t)

        cls_loss = self.criterion(logit_s[:, :self.ds_spec.num_classes], y)
        makd_loss = self.makd_criterion(logit_s, feat_t)
        loss = self.cfg.gamma * cls_loss + self.cfg.beta * makd_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, makd_loss=makd_loss)
        return loss

class KDLaViDModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.feature2question = nn.Linear(cfg.s_dim, cfg.qa_total_options)

    def forward(self, x, feat_t=None):
        if "vit" in self.cfg.model:
            feat_s, logit_s = vit_extractor(self.model1, x)
        else:
            feat_s, logit_s = self.model1(x, is_feat=True)
        if feat_t is None:
            return logit_s
        return logit_s, feat_s

    def training_step(self, batch, batch_idx):
        x, feat_t, y = batch
        logit_s, feat_s = self(x, feat_t)

        if "vit" not in self.cfg.model:
            feat_s = feat_s[-1]

        qoutput = self.feature2question(feat_s)
        lavid_loss = F.mse_loss(qoutput, feat_t)

        cls_loss = self.criterion(logit_s, y)
        loss = self.cfg.gamma * cls_loss + self.cfg.beta * lavid_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, lavid_loss=lavid_loss)
        return loss


class KDLaViDImageNetModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.feature2question = nn.ModuleDict()
        for group in ds_spec.imagenet_groups:
            self.feature2question[group] = nn.Linear(cfg.s_dim, cfg.qa_total_options[group])
        self.cls2group = ds_spec.imagenet_cls2group

    def forward(self, x, feat_t=None):
        feat_s, logit_s = self.model1(x, is_feat=True)
        if feat_t is None:
            return logit_s
        return logit_s, feat_s

    def training_step(self, batch, batch_idx):
        x, feat_t, y, class_names = batch
        logit_s, feat_s = self(x, feat_t)

        feat_s = feat_s[-1]

        lavid_loss = 0

        for i in range(len(x)):
            class_name = class_names[i]
            group = self.cls2group[class_name]
            qoutput = self.feature2question[group](feat_s[i])
            lavid_loss += F.mse_loss(qoutput, feat_t[i])
        lavid_loss /= len(x)

        cls_loss = self.criterion(logit_s, y)
        loss = self.cfg.gamma * cls_loss + self.cfg.beta * lavid_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, lavid_loss=lavid_loss)
        return loss


class KDTraditionalModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.teacher = load_teacher(cfg.teacher, cfg.teacher_path, ds_spec.num_classes)
        self.teacher.eval()

        self.kd_module, self.kd_config = _build_kd_module(
            self.model1, self.teacher, cfg.kd, cfg.logit_stand
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(dtype=torch.int64)

        if self.cfg.kd == "mlkd":
            logits, losses_dict = self.kd_module.forward_train(x[0], x[1], y, epoch=self.current_epoch)
        else:
            logits, losses_dict = self.kd_module.forward_train(x, y, epoch=self.current_epoch)
        loss = sum(losses_dict.values())
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, **{f'train/{k}': v for k, v in losses_dict.items()})
        return loss


class KDLaViDTraditionalModel(BaselineModel):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__(cfg, ds_spec)

        self.teacher = load_teacher(cfg.teacher, cfg.teacher_path, ds_spec.num_classes)
        self.teacher.eval()

        self.traditional_kd = cfg.kd.split("_")[-1] # get the traditional kd method

        self.kd_module, self.kd_config = _build_kd_module(
            self.model1, self.teacher, self.traditional_kd, cfg.logit_stand
        )

        self.feature2question = nn.Linear(cfg.s_dim, cfg.qa_total_options)

    def forward(self, x, feat_t=None):
        if "vit" in self.cfg.model:
            feat_s, logit_s = vit_extractor(self.model1, x)
        else:
            feat_s, logit_s = self.model1(x, is_feat=True)
        if feat_t is None:
            return logit_s
        return logit_s, feat_s

    def training_step(self, batch, batch_idx):
        x, feat_t, y = batch
        logit_s, feat_s = self(x[0], feat_t)

        if "vit" not in self.cfg.model:
            feat_s = feat_s[-1]

        if self.cfg.kd and "mlkd" in self.cfg.kd:
            _, other_losses_dict = self.kd_module.forward_train(x[0], x[1], y, epoch=self.current_epoch)
        else:
            _, other_losses_dict = self.kd_module.forward_train(x, y, epoch=self.current_epoch)
        other_loss = sum(other_losses_dict.values())

        qoutput = self.feature2question(feat_s)
        mse_loss = F.mse_loss(qoutput, feat_t)

        loss = self.cfg.gamma * other_loss + self.cfg.beta * mse_loss
        acc = (logit_s.argmax(dim=-1) == y).float().mean()
        self._log_train(loss, acc, **{f'{self.traditional_kd}_loss': other_loss, 'mse_loss': mse_loss})
        return loss
