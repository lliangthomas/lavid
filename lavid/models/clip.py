import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import clip
from lavid.config import DatasetSpec, TrainConfig


class ClassificationHead(nn.Linear):
    def __init__(self, normalize, weights):
        out_size, in_size = weights.shape
        super().__init__(in_size, out_size)
        self.normalize = normalize
        self.weight = nn.Parameter(weights.clone())
        self.bias = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class TextEmbeddingClassifier(nn.Module):
    def __init__(self, clip_model, class_names, device="cuda"):
        super().__init__()
        self.class_names = class_names
        self.device = device

        with torch.no_grad():
            weights = []
            for name in class_names:
                tokens = clip.tokenize("A photo of a {}".format(name)).to(device)
                emb = clip_model.encode_text(tokens)[0]
                weights.append(F.normalize(emb, dim=-1))
            weights = (torch.stack(weights) * clip_model.logit_scale.exp()).float()

        self.classifier = ClassificationHead(normalize=True, weights=weights)

    def forward(self, visual_features):
        return self.classifier(visual_features)


class CLIPFinetuneModel(pl.LightningModule):
    def __init__(self, cfg: TrainConfig, ds_spec: DatasetSpec):
        super().__init__()
        self.cfg = cfg
        self.ds_spec = ds_spec
        self.criterion = nn.CrossEntropyLoss()
        self.test_outputs = []
        self.test_results = {}

        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)

        self.text_classifier = TextEmbeddingClassifier(
            self.clip_model,
            cfg.class_names,
            device=self.device
        )

        self.feature2question = None

        if cfg.kd and "lavid" in cfg.kd:
            visual_dim = self.clip_model.visual.output_dim
            qa_total = cfg.qa_total_options if cfg.qa_total_options is not None else 0
            has_qa = qa_total > 0 if isinstance(qa_total, int) else len(qa_total) > 0
            if has_qa:
                if ds_spec.name == "imagenet":
                    self.feature2question = nn.ModuleDict()
                    for group in ds_spec.imagenet_groups:
                        self.feature2question[group] = nn.Linear(visual_dim, qa_total[group])
                else:
                    self.feature2question = nn.Linear(visual_dim, qa_total)

        for param in self.text_classifier.parameters():
            param.requires_grad_(False)

        for param in self.clip_model.ln_final.parameters():
            param.requires_grad_(True)
        for param in self.clip_model.visual.parameters():
            param.requires_grad_(True)

        for param in self.clip_model.transformer.parameters():
            param.requires_grad_(False)
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad_(False)
        self.clip_model.logit_scale.requires_grad_(False)

    def forward(self, x, feat_t=None, class_names=None, return_features=False):
        visual_features = self.clip_model.encode_image(x)

        text_logits = self.text_classifier(visual_features)

        lavid_output = None

        if self.feature2question is not None:
            if self.ds_spec.name == "imagenet" and class_names is not None:

                lavid_output = []
                for i, class_name in enumerate(class_names):
                    group = self.ds_spec.imagenet_cls2group[class_name]
                    lavid_output.append(self.feature2question[group](visual_features[i]))
                lavid_output = torch.stack(lavid_output)
            else:

                lavid_output = self.feature2question(visual_features)

        if return_features:
            return text_logits, lavid_output, visual_features

        if feat_t is None:
            return text_logits

        if lavid_output is not None:
            return text_logits, lavid_output
        
        return text_logits

    def training_step(self, batch, batch_idx):
        if self.cfg.kd and "lavid" in self.cfg.kd:
            if self.ds_spec.name == "imagenet":
                images, feat_t, labels, class_names = batch
                text_logits, lavid_output = self.forward(images, feat_t, class_names)
            else:
                images, feat_t, labels = batch
                text_logits, lavid_output = self.forward(images, feat_t)
        else:
            images, labels = batch
            text_logits = self.forward(images)
            lavid_output = None

        text_loss = self.criterion(text_logits, labels)
        text_acc = (text_logits.argmax(dim=1) == labels).float().mean()

        if lavid_output is not None:
            if feat_t.dtype != lavid_output.dtype:
                feat_t = feat_t.to(lavid_output.dtype)
            lavid_loss = F.mse_loss(lavid_output, feat_t)
            total_loss = self.cfg.gamma * text_loss + self.cfg.beta * lavid_loss
            self.log('train_text_loss', text_loss, on_step=True, on_epoch=True)
            self.log('train_lavid_loss', lavid_loss, on_step=True, on_epoch=True)
        else:
            total_loss = text_loss

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_text_acc', text_acc, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        text_logits = self.forward(images)

        text_loss = self.criterion(text_logits, labels)

        text_preds = torch.argmax(text_logits, dim=1)
        text_acc = (text_preds == labels).float().mean()


        self.log('val_loss', text_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_text_acc', text_acc, on_step=False, on_epoch=True, prog_bar=True)

        return text_loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        text_logits = self.forward(images)

        loss = self.criterion(text_logits, labels)

        preds = torch.argmax(text_logits, dim=1)
        acc = (preds == labels).float().mean()


        probs = F.softmax(text_logits, dim=1)
        self.test_outputs.append((probs.cpu(), labels.cpu()))

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(params, lr=self.ds_spec.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.ds_spec.num_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
