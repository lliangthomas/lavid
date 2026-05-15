import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2
import torchvision.transforms as transforms
import os
import json
from lavid.data.mlkd_augment import RandAugment, MultipleApply
from lavid.registry import get_kd_spec

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class BaseDataset(Dataset):
    def __init__(
        self, 
        dataset,
        dataset_name,
        num_classes,
        feature_type=None,
        split="train", 
        teacher=None,
        teacher_layer=-1, 
        last_token=True, 
        kd=None, 
        k=4096,
    ):
        self.dataset = dataset
        self.idx2cls = self.dataset.dataset.idx2cls if isinstance(self.dataset, Subset) else self.dataset.idx2cls

        if not isinstance(self.dataset, Subset):
            self.labels = dataset.labels
        else:
            self.labels = [item[2] for item in dataset]
    
        self.split = split
        self.teacher = teacher
        self.teacher_layer = teacher_layer
        self.last_token = last_token
        self.kd = kd
        self.kd_spec = get_kd_spec(kd)
        self.k = k
        self.feature_type = feature_type
        self.dataset_name = dataset_name

        self.transform = self._get_transforms()

        if self.kd_spec.data_kind == "mllm_crd":
            self._setup_crd(num_classes)
        elif self.kd_spec.data_kind == "lavid":
            self._setup_lavid_logits()
        elif self.kd_spec.data_kind == "makd":
            self._setup_makd_logits()

        if self.kd_spec.mlkd_augment:
            self._setup_mlkd()

    def _get_transforms(self):
        if self.split == "train":
            return v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD
                )
            ])
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
        
    def _setup_mlkd(self):
        train_transform_weak = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])
        train_transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

        self.transform = MultipleApply([train_transform_weak, train_transform_strong])

    def __len__(self):
        return len(self.dataset)

    def _setup_crd(self, num_classes):
        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(len(self.dataset)):
            self.cls_positive[self.dataset[i][2]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

    def _setup_lavid_logits(self):
        with open(f"questions/{self.dataset_name}/logits/{self.teacher}_{self.feature_type}.json", "r") as f:
            self.lavid_logits = json.load(f)

        for key in self.lavid_logits.keys():
            cur = []
            for q in self.lavid_logits[key]:
                if isinstance(q, list):
                    cur.extend(q)
                else:
                    cur.append(q)
            self.lavid_logits[key] = torch.tensor(cur)

    def _setup_makd_logits(self):
        with open(f"questions/{self.dataset_name}/logits/{self.teacher}_{self.feature_type}.json", "r") as f:
            self.makd_logits = json.load(f)

    def __getitem__(self, idx):
        if self.dataset_name == "waterbird":
            img, img_path, label, group = self.dataset[idx]
        else:
            img, img_path, label = self.dataset[idx]
        img = self.transform(img)
        
        if self.kd_spec.data_kind == "lavid":
            if self.dataset_name == "waterbird":
                class_name = self.idx2cls[int(img_path.split("/")[-2].split(".")[0]) - 1]
            else:
                class_name = self.idx2cls[label]
            feature = self.lavid_logits[class_name]

            if self.dataset_name == "imagenet":
                return img, feature, label, class_name

            return img, feature, label

        elif self.kd_spec.data_kind == "makd":
            if self.dataset_name == "caltech101":
                img_name = "/".join(img_path.split("/")[-2:])
            else:
                img_name = os.path.basename(img_path)
            feature = torch.tensor(self.makd_logits[img_name])
            return img, feature, label

        elif self.kd_spec.data_kind in ("mllm_crd", "mllm_mse"):
            feature = torch.load(f"feature/{self.feature_type}_{self.teacher}_{self.dataset_name}/{img_path}.pt", weights_only=False)

            gt_idx = 0
            token_idx = 0 if self.last_token else 1

            if self.kd_spec.data_kind == "mllm_crd":
                pos_idx = idx
                neg_idx = np.random.choice(self.cls_negative[label], self.k, replace=True)
                sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
                return img, feature[self.teacher_layer][gt_idx, token_idx], label, pos_idx, sample_idx

            return img, feature[self.teacher_layer][gt_idx, token_idx], label

        else:
            if self.dataset_name == "waterbird" and self.split != "train":
                return img, label, group
            return img, label