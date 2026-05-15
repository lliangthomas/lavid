from torch.utils.data import Dataset
import pandas as pd
import scipy.io
import numpy as np
from PIL import Image

import os
import importlib.resources
from collections import defaultdict
import json
import pickle

BASE_DIR = importlib.resources.files('lavid.data')
CLASSES_DIR = BASE_DIR / "classes"

class CustomCaltechLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = os.path.join(root_dir, "101_ObjectCategories")
        # official dataset classes, not equal to caltech101_classes.json
        self.classes = ["Faces", "Faces_easy", "Leopards", "Motorbikes", "accordion", "airplanes", "anchor", "ant", "barrel", "bass", "beaver", "binocular", "bonsai", "brain", "brontosaurus", "buddha", "butterfly", "camera", "cannon", "car_side", "ceiling_fan", "cellphone", "chair", "chandelier", "cougar_body", "cougar_face", "crab", "crayfish", "crocodile", "crocodile_head", "cup", "dalmatian", "dollar_bill", "dolphin", "dragonfly", "electric_guitar", "elephant", "emu", "euphonium", "ewer", "ferry", "flamingo", "flamingo_head", "garfield", "gerenuk", "gramophone", "grand_piano", "hawksbill", "headphone", "hedgehog", "helicopter", "ibis", "inline_skate", "joshua_tree", "kangaroo", "ketch", "lamp", "laptop", "llama", "lobster", "lotus", "mandolin", "mayfly", "menorah", "metronome", "minaret", "nautilus", "octopus", "okapi", "pagoda", "panda", "pigeon", "pizza", "platypus", "pyramid", "revolver", "rhino", "rooster", "saxophone", "schooner", "scissors", "scorpion", "sea_horse", "snoopy", "soccer_ball", "stapler", "starfish", "stegosaurus", "stop_sign", "strawberry", "sunflower", "tick", "trilobite", "umbrella", "watch", "water_lilly", "wheelchair", "wild_cat", "windsor_chair", "wrench", "yin_yang"]

        self.imgs = []
        cls2idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        self.clsidx2img = defaultdict(list)

        self.parsed_cls = list(json.load((CLASSES_DIR / "caltech101_classes.json").open("r")))

        self.idx2cls = {idx + 1: self.parsed_cls[idx] for idx in range(len(self.parsed_cls))}
        self.idx2cls[0] = self.parsed_cls[0]
        self.labels = []
        
        for class_name in self.classes:
            cls_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(cls_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg')):
                    idx = cls2idx[class_name]
                    path = os.path.join(cls_dir, file_name)
                    self.clsidx2img[idx].append(path)
                    self.imgs.append([path, idx])
                    self.labels.append(idx)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        return Image.open(img_path).convert('RGB'), img_path, label
    
class CustomCUB200Loader(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir

        image_class_labels_path = os.path.join(root_dir, "image_class_labels.txt")
        images_txt_path = os.path.join(root_dir, "images.txt")
        train_test_split_path = os.path.join(root_dir, "train_test_split.txt")
        classes_txt_path = os.path.join(root_dir, "classes.txt")

        for path in [image_class_labels_path, images_txt_path, train_test_split_path, classes_txt_path]:
            assert os.path.exists(path)
        
        self.images_df = pd.read_csv(images_txt_path, sep=" ", header=None, names=["img_id", "img_name"])
        self.labels_df = pd.read_csv(image_class_labels_path, sep=" ", header=None, names=["img_id", "label"])
        self.split_df = pd.read_csv(train_test_split_path, sep=" ", header=None, names=["img_id", "is_train"])
        self.classes_df = pd.read_csv(classes_txt_path, sep=" ", header=None, names=["class_id", "class_name"])

        merged = self.images_df.merge(self.labels_df, on="img_id")
        merged = merged.merge(self.split_df, on="img_id")

        if split == "train":
            self.imgs = merged[merged["is_train"] == 1]
        elif split == "test" or split == "val":
            self.imgs = merged[merged["is_train"] == 0]

        self.imgs.loc[:, "label"] = self.imgs["label"] - 1

        self.classes = list(json.load((CLASSES_DIR / "cub200_classes.json").open("r")))
        self.idx2cls = dict(zip(range(len(self.classes)), self.classes))
        self.labels = self.imgs["label"].tolist()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        row = self.imgs.iloc[idx]
        img_path = os.path.join(self.root_dir, "images", row["img_name"])
        label = row["label"]
        return Image.open(img_path).convert('RGB'), img_path, label

class CustomWaterbirdLoader(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        
        # Load metadata
        metadata_path = os.path.join(root_dir, "metadata.csv")
        assert os.path.exists(metadata_path), f"Metadata file not found at {metadata_path}"
        
        self.metadata_df = pd.read_csv(metadata_path)
        
        # Map split values: 0 - train, 1 - val, 2 - test
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_value = split_map.get(split)
        if split_value is None:
            raise ValueError(f"Split must be one of 'train', 'val', or 'test', got {split}")
        
        self.data = self.metadata_df[self.metadata_df['split'] == split_value]

        self.finegrained_classes = list(json.load((CLASSES_DIR / "cub200_classes.json").open("r")))
        self.idx2cls = {i: cls for i, cls in enumerate(self.finegrained_classes)}
        self.labels = self.data['y'].tolist()

        self.groups = []
        for _, row in self.data.iterrows():
            bird_type = row['y']
            place = row['place']
            group = f"{bird_type}_{place}"
            self.groups.append(group)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_filename = row['img_filename']
        img_path = os.path.join(self.root_dir, img_filename)
        
        label = row['y']
        group = self.groups[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        return image, img_path, label, group

class CustomFGVCAircraftLoader(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split

        images_dir = os.path.join(root_dir, "data", "images")
        annot_file = os.path.join(root_dir, "data", f"images_variant_{split}.txt")
        variants_file = os.path.join(root_dir, "data", "variants.txt")

        assert os.path.exists(images_dir), f"Missing images directory: {images_dir}"
        assert os.path.exists(annot_file), f"Missing annotations file: {annot_file}"
        assert os.path.exists(variants_file), f"Missing variants file: {variants_file}"

        with open(variants_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.cls2idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx2cls = self.classes
        
        self.imgs = []
        self.labels = []
        with open(annot_file, "r") as f:
            for line in f:
                img_name, class_name = line.strip().split(" ", 1)
                class_id = self.cls2idx[class_name]
                img_path = os.path.join(images_dir, img_name + ".jpg")
                assert os.path.exists(img_path)
                self.imgs.append([img_path, class_id])
                self.labels.append(class_id)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        return Image.open(img_path).convert('RGB'), img_path, label
    
class Custom102FlowersLoader(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split

        labels = scipy.io.loadmat(os.path.join(root_dir, "imagelabels.mat"))["labels"][0]
        setid = scipy.io.loadmat(os.path.join(root_dir, "setid.mat"))

        if split == "train":
            split_ids = setid["trnid"][0]
        elif split == "val":
            split_ids = setid["valid"][0]
        elif split == "trainval":
            split_ids = np.concatenate((setid["trnid"][0], setid["valid"][0]))
        elif split == "test":
            split_ids = setid["tstid"][0]
        elif split == "all":
            split_ids = np.concatenate((setid["trnid"][0], setid["valid"][0], setid["tstid"][0]))

        self.classes = list(json.load((CLASSES_DIR / "flowers_classes.json").open("r")))
        self.cls2idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.idx2cls = self.classes
        self.labels = []
        self.imgs = []
        for img_id in split_ids:
            img_name = f"image_{img_id:05d}.jpg"
            img_path = os.path.join(root_dir, "jpg", img_name)
            label = labels[img_id - 1] - 1  # Labels are 1-indexed
            assert os.path.exists(img_path)
            self.imgs.append((img_path, label))
            self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        return Image.open(img_path).convert('RGB'), img_path, label

class CustomOxfordPetsLoader(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        assert os.path.exists(os.path.join(root_dir, "images"))
        assert os.path.exists(os.path.join(root_dir, "annotations"))
        
        self.classes = []
        self.class_dict = {}  # maps class_id to class_name
        
        trainval_path = os.path.join(root_dir, "annotations", "trainval.txt")
        assert os.path.exists(trainval_path)
        
        # unofficial class names
        # cleaned up from official class names
        self.idx2cls = list(json.load((CLASSES_DIR / "oxfordpets_classes.json").open("r")))
        self.imgs = []
        self.labels = []
        
        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root_dir, "annotations", split_file)
        assert os.path.exists(split_path)
        
        with open(split_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    # Oxford Pets labels are 1-indexed
                    class_id = int(parts[1]) - 1
                    img_path = os.path.join(root_dir, "images", f"{img_name}.jpg")
                    if os.path.exists(img_path):
                        self.imgs.append([img_path, class_id])
                        self.labels.append(class_id)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        return Image.open(img_path).convert('RGB'), img_path, label

class CustomStanfordCarsLoader(Dataset):
    def __init__(self, root_dir, split):
        # root_dir should be data/StanfordCars which has cars_train and cars_test directories
        self.root_dir = root_dir
        assert os.path.exists(os.path.join(root_dir, "cars_train"))
        assert os.path.exists(os.path.join(root_dir, "cars_test"))
        
        train_annot_path = os.path.join(root_dir, "devkit/cars_train_annos.mat")
        test_annot_path = os.path.join(root_dir, "cars_test_annos_withlabels.mat")
        meta_path = os.path.join(root_dir, "devkit/cars_meta.mat")
        
        self.meta = scipy.io.loadmat(meta_path)
        self.classes = [x[0] for x in self.meta["class_names"][0]] # 1 indexed and map directly to class name 
        self.idx2cls = {idx: self.classes[idx] for idx in range(len(self.classes))}
        self.labels = []
        
        self.imgs = []

        if split == "train":
            train_annot = scipy.io.loadmat(train_annot_path)
            for ann in train_annot["annotations"][0]:
                img_name = ann[5][0]
                class_id = ann[4][0][0] - 1  # Stanford Cars labels start from 1
                img_path = os.path.join(root_dir, "cars_train", img_name)
                if os.path.exists(img_path):
                    self.imgs.append([img_path, class_id])
                    self.labels.append(class_id)
        elif split == "test":
            test_annot = scipy.io.loadmat(test_annot_path)
            for ann in test_annot["annotations"][0]:
                img_name = ann[5][0]
                class_id = ann[4][0][0] - 1  # Stanford Cars labels start from 1
                img_path = os.path.join(root_dir, "cars_test", img_name)
                if os.path.exists(img_path):
                    self.imgs.append([img_path, class_id])
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        return Image.open(img_path).convert('RGB'), img_path, label

class CustomCIFAR100Loader(Dataset):
    def __init__(self, root_dir, split="train"):
        # same as using cifar100 from torchvision
        self.root_dir = root_dir
        self.split = split

        batch_path = os.path.join(self.root_dir, split)
        with open(batch_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.labels = entry['fine_labels']
            self.filenames = entry['filenames']

        self.classes = list(json.load((CLASSES_DIR / "cifar100_classes.json").open("r")))

        self.idx2cls = {i: cls_name for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        arr = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(arr)

        filename = self.filenames[idx]
        label = self.labels[idx]

        return img, filename, label
    
class CustomImageNetLoader(Dataset):
    def __init__(self, root_dir, split="train", select_group=None):
        self.root_dir = root_dir
        self.split = split
        
        self.data_dir = os.path.join(root_dir, split)
        
        groups_file = BASE_DIR / "imagenet" / "imagenet_hierarchy.json"
        with groups_file.open('r') as f:
            self.group2cls = json.load(f)
            self.groups = self.group2cls.keys()
        meta_file = BASE_DIR / "imagenet" / "imagenet_class_index.json"
        with meta_file.open('r') as f:
            class_idx = json.load(f)
            # class_idx is a dict with format {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
        
        self.idx2cls = {int(idx): val[1] for idx, val in class_idx.items()}
        self.wnid2idx = {val[0]: int(idx) for idx, val in class_idx.items()}
        self.cls2idx = {val[1]: int(idx) for idx, val in class_idx.items()}
        self.cls2group = {}
        for key, val in self.group2cls.items():
            for cls_name in val:
                self.cls2group[cls_name] = key

        self.select_group_labels = set()
        if select_group:
            if select_group not in self.groups:
                raise ValueError(f"Group {select_group} not found in ImageNet groups.")
            for cls_name in self.group2cls[select_group]:
                wnid = self.cls2idx[cls_name]
                self.select_group_labels.add(wnid)

        self.wnid = sorted(os.listdir(self.data_dir))
        self.classes = self.cls2idx.keys()
        
        self.image_paths = []
        self.labels = []
        self.filenames = []
        
        for wnid in self.wnid:
            wnid_path = os.path.join(self.data_dir, wnid)
            if not os.path.isdir(wnid_path):
                continue
            label = self.wnid2idx.get(wnid)
            if select_group and label not in self.select_group_labels:
                continue

            for img_file in os.listdir(wnid_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(wnid_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.filenames.append(img_file)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        filename = self.filenames[idx]
        label = self.labels[idx]
        
        return img, filename, label
