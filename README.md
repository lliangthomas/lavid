# Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledge (ICML 2026)
Thomas Shih-Chao Liang*, Zhuoran Yu*, Yong Jae Lee

## Installation

Install uv, and then run this command

```bash
uv sync
```

This creates a virtual environment and installs all dependencies including the `lavid` package.

## Dataset

The following datasets are supported:
- Caltech101
- StanfordCars
- OxfordPets
- CUB200
- Waterbird
- FGVC Aircraft
- 102 Flowers
- ImageNet

## Usage

### LaViD

```bash
uv run train.py \
    --data_dir $DATASET_PATH \
    --dataset $DATASET_NAME \
    --model resnet18 \
    --teacher qwen2.5-7b \
    --kd lavid \
    --gamma 1 \
    --beta 40 \
    --feature_type chat_50-5
```

### LaViD with Logit Standardization

```bash
uv run train.py \
    --data_dir $DATASET_PATH \
    --dataset $DATASET_NAME \
    --model resnet18 \
    --teacher qwen2.5-7b \
    --teacher_ckpt $RESNET50_CKPT \
    --kd lavid_mlkd \
    --gamma 1 \
    --beta 40 \
    --logit_stand \
    --feature_type chat_50-5
```

### Key Parameters

- `--dataset`: Dataset name (caltech101, stanfordcars, oxfordpets, cub200, waterbird, fgvc_aircraft, flowers, imagenet)
- `--model`: resnet18, mobilenetv2, shufflenetv2, vit-b16, clip
- `--teacher`: Teacher model
  - LaViD
    - qwen2.5-7b (or any other teacher model)
  - MaKD
    - internvl2
  - MLLM MSE and CRD
    - llava1.5
  - Traditional
    - resnet50
- `--teacher_layer`: For baseline comparison with CRD 
- `--trad_teacher_path`: Path to teacher model ckpt for lavid_mlkd or traditional kd
- `--kd`: Knowledge distillation method
  - mllm_mse
  - mllm_crd
  - makd
  - dkd
  - mlkd
  - rkd
  - kd
  - lavid
  - lavid_mlkd   
- `--feature_type`: Type of distillation feature
  - LaViD
    - Template: {llm generator}_{number of questions}-{number of answer options}
    - chat_50-5 is the default in the paper
- `--imagenet_group`: Specify ImageNet WordNet hierarchy synset
- `--logit_stand`: Toggle logit standardization for MLKD
- `--gamma`: Weight for cross entropy loss
- `--beta`: Weight for knowledge distillation loss
- `--crd_feat_dim`: For baseline comparison with CRD 

## Evaluation

Evaluate a trained checkpoint using `evaluate.py`:

```bash
uv run evaluate.py --ckpt $CHECKPOINT_PATH_FOLDER
```