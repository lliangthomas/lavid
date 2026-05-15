from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
import torch
from torch.utils.data import Subset, random_split

from tqdm import trange

import os
from collections import defaultdict
from typing import Annotated

import typer

from lavid.data.custom_loaders import (
    CustomCaltechLoader, CustomCUB200Loader, CustomFGVCAircraftLoader, Custom102FlowersLoader,
    CustomOxfordPetsLoader, CustomStanfordCarsLoader,
)

device = "cuda"

app = typer.Typer()


def extract_vlm_feature(model_id, output_path, data_dir):
    if "llava-v1.6" in model_id:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
    elif "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
    elif "instructblip" in model_id:
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
    dataset_name = data_dir.split('/')[-1]

    if dataset_name == "caltech101":
        caltech_dataset = CustomCaltechLoader(root_dir=data_dir)
        dataset, _, _ = random_split(caltech_dataset, [4310, 0, 4367], generator=torch.Generator().manual_seed(27))
    elif dataset_name == "cub200":
        dataset = CustomCUB200Loader(root_dir=data_dir, split="train")
    elif dataset_name == "fgvc_aircraft":
        dataset = CustomFGVCAircraftLoader(root_dir=data_dir, split="trainval")
    elif dataset_name == "flowers":
        dataset = Custom102FlowersLoader(root_dir=data_dir, split="trainval")
    elif dataset_name == "oxfordpets":
        dataset = CustomOxfordPetsLoader(root_dir=data_dir, split="train")
    elif dataset_name == "stanfordcars":
        dataset = CustomStanfordCarsLoader(root_dir=data_dir, split="train")

    idx2cls = dataset.dataset.idx2cls if isinstance(dataset, Subset) else dataset.idx2cls

    layers = [-1, -6, -12, -18, -24, -30]

    for i in trange(0, len(dataset)):
        all_features = defaultdict(list) # for dynamic number of hidden layers
        img, img_path, label = dataset[i]
        cls_name = idx2cls[label]
        questions = [f"Is there a {cls_name} in this image?"]

        if "llava-v1.6-mistral" in model_id:
            prompts = [f"[INST] <image>\n{question}[/INST]" for question in questions]
        elif "llava-v1.6-vicuna" in model_id:
            prompts = [
                f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                for question in questions
            ]
        elif "blip" in model_id:
            prompts = [f"Question: {question} Answer:" for question in questions]
        else:
            prompts = [
                f"USER: <image>\n{question}\nASSISTANT:" for question in questions
            ]

        inputs = processor(
            text=prompts, images=img, padding=True, return_tensors="pt"
        ).to(device=device)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            if "blip" in model_id:
                outputs = outputs.language_model_outputs
            for layer in layers:
                hidden_states = outputs.hidden_states[layer]
                last_feature = hidden_states[:, -1, :].unsqueeze(1)
                avg_feature = hidden_states.mean(dim=1).unsqueeze(1)
                feature = torch.cat([last_feature, avg_feature], dim=1)
                all_features[layer] = feature.cpu()

            prefix_shard = f"{output_path}/{img_path}"
            dirname = os.path.dirname(prefix_shard)
            os.makedirs(dirname, exist_ok=True)
            torch.save(all_features, f"{prefix_shard}.pt")


@app.command()
def main(
    output_path: Annotated[str, typer.Option()],
    data_dir: Annotated[str, typer.Option()],
    model_id: Annotated[str, typer.Option()] = "llava-hf/llava-1.5-7b-hf",
):
    extract_vlm_feature(model_id, output_path, data_dir)


if __name__ == "__main__":
    app()
