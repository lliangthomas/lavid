from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
import torch
from PIL import Image
from tqdm import trange
from typing import Annotated

import typer
import glob
import os
import json

device = torch.device("cuda")

app = typer.Typer()


def extract_img_logits(model_id: str, dataset: str, batch_size: int):
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, device_map=device, torch_dtype=torch.bfloat16
    )

    dataset_to_image_folder_path = {
        "caltech101": "data/Caltech101/101_ObjectCategories/**/",
        "cub200": "data/cub200/images/**/",
        "oxfordpets": "data/oxfordpets/images/",
        "stanfordcars": "data/stanfordcars/cars_train/",
        "fgvc_aircraft": "data/fgvc_aircraft/data/images/",
        "flowers": "data/flowers/jpg/",
    }

    assert dataset in dataset_to_image_folder_path, f"Unknown dataset: {dataset}"
    image_folder_path = dataset_to_image_folder_path[dataset]

    multi_aspect_questions_path = f"makd/multi_aspect_question/{dataset}.txt"
    output_json_path = f"results/aspects/llava1.5_img_{dataset}.json"

    assert os.path.exists(multi_aspect_questions_path)

    output_data = {}
    multi_aspect_questions = []
    with open(multi_aspect_questions_path, "r", encoding="utf-8") as file:
        for row in file:
            multi_aspect_questions.append(row.strip())
    data = []
    for image_path in glob.glob(os.path.join(image_folder_path, "*.jpg"), recursive=True):
        split = image_path.split("/")[-2:]
        if split[0] == "BACKGROUND_Google":
            continue
        image_name = "/".join(split)
        for question in multi_aspect_questions:
            data.append([image_path, f"USER: <image>\n{question}\nASSISTANT:", image_name])

    tokenizer = processor.tokenizer
    yes_id = tokenizer.convert_tokens_to_ids("Yes")
    no_id = tokenizer.convert_tokens_to_ids("No")

    prev_image = None
    aspect_logits = []
    for i in trange(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        image_names = [image_name for _, _, image_name in batch]
        images = [Image.open(image_path) for image_path, _, _ in batch]
        prompts = [prompt for _, prompt, _ in batch]

        if image_names[0] != prev_image:
            if prev_image is not None:
                output_data[prev_image] = aspect_logits
            aspect_logits = []
            prev_image = image_names[0]

        inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device=device)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1]
            yes_logit = logits[:, yes_id].unsqueeze(1)
            no_logit = logits[:, no_id].unsqueeze(1)

            yesno_logit = torch.cat((yes_logit, no_logit), dim=1)
            softmax_probs = torch.softmax(yesno_logit, dim=1)
            aspect_logits = aspect_logits + softmax_probs[:, 0].tolist()

    output_data[prev_image] = aspect_logits
    with open(output_json_path, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, ensure_ascii=False, indent=4)


@app.command()
def main(
    model_id: Annotated[str, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    batch_size: Annotated[int, typer.Option()] = 25,
):
    extract_img_logits(model_id, dataset, batch_size)


if __name__ == "__main__":
    app()
