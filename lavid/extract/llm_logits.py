from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration
)
import torch
from tqdm import trange
from typing import Annotated, Optional

import typer
import os
import json
from collections import defaultdict
from importlib.resources import files

device = "auto"

app = typer.Typer()


def option_number(
    model_id: str,
    dataset: str,
    output_file: str,
    input_file: Optional[str] = None,
    batch_size: int = 25,
    temperature: float = 1.0,
    imagenet_group: Optional[str] = None,
    alphabet: bool = False,
):
    max_num_options = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    _data = files("lavid.data")
    if imagenet_group:
        with (_data / "imagenet" / "imagenet_hierarchy.json").open("r") as f:
            classes = json.load(f)[imagenet_group]
    else:
        resource = _data / "classes" / f"{dataset}_classes.json"
        assert resource.is_file(), f"Classes file not found for dataset: {dataset}"
        with resource.open("r") as f:
            classes = json.load(f)

    assert os.path.exists(input_file)

    assert output_file[-5:] == ".json", "output file should be json type"

    classes = list(set(classes))

    output_data = defaultdict(list)
    questions = []
    with open(input_file, "r", encoding="utf-8") as file:
        cur_question = []
        lines = [line for line in file]
        lines.append("")
        for line in lines:
            line = line.strip()
            if line == "":
                if cur_question is None:
                    continue
                num_options = len(cur_question) - 1
                max_num_options = max(max_num_options, num_options)
                questions.append((num_options, " ".join(cur_question)))
                cur_question = []
            else:
                cur_question.append(line)

    if "llava" in model_id.lower():
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    data = []
    for class_name in classes:
        for num_options, question in questions:
            question = question.replace("class", class_name)
            if "caltech101" in dataset:
                question = question.replace("the object", class_name)
            elif "oxfordpets" in dataset:
                question = question.replace("the animal", class_name)
                question = question.replace("the breed", class_name)
            elif "stanfordcars" in dataset:
                question = question.replace("the car", class_name)

            ans_type = "number" if not alphabet else "letter"

            if "llava" in model_id.lower():
                prompt = f"USER: Answer only with the option {ans_type}. {question}\nASSISTANT:"
            else:
                prompt = {"role": "user", "content": f"Answer only with the option {ans_type}. No additional commentary. {question}"}

            data.append([class_name, prompt, num_options])

    token_ids = []

    if "llava" in model_id.lower():
        tokenizer = processor.tokenizer

    if max_num_options > 0:
        if alphabet:
            for num_option in range(max_num_options):
                token_ids.append(tokenizer.convert_tokens_to_ids(chr(num_option + 65)))
        else:
            for num_option in range(1, max_num_options + 1):
                token_ids.append(tokenizer.convert_tokens_to_ids(str(num_option)))
    else:
        token_ids = [tokenizer.convert_tokens_to_ids(option) for option in ["Yes", "No"]]

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    for i in trange(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        if "llava" in model_id.lower():
            prompts = [prompt for _, prompt, _ in batch]
            inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device=device)
        else:
            prompts = [[prompt] for _, prompt, _ in batch]
            processed_prompts = tokenizer.apply_chat_template(
                prompts,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(processed_prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1]
            for nq in range(len(logits)):
                cur_options_logit = []
                logit = logits[nq]
                num_options = batch[nq][2]
                for num_option in range(num_options):
                    cur_options_logit.append(logit[token_ids[num_option]])
                cur_options_logit = torch.tensor(cur_options_logit)
                prob = torch.softmax(cur_options_logit / temperature, dim=-1)
                cur_class = batch[nq][0]
                output_data[cur_class].append(prob.tolist())

    with open(output_file, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, ensure_ascii=False, indent=4)


@app.command()
def main(
    model_id: Annotated[str, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    output_file: Annotated[str, typer.Option()],
    input_file: Annotated[Optional[str], typer.Option()] = None,
    batch_size: Annotated[int, typer.Option()] = 25,
    temperature: Annotated[float, typer.Option()] = 1.0,
    imagenet_group: Annotated[Optional[str], typer.Option()] = None,
    alphabet: Annotated[bool, typer.Option("--alphabet", is_flag=True)] = False,
):
    option_number(model_id, dataset, output_file, input_file, batch_size, temperature, imagenet_group, alphabet)


if __name__ == "__main__":
    app()
