from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from typing import Annotated

import typer
import os
import json

device = "auto"

prompt = """
 Your task:
1. Generate 50 questions for distinguishing between the classes in
    a dataset with the requirements below.
2. Each question should be centered around visual concepts while
    slight deviation is acceptable. An example of a deviation would
    be about the environment.
3. Each question should have 5 answer options and each class can
    only have one correct answer option. It's best to maximize
    the number classes that each pick a different answer option.
4. Each question should contain "the class" in the question.
5. Questions should maximize the separation between classes like a
    decision tree maximizing entropy.
6. Use your understanding of all of the classes and their visual
    differences to create these questions.
7. Only output ALL of the questions and answer options.
8. Do not repeat questions.
9. Do not write code.
10. Do not include class names in the answer options.
The classes: {classes}
Output format:
- For each question, use the specific format:
	[Question]
	1. [Option 1]
…
- Do not add additional commentary.
- Do not include the square brackets in the answer.
- Do not number the questions.
"""

app = typer.Typer()


def generate_questions(model_id: str, dataset: str, temperature: float, max_tokens: int):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    classes_path = f"data/classes/{dataset}_classes.json"
    assert os.path.exists(classes_path)
    with open(classes_path, "r") as f:
        classes = json.load(f)

    output_dir = f"questions/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "qwen2.5-72b_50-5.txt")

    messages = [
        {
            "role": "user",
            "content": f"{prompt.format(classes=classes)}"
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text.strip())


@app.command()
def main(
    dataset: Annotated[str, typer.Option()],
    model_id: Annotated[str, typer.Option()] = "Qwen/Qwen2.5-72B-Instruct",
    temperature: Annotated[float, typer.Option()] = 0.7,
    max_tokens: Annotated[int, typer.Option()] = 8192,
):
    generate_questions(model_id, dataset, temperature, max_tokens)


if __name__ == "__main__":
    app()
