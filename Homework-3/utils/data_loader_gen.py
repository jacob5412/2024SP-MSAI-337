import json

import torch


def encode_instance(fact, stem, choices, answer, tokenizer):
    input_text = f"[START] {fact} {stem} [A] {choices[0]} [B] {choices[1]} [C] {choices[2]} [D] {choices[3]} [ANSWER] {answer}"
    inputs = tokenizer(
        input_text, truncation=True, padding="max_length", max_length=512
    )
    return inputs


def read_data(file_name, tokenizer, print_obs_flag=False):
    answers = ["A", "B", "C", "D"]
    list_obs = []
    with open(file_name) as json_file:
        json_list = list(json_file)

    for i in range(len(json_list)):
        json_str = json_list[i]
        result = json.loads(json_str)

        fact = result["fact1"]
        stem = result["question"]["stem"]
        choices = [choice["text"] for choice in result["question"]["choices"]]
        answer = answers.index(result["answerKey"])
        answer_label = answers[answer]

        obs = encode_instance(fact, stem, choices, answer_label, tokenizer)
        list_obs.append([obs, answer_label])

    return list_obs


def process_data(data):
    input_ids = []
    attention_masks = []
    labels = []

    for inputs, label in data:
        input_ids.append(torch.tensor(inputs["input_ids"]).squeeze(0))
        attention_masks.append(torch.tensor(inputs["attention_mask"]).squeeze(0))
        labels.append(torch.tensor(inputs["input_ids"]).squeeze(0))

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.stack(labels)

    return input_ids, attention_masks, labels
