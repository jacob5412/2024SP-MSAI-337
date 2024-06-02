import json
import random
from itertools import permutations

import torch
import torch.nn.functional as F


def read_file(file_name):
    with open(file_name) as json_file:
        json_list = list(json_file)
    return json_list


def read_json_data(json_list, permute=True):
    answers = ["A", "B", "C", "D"]
    instances = []

    for json_str in json_list:
        result = json.loads(json_str)
        fact = result["fact1"]
        stem = result["question"]["stem"]
        correct_answer_index = result["answerKey"]
        choices = [result["question"]["choices"][j]["text"] for j in range(4)]

        if permute:
            for perm in permutations(enumerate(choices), len(choices)):
                perm_indices, perm_choices = zip(*perm)
                correct_choice_index = perm_indices.index(
                    answers.index(correct_answer_index)
                )
                correct_label = f"[{answers[correct_choice_index]}] {perm_choices[correct_choice_index]}"
                text = (
                    "[START] "
                    + fact
                    + stem
                    + " "
                    + " ".join(
                        f"[{answers[i]}] {choice}"
                        for i, choice in enumerate(perm_choices)
                    )
                    + " [ANSWER]"
                )
                instances.append([text, correct_label])
        else:
            correct_choice_index = answers.index(correct_answer_index)
            correct_label = f"[{correct_answer_index}] {choices[correct_choice_index]}"
            text = (
                "[START] "
                + fact
                + stem
                + " "
                + " ".join(f"[{answers[i]}] {choices[i]}" for i in range(len(choices)))
                + " [ANSWER]"
            )
            instances.append([text, correct_label])

    return instances


class MultipleChoiceDataloader:
    def __init__(
        self,
        data,
        tokenizer,
        max_length=512,
        batch_size=10,
    ):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data) // self.batch_size + (
            len(self.data) % self.batch_size > 0
        )

    def shuffle_data(self):
        random.shuffle(self.data)

    def _tokenize_text_label(self, text, label):
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].squeeze(0)
        label_ids = self.tokenizer(
            label,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].squeeze(0)

        concatenated_ids = torch.cat([input_ids, label_ids])
        label_start_index = input_ids.size(0)
        label_end_index = concatenated_ids.size(0)

        return concatenated_ids, label_start_index, label_end_index

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            texts, labels = zip(*batch)

            input_label_ids_list = []
            label_start_indices = []
            label_end_indices = []
            attention_masks = []

            for text, label in zip(texts, labels):
                input_label_ids, label_start_index, label_end_index = (
                    self._tokenize_text_label(text, label)
                )
                input_label_ids_list.append(input_label_ids)
                attention_masks.append(
                    (input_label_ids != self.tokenizer.pad_token_id).long()
                )
                label_start_indices.append(label_start_index)
                label_end_indices.append(label_end_index)

            max_len = max([ids.size(0) for ids in input_label_ids_list])
            input_label_ids_padded = torch.stack(
                [
                    F.pad(
                        ids,
                        (0, max_len - ids.size(0)),
                        value=self.tokenizer.pad_token_id,
                    )
                    for ids in input_label_ids_list
                ]
            )
            attention_masks_padded = torch.stack(
                [
                    F.pad(
                        mask,
                        (0, max_len - mask.size(0)),
                        value=0,
                    )
                    for mask in attention_masks
                ]
            )

            yield input_label_ids_padded, attention_masks_padded, label_start_indices, label_end_indices
