import json
import random
from itertools import permutations

import torch


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
        correct_answer_index = answers.index(result["answerKey"])
        choices = [result["question"]["choices"][j]["text"] for j in range(4)]

        if permute:
            # Generate all permutations of choices and corresponding labels
            for perm in permutations(enumerate(choices), len(choices)):
                perm_indices, perm_choices = zip(*perm)
                labels = [
                    1 if idx == correct_answer_index else 0 for idx in perm_indices
                ]
                text = (
                    "[CLS] "
                    + fact
                    + " [SEP] "
                    + stem
                    + " "
                    + " [SEP] ".join(perm_choices)
                    + " [END]"
                )
                instances.append([text, labels])
        else:
            labels = [0] * len(choices)
            labels[correct_answer_index] = 1
            text = (
                "[CLS] "
                + fact
                + " [SEP] "
                + stem
                + " "
                + " [SEP] ".join(choices)
                + " [END]"
            )
            instances.append([text, labels])
    return instances


class MultipleChoiceDataloader:
    def __init__(self, data, batch_size=10):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size + (
            len(self.data) % self.batch_size > 0
        )

    def shuffle_data(self):
        random.shuffle(self.data)

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i : i + self.batch_size]
            texts, labels = zip(*batch)
            labels = torch.tensor(labels, dtype=torch.long)
            yield texts, labels
