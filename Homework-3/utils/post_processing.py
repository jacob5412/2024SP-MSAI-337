import torch
import torch.nn.functional as F


def split_into_batches(tensors, batch_size):
    return [tensors[i : i + batch_size] for i in range(0, len(tensors), batch_size)]


def collect_tensors(tensors, pad_token_id, batch_size=25):
    # Split into batches of batch_size each
    tensor_batches = split_into_batches(tensors, batch_size)

    # Collect and pad each batch
    collected_batches = []
    for batch in tensor_batches:
        max_len = max(tensor.size(0) for tensor in batch)
        padded_tensors = torch.stack(
            [
                F.pad(tensor, (0, max_len - tensor.size(0)), value=pad_token_id)
                for tensor in batch
            ]
        )
        collected_batches.append(padded_tensors)
    return collected_batches


def extract_prediction(text, answer_token="[ANSWER]"):
    if answer_token in text:
        return text.split(answer_token, 1)[1].strip()
    return text
