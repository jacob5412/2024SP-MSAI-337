import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_metric

from .post_processing import collect_tensors, extract_prediction


def compute_masked_loss(logits, label_tensor, mask, loss_fn):
    # Flatten the logits and labels
    logits = logits.view(-1, logits.size(-1))
    label_tensor = label_tensor.view(-1)
    mask = mask.view(-1)

    # Filter out the logits and labels that are not masked
    masked_logits = logits[mask]
    masked_labels = label_tensor[mask]

    loss = loss_fn(masked_logits, masked_labels)
    return loss


def train_loop(
    dataloader,
    tokenizer,
    model,
    loss_fn,
    optimizer,
    device,
    log_every,
    decoding="beam",
):
    model.train()
    total_loss = 0
    metric = load_metric("rouge")
    start_time = time.time()

    for i, (texts_labels, attention_masks, start_indices, end_indices) in enumerate(
        dataloader
    ):
        input_label_ids = texts_labels.to(device)
        attention_masks = attention_masks.to(device)
        outputs = model(input_ids=input_label_ids, attention_mask=attention_masks)
        logits = outputs.logits

        # Generate mask and labels tensor for the answer part only
        label_tensor = torch.full_like(input_label_ids, -100)
        mask = torch.zeros_like(input_label_ids, dtype=torch.bool)
        for idx, (input_label_id, start_index, end_index) in enumerate(
            zip(input_label_ids, start_indices, end_indices)
        ):
            label_tensor[idx, start_index:end_index] = input_label_ids[
                idx, start_index:end_index
            ]
            mask[idx, start_index:end_index] = 1

        label_tensor = label_tensor.to(device)
        mask = mask.to(device)

        loss = compute_masked_loss(logits, label_tensor, mask, loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_every == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Batch {i+1}/{len(dataloader)}: Loss = {loss.item()}, Time Elapsed = {elapsed_time:.2f} seconds"
            )
            start_time = time.time()

            # Decode predictions and references using beam search until stop token
            pred_inputs, pred_attention_masks, labels = [], [], []

            for input_id, attention_mask, start_index, end_index in zip(
                input_label_ids, attention_masks, start_indices, end_indices
            ):
                pred_input = input_id[:start_index]
                pred_attention_mask = attention_mask[:start_index]
                label = input_id[start_index:end_index]

                pred_inputs.append(pred_input)
                pred_attention_masks.append(pred_attention_mask)
                labels.append(label)

            pred_input_batches = collect_tensors(
                pred_inputs, tokenizer.pad_token_id, 20
            )
            pred_attention_mask_batches = collect_tensors(pred_attention_masks, 0, 20)
            label_batches = collect_tensors(labels, tokenizer.pad_token_id, 20)

            for pred_inputs, pred_attention_masks, labels in zip(
                pred_input_batches, pred_attention_mask_batches, label_batches
            ):
                if decoding == "beam":
                    preds = model.generate(
                        input_ids=pred_inputs,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True,
                        attention_mask=pred_attention_masks,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:
                    preds = model.generate(
                        input_ids=pred_inputs,
                        max_length=512,
                        early_stopping=True,
                        attention_mask=pred_attention_masks,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
                references = [
                    tokenizer.decode(label, skip_special_tokens=True)
                    for label in labels
                ]
                predictions = [extract_prediction(pred) for pred in predictions]

                metric.add_batch(predictions=predictions, references=references)

    final_metrics = metric.compute()

    return total_loss / len(dataloader), final_metrics
