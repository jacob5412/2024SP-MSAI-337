import sys
import time

import torch
from datasets import load_metric

from .post_processing import collect_tensors, extract_prediction


def valid_loop(dataloader, tokenizer, model, device, print_every, decoding="beam"):
    model.eval()
    metric = load_metric("rouge")

    with torch.no_grad():
        for i, (texts_labels, attention_masks, start_indices, end_indices) in enumerate(
            dataloader
        ):
            input_label_ids = texts_labels.to(device)

            pred_inputs = []
            pred_attention_masks = []
            labels = []

            for input_id, attention_mask, start_index, end_index in zip(
                input_label_ids, attention_masks, start_indices, end_indices
            ):
                pred_input = input_id[:start_index]
                pred_attention_mask = attention_mask[:start_index]
                label = input_id[start_index:end_index]

                pred_inputs.append(pred_input)
                pred_attention_masks.append(pred_attention_mask)
                labels.append(label)

            pred_input_batches = collect_tensors(pred_inputs, tokenizer.pad_token_id)
            pred_attention_mask_batches = collect_tensors(pred_attention_masks, 0)
            label_batches = collect_tensors(labels, tokenizer.pad_token_id)

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

            end_time = time.time()
            batch_time = end_time - start_time

            if (i + 1) % print_every == 0:
                print(
                    f"Validation Batch {i+1}/{len(dataloader)}: Time = {batch_time:.2f} seconds"
                )

    final_metrics = metric.compute()

    return final_metrics
