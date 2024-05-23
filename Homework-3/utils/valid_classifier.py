import time

import torch


def valid_loop(dataloader, tokenizer, model, linear, loss_fn, device, print_every=10):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    start_time = time.time()
    last_print_time = start_time

    with torch.no_grad():
        for batch_index, (texts, labels) in enumerate(dataloader):
            encoded_inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}

            _, labels_indices = labels.max(dim=1)
            labels_indices = labels_indices.to(device)

            output = model(**encoded_inputs)
            cls_embeddings = output.last_hidden_state[:, 0, :]
            logits = linear(cls_embeddings)
            loss = loss_fn(logits, labels_indices)

            preds = logits.argmax(dim=1)
            correct = (preds == labels_indices).sum().item()
            total_correct += correct
            total_samples += labels_indices.size(0)
            total_loss += loss.item()

            if (batch_index + 1) % print_every == 0:
                current_time = time.time()
                batch_loss = loss.item()
                batch_accuracy = correct / labels_indices.size(0)
                print(
                    f"Validation Batch {batch_index + 1}: Loss = {batch_loss:.4f}, Accuracy = {batch_accuracy:.4f}, Time = {current_time - last_print_time:.2f}s"
                )
                last_print_time = current_time

    total_time = time.time() - start_time
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(
        f"Validation finished: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, Total time = {total_time:.2f}s"
    )

    return avg_loss, accuracy
