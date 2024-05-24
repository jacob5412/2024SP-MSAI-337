import torch


def evaluate(dataloader, tokenizer, model, linear, loss_fn, device):
    model.eval()
    linear.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

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

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
