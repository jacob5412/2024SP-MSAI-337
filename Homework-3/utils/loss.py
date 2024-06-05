from torch.nn import CrossEntropyLoss


def custom_loss(outputs, labels, magnification_factor=10):
    # Logits shape: (batch_size, sequence_length, vocab_size)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    mask = shift_labels != -100
    loss[:, -1] *= magnification_factor

    # Compute the mean loss only for non-padding tokens
    loss = (loss * mask).sum() / mask.sum()

    return loss
