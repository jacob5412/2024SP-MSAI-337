def generate_batch_answers(model, tokenizer, input_texts, device):
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    outputs = model.generate(
        inputs["input_ids"], max_length=512, pad_token_id=tokenizer.eos_token_id
    )
    decoded_outputs = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return decoded_outputs


def evaluate_zero_shot_model(test_data, model, tokenizer, device, batch_size):
    val_predictions = []
    val_references = []

    correct = 0
    for start_idx in range(0, len(test_data), batch_size):
        end_idx = min(start_idx + batch_size, len(test_data))
        batch = test_data[start_idx:end_idx]

        input_texts = []
        labels = []
        for item in batch:
            input, label = item
            input_text = tokenizer.decode(input["input_ids"], skip_special_tokens=True)
            input_text1 = input_text.split("[ANSWER]")[0] + "[ANSWER]"
            input_texts.append(input_text1)
            labels.append(input_text.split("[ANSWER]")[1].strip())

        generated_answers = generate_batch_answers(
            model, tokenizer, input_texts, device
        )

        for idx, generated_answer in enumerate(generated_answers):
            input_answer = labels[idx]
            if input_answer[0] == generated_answer.split("[ANSWER]")[1].strip()[0]:
                correct += 1

            val_predictions.append(generated_answer)
            val_references.append(input_answer)

    accuracy = correct / len(test_data)
    return accuracy
