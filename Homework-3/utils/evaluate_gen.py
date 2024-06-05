from tqdm import tqdm


def generate_batch_answers(model, tokenizer, input_texts, device):
    inputs = tokenizer(
        input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    outputs = model.generate(
        inputs["input_ids"], max_length=512, pad_token_id=tokenizer.eos_token_id
    )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def evaluate_model(test_data, model, tokenizer, device, batch_size=13):
    val_predictions = []
    val_references = []

    correct = 0
    for start_idx in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        end_idx = min(start_idx + batch_size, len(test_data))
        batch = test_data[start_idx:end_idx]

        input_texts = []
        labels = []
        for item in batch:
            input_, label = item
            input_texts.append(input_["input_ids"])

        input_texts_decoded = tokenizer.batch_decode(
            input_texts, skip_special_tokens=True
        )
        input_texts_processed = [
            text.split("[ANSWER]")[0] + "[ANSWER]" for text in input_texts_decoded
        ]
        labels = [text.split("[ANSWER]")[1].strip() for text in input_texts_decoded]

        generated_answers = generate_batch_answers(
            model, tokenizer, input_texts_processed, device
        )

        for idx, generated_answer in enumerate(generated_answers):
            input_answer = labels[idx]
            if input_answer[0] == generated_answer.split("[ANSWER]")[1].strip()[0]:
                correct += 1

            val_predictions.append(generated_answer)
            val_references.append(input_answer)

    accuracy = correct / len(test_data)
    return accuracy
