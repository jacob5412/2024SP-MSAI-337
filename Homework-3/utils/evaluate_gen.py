def generate_answer(model, tokenizer, input_text, device):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_length=512, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(test_data, model, tokenizer, device, print_answers=False):
    val_predictions = []
    val_references = []

    correct = 0
    for idx, item in enumerate(test_data):
        input, label = item

        input_text = tokenizer.decode(input["input_ids"], skip_special_tokens=True)
        input_text1 = input_text.split("[ANSWER]")[0] + "[ANSWER]"

        generated_answer = generate_answer(model, tokenizer, input_text1, device)
        input_answer = input_text.split("[ANSWER]")[1].strip()
        if print_answers and idx % 100 == 0:
            print("input text for generation:", input_text1)

            print("generated answer:", generated_answer)
            print("actual answer:", input_answer)

        if input_answer[0] == generated_answer.split("[ANSWER]")[1].strip()[0]:
            correct += 1

        val_predictions.append(generated_answer)
        val_references.append(input_answer)

    accuracy = correct / len(test_data)
    return accuracy
