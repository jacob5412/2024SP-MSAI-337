import random


def generate_answer(model, tokenizer, input_text, device):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_length=512, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_random_samples(data_loader, model, tokenizer, device, num_samples=10):
    model.eval()
    all_data = []
    for batch in data_loader:
        input_, label = batch
        all_data.append((input_, label))
    random_samples = random.sample(all_data, num_samples)

    for sample in random_samples:
        input_, label = sample
        input_text = tokenizer.decode(input_["input_ids"], skip_special_tokens=True)
        input_text_processed = input_text.split("[ANSWER]")[0] + "[ANSWER]"
        actual_answer = input_text.split("[ANSWER]")[1].strip()

        generated_answer = generate_answer(
            model, tokenizer, input_text_processed, device
        )
        generated_answer_text = generated_answer.split("[ANSWER]")[1].strip()

        print("Input text for generation:", input_text_processed)
        print("Generated answer:", generated_answer_text)
        print("Actual answer:", actual_answer)
        print("----")
