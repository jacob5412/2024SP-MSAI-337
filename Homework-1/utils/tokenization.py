def chunked_tokenization(text, tokenizer, chunk_size=1000000):
    tokens = []
    for i in range(0, len(text), chunk_size):
        text_chunk = text[i : i + chunk_size]
        tokens.extend([token.text for token in tokenizer(text_chunk)])
    return tokens

def chunked_tokenization_gpt2(text, tokenizer, chunk_size=5000000):
    tokens = []
    for i in range(0, len(text), chunk_size):
        text_chunk = text[i : i + chunk_size]
        tokens.extend(tokenizer.tokenize(text_chunk))
    return tokens
