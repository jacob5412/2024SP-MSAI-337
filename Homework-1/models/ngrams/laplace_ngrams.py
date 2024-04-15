import math

from .ngrams import get_ngram_model


def test_laplace_ngram_model(test_tokens, ngram_counts, n_minus_1_gram_counts, n):
    """
    Calculate the perplexity of a test dataset using an n-gram model with Laplace smoothing,
    handling unigrams and higher n-grams differently.

    Args:
        test_tokens (list of str): The list of tokens to test the model.
        ngram_counts (Counter): The n-gram counts from the training data.
        n_minus_1_gram_counts (Counter or None): The (n-1)-gram counts from the training data, or None for unigrams.
        n (int): The n-gram size.

    Returns:
        float: The calculated perplexity score.
    """
    log_likelihood = 0.0
    N = 0  # total n-grams in the test data
    num_oov = 0  # count of out-of-vocabulary n-grams

    if n == 1:
        total_unigrams = sum(ngram_counts.values())
        vocab_size = len(ngram_counts)
    else:
        vocab_size = len(n_minus_1_gram_counts)

    # Calculate the probability of the n-gram
    for i in range(len(test_tokens) - n + 1):
        test_ngram = tuple(test_tokens[i : i + n])
        test_n_minus_1_gram = test_ngram[:-1] if n > 1 else None

        numerator = ngram_counts.get(test_ngram, 0) + 1
        if numerator == 1:
            num_oov += 1
        
        if n > 1:
            denominator = n_minus_1_gram_counts.get(test_n_minus_1_gram, 0) + vocab_size
        else:
            denominator = total_unigrams + vocab_size

        prob = numerator / denominator
        log_likelihood += math.log(prob)
        N += 1

    perplexity = math.exp(-log_likelihood / N)
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Number of OOV instances: {num_oov}")
    return perplexity


def calculate_laplace_perplexities(train_data, test_data):
    perplexities = {}
    for n in [1, 2, 3, 7]:
        ngram_counts, n_minus_1_gram_counts = get_ngram_model(train_data, n)
        if len(test_data) >= n:
            perplexity = test_laplace_ngram_model(
                test_data, ngram_counts, n_minus_1_gram_counts, n
            )
            perplexities[f"{n}-gram"] = perplexity
    return perplexities
