import math
from collections import Counter


def generate_ngram_tokens(tokens, n):
    """
    Generate a Counter of n-gram tuples from a list of tokens.

    Args:
        tokens (list of str): The list of tokens from which to generate n-grams.
        n (int): The number of tokens in each n-gram.

    Returns:
        Counter: A Counter object mapping each n-gram tuple to its frequency.
    """
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def get_ngram_model(train_tokens, n):
    """
    Generate n-gram and (n-1)-gram models from training tokens.

    Args:
        train_tokens (list of str): The list of tokens to train the model.
        n (int): The n-gram size.

    Returns:
        tuple: A tuple containing the n-gram Counter object and, if n > 1, the (n-1)-gram Counter object.
        For unigrams, the second element is None.
    """
    ngram_counts = generate_ngram_tokens(train_tokens, n)
    if n == 1:
        return ngram_counts, None
    else:
        n_minus_1_gram_counts = generate_ngram_tokens(train_tokens, n - 1)
        return ngram_counts, n_minus_1_gram_counts


def test_ngram_model(test_tokens, ngram_counts, n_minus_1_gram_counts, n, epsilon=1e-3):
    """
    Calculate the perplexity of a test dataset using an n-gram model.

    Args:
        test_tokens (list of str): The list of tokens to test the model.
        ngram_counts (Counter): The n-gram counts from the training data.
        n_minus_1_gram_counts (Counter): The (n-1)-gram counts from the training data.
        n (int): The n-gram size.
        epsilon (float): A small value to prevent zero-error in probability calculation.

    Returns:
        float: The calculated perplexity score.
    """
    log_likelihood = 0.0
    N = 0  # Total number of n-grams considered in the test data
    num_oov = 0  # count of out-of-vocabulary n-grams

    if n == 1:
        total_unigrams = sum(ngram_counts.values())
        vocab_size = len(ngram_counts)
    else:
        vocab_size = len(n_minus_1_gram_counts)

    for i in range(len(test_tokens) - n + 1):
        test_ngram = tuple(test_tokens[i : i + n])
        test_n_minus_1_gram = test_ngram[:-1] if n > 1 else None

        # Calculate the probability of the n-gram
        numerator = ngram_counts.get(test_ngram, 0) + epsilon
        if numerator == epsilon:
            num_oov += 1

        if n > 1:
            # When n > 1, we use the (n-1)-gram model for context
            denominator = n_minus_1_gram_counts.get(test_n_minus_1_gram, 0) + (
                epsilon * vocab_size
            )
        else:
            # For unigrams, there is no context and the denominator is different
            denominator = total_unigrams + (epsilon * vocab_size)

        prob = numerator / denominator

        # Accumulate total log likelihood
        log_likelihood += math.log(prob)

        N += 1

    # Calculate and return perplexity
    perplexity = math.exp(-log_likelihood / N)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Number of OOV instances: {num_oov}")
    return perplexity


def calculate_perplexities(train_data, test_data):
    perplexities = {}
    for n in [1, 2, 3, 7]:
        ngram_counts, n_minus_1_gram_counts = get_ngram_model(train_data, n)
        perplexity = test_ngram_model(test_data, ngram_counts, n_minus_1_gram_counts, n)
        perplexities[f"{n}-gram"] = perplexity
    return perplexities
