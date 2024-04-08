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
        tuple: A tuple containing two Counter objects for n-gram and (n-1)-gram counts.
    """
    ngram_counts = generate_ngram_tokens(train_tokens, n)
    n_minus_1_gram_counts = generate_ngram_tokens(train_tokens, n - 1)

    return ngram_counts, n_minus_1_gram_counts

def test_ngram_model(test_tokens, ngram_counts, n_minus_1_gram_counts, n, epsilon=1e-6):
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
    N = 0  # total n-grams in the test data

    for i in range(len(test_tokens) - n + 1):
        test_ngram = tuple(test_tokens[i : i + n])
        test_n_minus_1_gram = test_ngram[:-1]

        # Calculate the probability of the n-gram
        numerator = ngram_counts.get(test_ngram, 0) + epsilon
        denominator = n_minus_1_gram_counts.get(test_n_minus_1_gram, 0) + (
            epsilon * len(n_minus_1_gram_counts)
        )
        prob = numerator / denominator

        # total log likelihood
        log_likelihood += math.log(prob)

        N += 1

    # Calculate perplexity
    perplexity = math.exp(-log_likelihood / N)
    return perplexity

def calculate_perplexities(train_data, test_data):
    perplexities = {}
    for n in [1, 2, 3, 7]:
        ngram_counts, n_minus_1_gram_counts = get_ngram_model(train_data, n)
        perplexity = test_ngram_model(test_data, ngram_counts, n_minus_1_gram_counts, n)
        perplexities[f"{n}-gram"] = perplexity
    return perplexities
