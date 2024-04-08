import math
from .ngrams import get_ngram_model

def test_laplace_ngram_model(test_tokens, ngram_counts, n_minus_1_gram_counts, n):
    """
    Calculate the perplexity of a test dataset using an n-gram model with Laplace smoothing.

    Args:
        test_tokens (list of str): The list of tokens to test the model.
        ngram_counts (Counter): The n-gram counts from the training data.
        n_minus_1_gram_counts (Counter): The (n-1)-gram counts from the training data.
        n (int): The n-gram size.

    Returns:
        float: The calculated perplexity score.
    """
    log_likelihood = 0.0
    N = 0  # total n-grams in the test data

    # Vocabulary size for Laplace smoothing
    V = len(n_minus_1_gram_counts)

    for i in range(len(test_tokens) - n + 1):
        test_ngram = tuple(test_tokens[i : i + n])
        test_n_minus_1_gram = test_ngram[:-1]

        # Calculate the probability of the n-gram
        numerator = ngram_counts.get(test_ngram, 0) + 1
        denominator = n_minus_1_gram_counts.get(test_n_minus_1_gram, 0) + V
        prob = numerator / denominator

        # total log likelihood
        log_likelihood += math.log(prob)

        N += 1

    # Calculate perplexity
    perplexity = math.exp(-log_likelihood / N)
    return perplexity

def calculate_laplace_perplexities(train_data, test_data):
    perplexities = {}
    for n in [1, 2, 3, 7]:
        ngram_counts, n_minus_1_gram_counts = get_ngram_model(train_data, n)
        perplexity = test_laplace_ngram_model(test_data, ngram_counts, n_minus_1_gram_counts, n)
        perplexities[f"{n}-gram"] = perplexity
    return perplexities
