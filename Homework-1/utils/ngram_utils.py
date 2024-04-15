# File: ngram_utils.py

import io
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
import math

# Ensure NLTK resources are available
nltk.download('punkt')

def read_file(file_path):
    """Read the file content and return as a single string."""
    with io.open(file_path, 'r', encoding='utf8') as file:
        return file.read()

def tokenize(text):
    """Tokenize text using NLTK's word tokenizer."""
    return word_tokenize(text.lower())

def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens."""
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def get_ngrams_from_file(file_path, n):
    """Read file, tokenize and generate n-grams."""
    text = read_file(file_path)
    tokens = tokenize(text)
    ngrams = generate_ngrams(tokens, n)
    return ngrams

def ngram_frequencies(ngrams):
    """Calculate frequencies of n-grams."""
    return Counter(ngrams)

def calculate_probability(ngrams, ngram_freqs, total_ngrams, vocabulary_size, smoothing=1):
    """Calculate probabilities with smoothing for n-grams."""
    probabilities = {}
    for ngram in ngrams:
        count = ngram_freqs.get(ngram, 0)
        probability = (count + smoothing) / (total_ngrams + smoothing * vocabulary_size)
        probabilities[ngram] = probability
    return probabilities

def calculate_perplexity(test_ngrams, ngram_probabilities):
    """Calculate the perplexity of a test set given n-gram probabilities."""
    log_prob_sum = 0
    count = 0
    for ngram in test_ngrams:
        if ngram in ngram_probabilities:
            probability = ngram_probabilities[ngram]
            log_prob_sum += math.log(probability)
            count += 1
    if count == 0:
        return float('inf')  # All test ngrams were unseen
    return math.exp(-log_prob_sum / count)


