#!/usr/bin/env python3
"""
Amharic N-gram Language Model Implementation
"""
import os
import time
import random
import argparse
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib
# Set font family to handle Amharic text properly
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# File paths
SAMPLE_DATA_PATH = "GPAC_sample.txt"
FULL_DATA_PATH = "GPAC.txt"
OUTPUT_DIR = "output"
STOPWORDS_PATH = "stopwords.txt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_corpus(file_path, max_lines=None):
    """Read the corpus file and return a list of cleaned sentences"""
    print(f"Reading corpus from {file_path}...")
    
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                sentences.append(line)
                line_count += 1
                
                if max_lines and line_count >= max_lines:
                    break
                
                if line_count % 10000 == 0:
                    print(f"Read {line_count} lines...")
    
    print(f"Loaded {len(sentences)} sentences.")
    return sentences

def get_stopwords(file_path):
    """Read Amharic stopwords from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Warning: Stopwords file {file_path} not found. Using empty stopwords list.")
        return set()

def tokenize_sentence(sentence):
    """Tokenize a sentence into words"""
    # Simple tokenization by whitespace for Amharic
    return sentence.split()

def generate_ngrams(tokens, n):
    """Generate n-grams from a list of tokens"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def log_ngram_examples(ngram_counter, n, num_examples=3):
    """Print example ngrams and their contexts to understand structure"""
    print(f"\n=== N-gram Structure Examples (n={n}) ===")
    
    # Show most common n-grams
    print(f"Top {num_examples} most common {n}-grams:")
    for ngram, count in ngram_counter.most_common(num_examples):
        print(f"  N-gram: '{' '.join(ngram)}' (Count: {count})")
        
        # For n>1, explain conditional probability concept
        if n > 1:
            context = ' '.join(ngram[:-1])
            next_word = ngram[-1]
            print(f"    Context: '{context}' → Next word: '{next_word}'")
            print(f"    This represents P('{next_word}' | '{context}')")
    
    # Show random examples for variety
    print(f"\nRandom {n}-gram examples:")
    random_samples = random.sample(list(ngram_counter.items()), min(num_examples, len(ngram_counter)))
    for ngram, count in random_samples:
        print(f"  N-gram: '{' '.join(ngram)}' (Count: {count})")
        if n > 1:
            context = ' '.join(ngram[:-1])
            next_word = ngram[-1]
            print(f"    This means after seeing '{context}', '{next_word}' appeared {count} times")

def compute_ngram_frequencies(sentences, n, remove_stopwords=False, stopwords=None):
    """Compute frequencies of n-grams from a list of sentences"""
    ngram_counter = Counter()
    token_count = 0
    sentence_with_ngrams = 0
    
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        token_count += len(tokens)
        
        if remove_stopwords and stopwords:
            original_len = len(tokens)
            tokens = [token for token in tokens if token not in stopwords]
            removed = original_len - len(tokens)
            # Only log if stopwords were actually removed
            if removed > 0 and len(tokens) > 0:
                print(f"  Example: Removed {removed} stopwords from: '{' '.join(tokens)}'") if sentence_with_ngrams < 2 else None
        
        if len(tokens) >= n:
            sentence_with_ngrams += 1
            ngrams = generate_ngrams(tokens, n)
            ngram_counter.update(ngrams)
            
            # Log example of how a sentence becomes n-grams (only log first 2 sentences)
            if sentence_with_ngrams <= 2 and not remove_stopwords:
                print(f"\nExample of tokenizing sentence into {n}-grams:")
                print(f"  Sentence: '{sentence}'")
                print(f"  Tokens: {tokens}")
                print(f"  {n}-grams generated:")
                for i, ngram in enumerate(ngrams[:5]):  # Show only first 5 n-grams
                    print(f"    {i+1}. {ngram}")
                if len(ngrams) > 5:
                    print(f"    ... and {len(ngrams)-5} more {n}-grams")
    
    print(f"\nProcessed {token_count} total tokens across {sentence_with_ngrams} sentences")
    print(f"Found {len(ngram_counter)} unique {n}-grams out of {sum(ngram_counter.values())} total {n}-grams")
    
    return ngram_counter

def compute_ngram_probabilities(ngram_counter, total_ngrams):
    """Compute probabilities of n-grams"""
    ngram_probs = {ngram: count / total_ngrams for ngram, count in ngram_counter.items()}
    
    # Log information about probability distribution
    prob_sum = sum(ngram_probs.values())
    max_prob = max(ngram_probs.values()) if ngram_probs else 0
    min_prob = min(ngram_probs.values()) if ngram_probs else 0
    
    print(f"\nProbability statistics:")
    print(f"  Sum of all probabilities: {prob_sum} (should be close to 1.0)")
    print(f"  Maximum probability: {max_prob:.6f}")
    print(f"  Minimum probability: {min_prob:.6f}")
    print(f"  Probability range: {max_prob - min_prob:.6f}")
    
    return ngram_probs

def compute_conditional_probabilities(bigram_counter, unigram_counter):
    """Compute conditional probabilities P(word2|word1) using bigrams"""
    conditional_probs = {}
    
    for (word1, word2), bigram_count in bigram_counter.items():
        unigram_count = unigram_counter[(word1,)]
        conditional_probs[(word1, word2)] = bigram_count / unigram_count
    
    return conditional_probs

def create_wordcloud(ngram_counter, n, output_path):
    """Create and save wordcloud for n-grams"""
    # Convert ngrams to strings for wordcloud
    word_freq = {' '.join(ngram): count for ngram, count in ngram_counter.items()}
    
    # Find appropriate font for Amharic - try multiple options
    font_options = [
        '/System/Library/Fonts/Supplemental/Noto Sans Ethiopic.ttc',  # macOS path for Noto Sans Ethiopic
        '/System/Library/Fonts/Supplemental/Kefa.ttc',                # macOS alternate
        '/Library/Fonts/NotoSansEthiopic-Regular.ttf',                # Another possible location
        '/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf', # Linux path
        '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf',    # Linux Abyssinica font
        'C:\\Windows\\Fonts\\NotoSansEthiopic-Regular.ttf',           # Windows path
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',       # Fallback
        None  # Let WordCloud use default font
    ]
    
    # Try to find a working font
    font_path = None
    for font in font_options:
        if font and os.path.exists(font):
            font_path = font
            print(f"Using font: {font_path}")
            break
    
    # Create wordcloud with appropriate settings for Amharic
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        font_path=font_path,
        regexp=r'\S+',  # Match any non-whitespace as a word
        collocations=False,  # Avoid duplicate phrases
        prefer_horizontal=1.0,  # Keep text horizontal
        min_font_size=12,
        max_font_size=100
    ).generate_from_frequencies(word_freq)
    
    # Plot and save
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{n}-gram Word Cloud")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Higher DPI for better quality
    plt.close()
    
    print(f"Saved word cloud to {output_path}")

def calculate_sentence_probability(sentence, ngram_models, n, verbose=False):
    """Calculate probability of a sentence using n-gram model with detailed logging"""
    tokens = tokenize_sentence(sentence)
    
    if verbose:
        print(f"\nCalculating probability for sentence: '{sentence}'")
        print(f"Tokenized to: {tokens}")
    
    if len(tokens) < n:
        if verbose:
            print(f"Sentence too short for {n}-gram model (needs at least {n} tokens, has {len(tokens)})")
        return 0.0
    
    # Extract relevant model
    ngram_probs = ngram_models[n]['probs']
    
    # For n-grams, probability is product of conditional probabilities
    log_prob = 0.0
    ngram_probs_list = []
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        
        # If ngram not in model, use a small probability (smoothing)
        if ngram in ngram_probs:
            prob = ngram_probs[ngram]
            if verbose:
                print(f"  N-gram '{' '.join(ngram)}' found with prob={prob:.10f}")
        else:
            prob = 1e-10  # Smoothing for unknown n-grams
            if verbose:
                print(f"  N-gram '{' '.join(ngram)}' not found, using smoothing value: {prob:.10f}")
                
        log_prob += np.log(prob)
        ngram_probs_list.append(prob)
    
    final_prob = np.exp(log_prob)
    
    if verbose:
        print(f"Log probability sum: {log_prob:.4f}")
        print(f"Final probability: {final_prob:.10f}")
        print(f"Calculation: exp({log_prob:.4f}) = {final_prob:.10f}")
        
    return final_prob

def generate_random_sentence(ngram_models, n, max_length=20, verbose=False):
    """Generate a random sentence using n-gram model with detailed logging"""
    # Get the most common starting n-gram
    ngram_counter = ngram_models[n]['counter']
    starting_ngrams = [ngram for ngram in ngram_counter.keys() if ngram[0].istitle()]
    
    if verbose:
        print(f"\nGenerating random sentence using {n}-gram model")
        print(f"Found {len(starting_ngrams)} capitalized starting n-grams")
    
    if not starting_ngrams:
        # If no titled ngrams, just pick a random one
        starting_ngrams = list(ngram_counter.keys())
        if verbose:
            print("No capitalized n-grams found, using random n-grams instead")
    
    # Select a random starting n-gram weighted by frequency
    weights = [ngram_counter[ngram] for ngram in starting_ngrams]
    starting_ngram = random.choices(starting_ngrams, weights=weights, k=1)[0]
    
    if verbose:
        print(f"Selected starting {n}-gram: '{' '.join(starting_ngram)}'")
    
    # Initialize sentence with the starting n-gram
    generated_words = list(starting_ngram)
    
    # Generate the rest of the sentence
    for i in range(max_length - n):
        # Get the last n-1 words as context
        context = tuple(generated_words[-(n-1):])
        
        if verbose and i < 5:  # Only log first few steps to avoid excessive output
            print(f"\nStep {i+1}: Current context: '{' '.join(context)}'")
        
        # Find all possible next words
        possible_next = []
        for ngram in ngram_counter:
            if ngram[:-1] == context:
                possible_next.append((ngram[-1], ngram_counter[ngram]))
        
        if not possible_next:
            if verbose:
                print("No continuation found for this context, ending sentence")
            break  # No continuation found
        
        # Choose next word based on probabilities
        next_words, next_weights = zip(*possible_next)
        next_word = random.choices(next_words, weights=next_weights, k=1)[0]
        
        if verbose and i < 5:  # Only log first few steps
            print(f"Found {len(possible_next)} possible next words")
            top_3 = sorted(possible_next, key=lambda x: x[1], reverse=True)[:3]
            print(f"Top 3 candidates: {[word for word, _ in top_3]}")
            print(f"Selected next word: '{next_word}'")
        
        generated_words.append(next_word)
        
        # Stop if we encounter sentence-ending punctuation
        if next_word.endswith(('.', '?', '!')):
            if verbose:
                print(f"Ending sentence due to punctuation: '{next_word}'")
            break
    
    sentence = ' '.join(generated_words)
    if verbose:
        print(f"\nGenerated sentence: '{sentence}'")
    
    return sentence

def intrinsic_evaluation(test_sentences, ngram_models):
    """Evaluate language models using perplexity"""
    results = {}
    
    for n in ngram_models:
        if n == 1:  # Skip unigrams as they don't provide context
            continue
            
        model = ngram_models[n]
        ngram_probs = model['probs']
        
        total_log_prob = 0
        total_tokens = 0
        
        for sentence in test_sentences:
            tokens = tokenize_sentence(sentence)
            
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    
                    # If ngram not in model, use a small probability (smoothing)
                    prob = ngram_probs.get(ngram, 1e-10)
                    total_log_prob += np.log2(prob)
                    total_tokens += 1
        
        # Calculate perplexity
        perplexity = 2 ** (-total_log_prob / total_tokens) if total_tokens > 0 else float('inf')
        results[n] = perplexity
        
    return results

def extrinsic_evaluation(test_sentences, ngram_models):
    """Evaluate language models on next word prediction task"""
    results = {}
    
    for n in ngram_models:
        if n == 1:  # Skip unigrams as they don't provide context
            continue
            
        correct_predictions = 0
        total_predictions = 0
        
        for sentence in test_sentences:
            tokens = tokenize_sentence(sentence)
            
            if len(tokens) >= n:
                for i in range(len(tokens) - n):
                    context = tuple(tokens[i:i+n-1])
                    actual_next_word = tokens[i+n-1]
                    
                    # Find the most likely next word given context
                    best_next_word = None
                    best_prob = 0
                    
                    for ngram, prob in ngram_models[n]['probs'].items():
                        if ngram[:-1] == context:
                            if prob > best_prob:
                                best_prob = prob
                                best_next_word = ngram[-1]
                    
                    if best_next_word == actual_next_word:
                        correct_predictions += 1
                    total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        results[n] = accuracy
        
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Amharic N-gram Language Model')
    parser.add_argument('--full-corpus', action='store_true', 
                        help='Use the full corpus instead of the sample')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Maximum number of lines to process from corpus')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging for detailed n-gram information')
    args = parser.parse_args()
    
    verbose = args.verbose
    
    start_time = time.time()
    
    # Determine which corpus to use
    if args.full_corpus and os.path.exists(FULL_DATA_PATH):
        data_path = FULL_DATA_PATH
    else:
        # Check if sample exists, if not create it
        if not os.path.exists(SAMPLE_DATA_PATH) and os.path.exists(FULL_DATA_PATH):
            print("Sample corpus not found. Creating it now...")
            import sample_corpus
            data_path = SAMPLE_DATA_PATH
        else:
            data_path = SAMPLE_DATA_PATH if os.path.exists(SAMPLE_DATA_PATH) else FULL_DATA_PATH
    
    # Read corpus
    sentences = read_corpus(data_path, max_lines=args.max_lines)
    
    # Split data for training and testing
    split_idx = int(len(sentences) * 0.9)
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    
    print(f"Training on {len(train_sentences)} sentences, testing on {len(test_sentences)} sentences.")
    
    # Load stopwords
    stopwords = get_stopwords(STOPWORDS_PATH)
    print(f"Loaded {len(stopwords)} stopwords.")
    if verbose and stopwords:
        print("Sample stopwords:")
        print(', '.join(list(stopwords)[:10]))
    
    # Task 1.1: Create n-grams for n=1,2,3,4
    print("\n=== Task 1.1: Creating n-grams ===")
    ngram_models = {}
    
    for n in range(1, 5):
        print(f"\nGenerating {n}-grams...")
        
        # Without stopword removal
        ngram_counter = compute_ngram_frequencies(train_sentences, n)
        total_ngrams = sum(ngram_counter.values())
        
        # Log detailed examples to understand n-gram structure
        if verbose:
            log_ngram_examples(ngram_counter, n)
        
        ngram_probs = compute_ngram_probabilities(ngram_counter, total_ngrams)
        
        # Store models
        ngram_models[n] = {
            'counter': ngram_counter,
            'probs': ngram_probs,
            'total': total_ngrams
        }
        
        print(f"Generated {len(ngram_counter)} unique {n}-grams from {total_ngrams} total {n}-grams.")
        
        # Print sample of n-grams
        print(f"Sample of most common {n}-grams:")
        for ngram, count in ngram_counter.most_common(5):
            print(f"  {' '.join(ngram)}: {count}")
    
    # Task 1.2: Calculate probabilities and find top 10 n-grams
    print("\n=== Task 1.2: Top 10 most likely n-grams ===")
    for n in range(1, 5):
        print(f"Top 10 most likely {n}-grams:")
        top_ngrams = sorted(ngram_models[n]['probs'].items(), key=lambda x: x[1], reverse=True)[:10]
        for ngram, prob in top_ngrams:
            print(f"  {' '.join(ngram)}: {prob:.6f}")
    
    # Task 1.3: Calculate conditional probabilities using bigrams
    print("\n=== Task 1.3: Conditional probabilities using bigrams ===")
    cond_probs = compute_conditional_probabilities(
        ngram_models[2]['counter'],
        ngram_models[1]['counter']
    )
    
    print("Sample of conditional probabilities P(word2|word1):")
    for (word1, word2), prob in sorted(cond_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  P({word2}|{word1}): {prob:.6f}")
    
    # Task 1.4: Remove stopwords and recompute
    print("\n=== Task 1.4: N-grams after stopword removal ===")
    ngram_models_no_stopwords = {}
    
    for n in range(1, 5):
        print(f"Generating {n}-grams without stopwords...")
        
        # With stopword removal
        ngram_counter = compute_ngram_frequencies(train_sentences, n, remove_stopwords=True, stopwords=stopwords)
        total_ngrams = sum(ngram_counter.values())
        ngram_probs = compute_ngram_probabilities(ngram_counter, total_ngrams)
        
        # Store models
        ngram_models_no_stopwords[n] = {
            'counter': ngram_counter,
            'probs': ngram_probs,
            'total': total_ngrams
        }
        
        print(f"Generated {len(ngram_counter)} unique {n}-grams from {total_ngrams} total {n}-grams.")
        
        # Top 10 n-grams after stopword removal
        print(f"Top 10 most common {n}-grams after stopword removal:")
        for ngram, count in ngram_counter.most_common(10):
            print(f"  {' '.join(ngram)}: {count}")
    
    # Task 1.5: Create word clouds
    print("\n=== Task 1.5: Creating word clouds ===")
    for n in range(1, 4):  # Unigrams, bigrams, trigrams
        # Word cloud with stopwords
        print(f"Creating word cloud for {n}-grams with stopwords...")
        create_wordcloud(
            ngram_models[n]['counter'],
            n,
            os.path.join(OUTPUT_DIR, f"{n}gram_wordcloud.png")
        )
        
        # Word cloud without stopwords
        print(f"Creating word cloud for {n}-grams without stopwords...")
        create_wordcloud(
            ngram_models_no_stopwords[n]['counter'],
            n,
            os.path.join(OUTPUT_DIR, f"{n}gram_no_stopwords_wordcloud.png")
        )
    
    # Task 1.6: Calculate probability of sample sentences
    print("\n=== Task 1.6: Sentence probability ===")
    sample_sentences = [
        "ኢትዮጵያ ታሪካዊ ሀገር ናት",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነው",
        "የአማርኛ ቋንቋ ብዙ ተናጋሪዎች አሉት"
    ]
    
    for sentence in sample_sentences:
        print(f"Sentence: {sentence}")
        for n in range(1, 5):
            prob = calculate_sentence_probability(sentence, ngram_models, n, verbose=verbose)
            print(f"  Probability using {n}-gram model: {prob:.10f}")
    
    # Task 1.7: Generate random sentences
    print("\n=== Task 1.7: Random sentence generation ===")
    for n in range(2, 5):  # Bigrams, trigrams, 4-grams
        print(f"Generating sentences using {n}-gram model:")
        for i in range(3):
            # Use verbose mode for the first sentence of each n
            sentence = generate_random_sentence(ngram_models, n, verbose=(verbose and i==0))
            print(f"  {sentence}")
    
    # Task 2: Intrinsic evaluation
    print("\n=== Task 2: Intrinsic evaluation (perplexity) ===")
    perplexities = intrinsic_evaluation(test_sentences, ngram_models)
    
    for n, perplexity in perplexities.items():
        print(f"  {n}-gram model perplexity: {perplexity:.4f}")
    
    # Task 3: Extrinsic evaluation
    print("\n=== Task 3: Extrinsic evaluation (next word prediction) ===")
    accuracies = extrinsic_evaluation(test_sentences, ngram_models)
    
    for n, accuracy in accuracies.items():
        print(f"  {n}-gram model accuracy: {accuracy:.4f}")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 