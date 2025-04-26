#!/usr/bin/env python3
"""
Demonstration script for understanding n-gram language models
This script shows simplified examples to help understand how n-grams work
"""
import os
import numpy as np
from collections import Counter

def main():
    print("=== N-GRAM LANGUAGE MODEL DEMONSTRATION ===")
    
    # Example Amharic corpus (simplified for demonstration)
    corpus = [
        "ኢትዮጵያ ታሪካዊ ሀገር ናት",
        "ኢትዮጵያ ውብ ሀገር ናት",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነው",
        "የኢትዮጵያ ብሔራዊ ቋንቋ አማርኛ ነው",
        "አማርኛ ቋንቋ ብዙ ተናጋሪዎች አሉት"
    ]
    
    print("\n1. Sample corpus (5 sentences):")
    for i, sentence in enumerate(corpus):
        print(f"   {i+1}. {sentence}")
    
    # Tokenize the sentences
    tokenized_corpus = [sentence.split() for sentence in corpus]
    
    print("\n2. Tokenized corpus:")
    for i, tokens in enumerate(tokenized_corpus):
        print(f"   {i+1}. {tokens}")
    
    # Generate n-grams for n=1,2,3
    print("\n3. Generated n-grams by sentence:")
    
    for n in range(1, 4):
        print(f"\n   N-GRAM (n={n}):")
        
        for i, tokens in enumerate(tokenized_corpus):
            if len(tokens) >= n:
                ngrams = []
                for j in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[j:j+n])
                    ngrams.append(ngram)
                
                print(f"   Sentence {i+1}: {ngrams}")
            else:
                print(f"   Sentence {i+1}: Too short for {n}-grams")
    
    # Unigram counts
    all_tokens = []
    for tokens in tokenized_corpus:
        all_tokens.extend(tokens)
    
    unigram_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    print("\n4. Unigram counts and probabilities:")
    print(f"   Total tokens: {total_tokens}")
    
    for token, count in unigram_counter.most_common():
        prob = count / total_tokens
        print(f"   '{token}': count={count}, P('{token}')={prob:.4f}")
    
    # Bigram counts and conditional probabilities
    bigram_counter = Counter()
    for tokens in tokenized_corpus:
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                bigram_counter[bigram] += 1
    
    print("\n5. Bigram counts and conditional probabilities:")
    
    for (w1, w2), count in bigram_counter.most_common():
        # Conditional probability: P(w2|w1) = count(w1,w2) / count(w1)
        cond_prob = count / unigram_counter[w1]
        print(f"   '{w1} {w2}': count={count}, P('{w2}'|'{w1}')={cond_prob:.4f}")
    
    # Sentence probability calculation
    print("\n6. Calculating sentence probability:")
    
    test_sentence = "ኢትዮጵያ ታሪካዊ ሀገር ናት"
    tokens = test_sentence.split()
    
    print(f"   Sentence: '{test_sentence}'")
    print(f"   Tokens: {tokens}")
    
    # Unigram probability
    unigram_prob = 1.0
    print("\n   a. Using Unigram model:")
    
    for token in tokens:
        token_prob = unigram_counter.get(token, 0) / total_tokens
        print(f"     P('{token}') = {token_prob:.4f}")
        unigram_prob *= token_prob
    
    print(f"     P(sentence) = {unigram_prob:.8f}")
    
    # Bigram probability
    print("\n   b. Using Bigram model:")
    print("     For bigrams, we calculate: P(w1) * P(w2|w1) * P(w3|w2) * ...")
    
    bigram_prob = unigram_counter.get(tokens[0], 0) / total_tokens  # P(w1)
    print(f"     P('{tokens[0]}') = {bigram_prob:.4f}")
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        pair = (w1, w2)
        
        # Conditional probability
        pair_count = bigram_counter.get(pair, 0)
        w1_count = unigram_counter.get(w1, 0)
        
        if w1_count > 0:
            cond_prob = pair_count / w1_count
        else:
            cond_prob = 0.0
        
        print(f"     P('{w2}'|'{w1}') = {cond_prob:.4f}")
        bigram_prob *= cond_prob
    
    print(f"     P(sentence) = {bigram_prob:.8f}")
    
    # Smoothing explanation
    print("\n7. Smoothing techniques:")
    print("   When we encounter unseen n-grams (zero probabilities), we need smoothing.")
    
    print("\n   a. Add-one (Laplace) smoothing:")
    print("      P(w2|w1) = [count(w1,w2) + 1] / [count(w1) + V]")
    print("      Where V is vocabulary size (number of unique words)")
    
    print("\n   b. Backoff smoothing:")
    print("      If bigram not found, back off to unigram probability")
    print("      P(w2|w1) ≈ P(w2) if (w1,w2) not seen")
    
    # Practical applications
    print("\n8. N-gram applications:")
    print("   - Text prediction (what's the next likely word?)")
    print("   - Spelling correction (which correction is most probable?)")
    print("   - Machine translation (which translation is most fluent?)")
    print("   - Speech recognition (which transcription is most likely?)")
    
    print("\n=== END OF DEMONSTRATION ===")

if __name__ == "__main__":
    main() 