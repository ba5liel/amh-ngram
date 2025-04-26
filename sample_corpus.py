#!/usr/bin/env python3
"""
Create a smaller sample of the GPAC corpus for testing purposes
"""
import random
import re

# Configuration
ORIGINAL_CORPUS = "GPAC.txt"
SAMPLE_CORPUS = "GPAC_sample.txt"
SAMPLE_SIZE = 5000  # Number of sentences to include
CHUNK_SIZE = 10000  # Read the file in chunks

def split_into_sentences(text):
    """Split Amharic text into sentences based on sentence-ending punctuation"""
    # Split on typical sentence-ending punctuation in Amharic
    # This includes the ASCII period, exclamation, and question mark,
    # plus special Amharic full stops (፡፡) and other separators
    sentences = re.split(r'[.!?።፡፡]+(?=\s|$)', text)
    return [s.strip() for s in sentences if s.strip()]

# Process the corpus
print(f"Reading and processing original corpus: {ORIGINAL_CORPUS}")
try:
    all_sentences = []
    
    with open(ORIGINAL_CORPUS, 'r', encoding='utf-8') as f:
        # Read and process the file in chunks
        chunk = f.read(CHUNK_SIZE)
        buffer = ""
        
        while chunk:
            buffer += chunk
            
            # Split buffer into sentences
            new_sentences = split_into_sentences(buffer)
            
            if new_sentences:
                # Keep the last sentence in buffer in case it's incomplete
                all_sentences.extend(new_sentences[:-1])
                buffer = new_sentences[-1]
            
            # Read next chunk
            chunk = f.read(CHUNK_SIZE)
            
            if len(all_sentences) % 1000 == 0 and all_sentences:
                print(f"Processed {len(all_sentences)} sentences so far...")
        
        # Process any remaining text in buffer
        if buffer:
            remaining_sentences = split_into_sentences(buffer)
            all_sentences.extend(remaining_sentences)
    
    total_sentences = len(all_sentences)
    print(f"Original corpus contains {total_sentences} sentences")
    
    # Take a random sample
    if SAMPLE_SIZE >= total_sentences:
        sample = all_sentences
    else:
        sample = random.sample(all_sentences, SAMPLE_SIZE)
    
    # Write the sample corpus
    with open(SAMPLE_CORPUS, 'w', encoding='utf-8') as f:
        for sentence in sample:
            f.write(sentence + '\n')
    
    print(f"Created sample corpus with {len(sample)} sentences in {SAMPLE_CORPUS}")
    
except Exception as e:
    print(f"Error processing corpus: {e}")

print("Done!") 