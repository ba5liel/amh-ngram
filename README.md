# Amharic N-gram Language Model

This project implements n-gram language models for Amharic text using the GPAC (Genetically Pooled Amharic Corpus). The implementation covers n-grams for n=1,2,3,4 and includes both intrinsic and extrinsic evaluation methods.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- WordCloud

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install numpy matplotlib wordcloud
   ```
4. Place the `GPAC.txt` corpus file in the root directory

## Features

1. **N-gram Model Implementation**
   - Creates n-grams for n=1,2,3,4
   - Calculates n-gram probabilities
   - Calculates conditional probabilities
   - Implements stopword removal
   - Generates word clouds
   - Calculates sentence probabilities
   - Generates random sentences

2. **Evaluation Methods**
   - Intrinsic evaluation using perplexity
   - Extrinsic evaluation using next word prediction

## Running the Code

```bash
python main.py
```

The script generates results in the `output` directory, including word clouds and model statistics.

## Implementation Details

- Tokenization is performed using simple whitespace splitting
- N-gram probabilities are calculated as frequency/total
- Conditional probabilities use bigram and unigram counts
- Perplexity is used as intrinsic evaluation metric
- Next word prediction accuracy is used for extrinsic evaluation
- Simple smoothing is applied for unknown n-grams

## Corpus Information

The Amharic corpus used in this project is from the GPAC (Genetically Pooled Amharic Corpus) as described in the paper: https://doi.org/10.3390/info12010020. 