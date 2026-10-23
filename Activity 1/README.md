# Text Classification with RNN

## Overview
This project uses a simple RNN to classify IMDB movie reviews as positive or negative.

## Dataset
- Dataset: IMDB
- Samples: 3,000 reviews
- Train/Test Split: 80% / 20%

## Model & Training
- Tokenizer: `tiktoken` (`cl100k_base`)
- Maximum Length: 256 tokens
- Embedding & Hidden Dimension: 64
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Epochs: 5
- Batch Size: 32

## Conclusion
The RNN learned basic sentiment patterns but showed limited performance on unseen reviews.
