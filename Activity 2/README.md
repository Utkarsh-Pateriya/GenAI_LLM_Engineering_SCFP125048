# Text Classification with LSTM

## Overview
This project uses an LSTM network to classify IMDB movie reviews as positive or negative.

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

## Comparison with RNN
The LSTM improved test accuracy and achieved a lower training loss than the simple RNN.

## Conclusion
The LSTM performed better than the RNN, showing improved ability to learn sentiment patterns from movie reviews.
