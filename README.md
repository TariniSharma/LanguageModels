## What's There
This repo contains the Pytorch implementation of a transformer decoder and is trained as a character level LLM on Shakespeare's collected works.
The notebook takes about 25 minutes to run on a GPU P100 processor.

## Details
As in the transformers paper by Vishwani et all, the decoder is composed of N=6 layers, each having 2 sublayers - Multi Headed Masked Self Attention & Position Wise FeedForward Network.
For regularization, each sublayer is encapsulated in a residual block with Layer Normalization and we employ dropout with dropout ratio as 0.2. Early stopping and label smoothing was also added.
Input and positional embeddings are learned through an nn.Embedding layer.

| Regularization | Training Loss | Validation Loss | Variance
| --- | --- | --- | --- |
| Dropout ratio = 0.1 | 0.91 | 1.61 | 0.7 |
| Dropout ratio = 0.1 + Early Stopping | 1.01 | 1.55 | 0.54 |
| Dropout ratio = 0.2 + Early Stopping | 1.057 | 1.53 | 0.473 |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.05 | 1.4 | 1.78 | 0.38 |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.1 | 1.68 | 2 | 0.32 |
