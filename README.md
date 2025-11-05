## What's In It
This repo contains the Pytorch implementation of a transformer decoder and is trained as a character level LLM on Shakespeare's collected works.
The notebook takes about 25 minutes to run on a GPU P100 processor.

## Technical Details
As in the transformers paper by Vishwani et all, the decoder is composed of N=6 layers, each having 2 sublayers - Multi Headed Masked Self Attention & Position Wise FeedForward Network.
Cross entropy loss is utilized along with Adam optimizer with learning rate of 3e-4
For regularization, each sublayer is encapsulated in a residual block with Layer Normalization and we employ dropout with dropout ratio as 0.2. Early stopping and label smoothing was also added.
Input and positional embeddings are learned through an nn.Embedding layer.

| Regularization | Training Loss | Validation Loss | Variance | %Bias Increment | %Variance Drop
| --- | --- | --- | --- | --- | --- |
| Dropout ratio = 0.1 | 0.91 | 1.61 | 0.7 | Baseline | Baseline |
| Dropout ratio = 0.1 + Early Stopping | 1.01 | 1.55 | 0.54 | 11% | 23% |
| Dropout ratio = 0.2 + Early Stopping | 1.11 | 1.51 | 0.4 | 22% | 43% |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.05 | 1.4 | 1.78 | 0.38 | 54% | 45% |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.1 | 1.68 | 2 | 0.32 | 85% | 54% |
