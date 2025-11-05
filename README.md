## What's In It
This repo contains the Pytorch implementation of a transformer decoder and is trained as a character level LLM on Shakespeare's collected works.
The notebook takes about 25 minutes to run on a GPU P100 processor.

## Technical Details
As in the transformers paper by Vishwani et all, the decoder is composed of N=6 layers, each having 2 sublayers - Multi Headed Masked Self Attention & Position Wise FeedForward Network. </br>
Cross entropy loss is utilized along with Adam optimizer with learning rate of 3e-4 </br>
Input and positional embeddings are learned through an nn.Embedding layer. </br>
Training curve: </br>
<img width="500" height="550" alt="image" src="https://github.com/user-attachments/assets/d264d6f2-d4f8-4bce-90de-43d7f3097845" />



## Error Analysis of Regularization Methods
For regularization, each sublayer is encapsulated in a residual block with Layer Normalization and we employ dropout. Early stopping and label smoothing was also added. </br>

| Regularization | Training Loss | Validation Loss | Variance | %Bias Increment | %Variance Drop
| --- | --- | --- | --- | --- | --- |
| Dropout ratio = 0.1 | 0.91 | 1.61 | 0.7 | Baseline | Baseline |
| Dropout ratio = 0.1 + Early Stopping | 1.01 | 1.55 | 0.54 | 11% | 23% |
| Dropout ratio = 0.2 + Early Stopping | 1.11 | 1.51 | 0.4 | 22% | 43% |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.05 | 1.4 | 1.78 | 0.38 | 54% | 45% |
| Dropout ratio = 0.2 + Early Stopping + Label Smoothing = 0.1 | 1.68 | 2 | 0.32 | 85% | 54% |

The bias-variance tradeoff seems pretty evident from the comparison above. The most reliable method to improve upon variance without effecting bias is to add more training data.
</br>
% Bias Increment and % Variance Drop measure the increment and decrement of training loss and variance in each successive regularization method, keeping the 1st method as baseline. Keeping the tradeoff in mind, I drop label smoothing because it hurts bias more than improving variance.

## Generating Shakespeare!
The model was able to understand the structure of Shakespeare texts which composes of a speaker followed by the dialogue. Although the text does not seem to make much sense, which can be expected since we are modelling at the character level, the grammatics, and text structure was learnt well by the model. </br>
Sample generation: </br>
<img width="400" height="450" alt="Shakespeare text generation" src="https://github.com/user-attachments/assets/f3dea625-b859-4285-bb91-7a386ea5f452" />

