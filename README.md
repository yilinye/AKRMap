## Introduction
This code implements an adaptive kernel regression guided parametric t-SNE for multimodal embeddings mapping.
The baseline parametric t-SNE loss follows the implementation of the repo https://github.com/Academich/parametric_tsne_pytorch.
On top of that, we introduce a novel kernel regression supervision technique to show the distribution of cross-modal embedding metric like CLIPScore, HPSv2 and PickScore, which are commonly used in evaluation of cross-modal generation like text-to-image or text-to-video generation.

## Run the code
The regression.ipynb shows how to use the code to train a projection and mapping model and draw a static projection map.
For the fully interactive features, we have another notebook Contour_mapping_interactive.ipynb which we suggest testing on colab.

You need the precomputed embeddings to test the code.
We provide the precomputed HPSv2 embeddings on HPD here: https://drive.google.com/drive/folders/1skItPzWBaSssmOr8kWo9Zda-7yAIznea?usp=sharing
  
