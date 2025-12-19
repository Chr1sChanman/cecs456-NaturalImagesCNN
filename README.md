# Final Project for CECS 456

A Convolutional Neural Network (CNN) built with PyTorch to classify images from the [Natural Images dataset](https://www.kaggle.com/datasets/prasunroy/natural-images) into 8 categories: airplane, car, cat, dog, flower, fruit, motorbike, and person.

## Notebook Versions

Two versions of the final project notebook are included:

- **report-ver.ipynb** – CNN used for the report, achieving a best validation accuracy of **92.60%**
- **testSet-ver.ipynb** – CNN with an implemented test set evaluation, added after the report was completed to align with the example CNN notebook structure

## Model Overview

The CNN architecture features:
- 4 convolutional blocks with progressive channel expansion (32 → 64 → 128 → 256)
- Batch normalization and ReLU activations
- Dropout regularization (0.5) to prevent overfitting
- AdamW optimizer with adaptive learning rate scheduling
