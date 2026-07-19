---
name: CNN Builder
description: Builds and trains the baseline PyTorch convolutional neural network required for fish species classification.
---

# Role

Implement the baseline CNN classifier using PyTorch.

The goal is to create the required baseline model only.
Do not perform hyperparameter optimization or architecture experimentation.

# Framework Requirement

Use:
- PyTorch
- torch.nn.Module
- PyTorch DataLoader

Do not implement neural network layers or backpropagation manually with NumPy.

# Architecture Requirements

Create a CNN from scratch.

The baseline model architecture must match this required structure.

Do not add, remove, or alter convolution blocks.
Do not change filter counts.
Do not tune architecture parameters.

## Convolution Block 1

Required:
- Conv2D
- Input channels: 3
- Output channels: 32
- ReLU activation
- MaxPool

## Convolution Block 2

Required:
- Conv2D
- Output channels: 64
- ReLU activation
- MaxPool

## Convolution Block 3

Required:
- Conv2D
- Output channels: 128
- ReLU activation
- MaxPool

## Classifier

Required:
- Flatten feature maps
- At least one fully connected hidden layer
- Final classification layer matching number of fish classes

The final layer must produce class scores for six fish species.

Use PyTorch-compatible output:
- Return logits from the model.
- Apply softmax only during evaluation/inference.

# Training Requirements

Implement baseline training only.

Use:

Optimizer:
- Adam

Learning rate:
- 0.001

Batch size:
- 32

Training:
- Fixed number of epochs
- Track training and validation metrics each epoch

Record:

- training loss
- validation loss
- training accuracy
- validation accuracy

# Required Outputs

Generate:

Model:
- Save baseline model weights

History:
- Save training history containing loss and accuracy metrics

Plots:
- Training vs validation accuracy curve
- Training vs validation loss curve

Artifact Naming:

Save baseline artifacts separately from optimized artifacts.

Required baseline files:

Model:
- outputs/models/baseline_cnn.pt

History:
- outputs/history/baseline_history.json

Plots:
- outputs/plots/baseline_accuracy_curve.png
- outputs/plots/baseline_loss_curve.png

Do not overwrite optimized model artifacts.

# Integration Rules

Use the existing:
- dataset pipeline
- train/validation/test split
- augmentation pipeline

Do not:
- rewrite preprocessing
- change dataset loading
- create new data formats

# Constraints

Do not:
- tune hyperparameters
- add transfer learning
- use pretrained models
- add unnecessary architectural complexity
- modify unrelated project files

Keep the implementation simple, reproducible, and aligned with the assignment baseline requirements.

# Completion Checklist

Before finishing, verify:

- [ ] PyTorch CNN implementation
- [ ] Three convolution blocks
- [ ] Filters are 32/64/128
- [ ] ReLU after each convolution
- [ ] MaxPool after each block
- [ ] Fully connected hidden layer exists
- [ ] Six-class output layer exists
- [ ] Adam optimizer
- [ ] Learning rate 0.001
- [ ] Batch size 32
- [ ] Model weights saved
- [ ] Accuracy and loss curves generated
