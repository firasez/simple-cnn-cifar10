# Simple CNN for CIFAR-10 Classification

This project implements a simple Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using PyTorch. The model is designed with three convolutional layers followed by fully connected layers for image classification.

## Project Overview

- **Dataset**: CIFAR-10
- **Model**: Custom Convolutional Neural Network (CNN)
- **Framework**: PyTorch
- **Goal**: Train a CNN to classify 10 classes of images from the CIFAR-10 dataset.

## Model Architecture

The CNN model consists of:

1. **Convolutional Layer 1**: 
   - 3 input channels (RGB image)
   - 32 output channels (feature maps)
   - 3x3 kernel size
   - ReLU activation
   - Max pooling (2x2)

2. **Convolutional Layer 2**: 
   - 32 input channels
   - 64 output channels
   - 3x3 kernel size
   - ReLU activation
   - Max pooling (2x2)

3. **Convolutional Layer 3**: 
   - 64 input channels
   - 128 output channels
   - 3x3 kernel size
   - ReLU activation
   - Max pooling (2x2)

4. **Fully Connected Layer 1**:
   - 512 hidden units
   - ReLU activation

5. **Fully Connected Layer 2**:
   - Output layer with 10 units for classification (CIFAR-10 has 10 classes)

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm

## Train the model by running
-python train.py or by simply running each section of code

## Example output
Epoch 1/5
Train Loss: 1.7895, Test Loss: 1.4297, Test Accuracy: 0.4860
Epoch 2/5
Train Loss: 1.1167, Test Loss: 1.1345, Test Accuracy: 0.6103
Epoch 3/5
Train Loss: 0.8754, Test Loss: 1.0245, Test Accuracy: 0.6611
Epoch 4/5
Train Loss: 0.7023, Test Loss: 1.0248, Test Accuracy: 0.6852
Epoch 5/5
Train Loss: 0.5912, Test Loss: 1.0743, Test Accuracy: 0.7020

## Visualizing results
After training, the training and test loss curves, as well as the test accuracy, will be plotted using matplotlib.
To visualize the results, the following plots will be displayed:
Train / Test Loss: Shows how the loss changes over epochs.
Test Accuracy: Displays the accuracy of the model on the test set.


## Sample plots
Train / Test Loss: A graph comparing the training and testing losses over each epoch.
Test Accuracy: A graph displaying the accuracy on the test set after each epoch.

