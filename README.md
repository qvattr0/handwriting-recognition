# Introduction

A project attempting to implement a machine learning algorithm that deploys a multi-layer perceptron to recognize handwritten digits represented by images with a 28x28 image resolution. The model will be trained on a MNIST database of handwritten digit images collected by the US postal office. The training method will heavily rely on a backpropagation algorithm.

The primary purpose of this project is to act as a "playground" for exploring the process of developing a neural network, understanding the intricate relationship between the various hyper-parameters used to design the network, investigate the impact of network aspects (activation function, cost function, parameter initialization, etc.) on network performance and accuracy, and most importantly developing a deeper understanding of how neural networks work.

> [!NOTE]
> Some of the print statements use Nerd Font icons so it is suggested to use a Nerd Font compatible font family in your terminal/IDE for the ideal experience using this program.

# Features

## Network Parameters
The neural network architecture used, as mentioned prior, is a multi-layer perceptron. The network utilizes:
- Randomized weight and bias initialization
- Sigmoid activation function
- Square-function cost calculation

## Quality-of-life
Various functions were implemented for accelerating network iteration and analysis workflows. Some of those include:
- Saving network parameters into an HDF file for future reference and analysis
- Loading weight and bias parameters from a previously trained network
