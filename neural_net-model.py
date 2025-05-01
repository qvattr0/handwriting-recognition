import numpy as np


# Input data parameters
image_res = 28*28


# Neural network abstraction
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layer_sizes = sizes
        self.weights = [np.random.randn(sizes[:-1]), np.random.rand(sizes[1:])]
        self.biases = np.random.randn(sizes[1:], 1)
        # TODO: generalize the definition of weights and biases to work on networks with more than 3 layers

        # neuron activation function
        # necessary for using backpropagation as the sigmoid function is differentiable, unlike the alternatively used step function
        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
