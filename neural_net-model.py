import numpy as np


# Input data parameters
image_res = 28*28


# Neural network abstraction
class Network(object):

    def __init__(self, sizes):     
        """Neural network initialization constructor

        Args:
            sizes (array[int]): an array integer that represents the layer sizes of the network
                                - for instance, [10,20,10] would produce a 3-layer neural network with a first layer of 10 neurons,
                                  second layer of 20 neurons, and third layer of 10 neurons.

        Returns:
            Network: a network object for storing neural network's parameters
        """        
        self.num_layers = len(sizes)
        self.layer_sizes = sizes
        self.weights = [np.random.randn(sizes[:-1]), np.random.rand(sizes[1:])]
        self.biases = np.random.randn(sizes[1:], 1)
        # TODO: generalize the definition of weights and biases to work on networks with more than 3 layers

        # neuron activation function
        # necessary for using backpropagation as the sigmoid function is differentiable, unlike the alternatively used step function
        def sigmoid(z):
            """Simple sigmoid function to be used as the activation function of each neuron in the network
               Necessary for using backpropagation as the sigmoid function is differentiable, unlike the 
               alternatively used step function.

            Args:
                z (array or int): input integer or array of integers

            Returns:
                array or int: result of the sigmoid function
            """            
            return 1/(1 + np.exp(-z))
        
        
        def feedforward(self, input_data):
            """Calculates the output of the network when input_data is fed into it

            Args:
                input_data (array): data array representing input into the network

            Returns:
                array: response of the network
            """            
            for w, b in zip(self.weights, self.biases):
                response = sigmoid(w @ input_data + b)
            return response


