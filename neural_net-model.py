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
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[:-1], sizes[1:])]
        self.biases =  [np.random.randn(x, 1) for x in sizes[1:]]

    @staticmethod
    def sigmoid(z):
        """Simple sigmoid function to be used as the activation function of each neuron in the network
            Necessary for using backpropagation as the sigmoid function is differentiable, unlike the 
            alternatively used step function.

        Args:
            z (int): input integer or array of integers

        Returns:
            int: result of the sigmoid function
        """            
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        """The derivative of the sigmoid activation function

        Args:
            z (int): values for which the function is computed

        Returns:
            int: result of the function
        """        
        return np.exp(-z)/(1 + np.exp(-z))**2
    
    def feedforward(self, input_data):
        """Calculates the output of the network when input `input_data` is fed into it

        Args:
            input (array[int]): data array representing input into the network

        Returns:
            array[int]: response of the network
        """
        activ = input_data # setting the input as the activation signal

        # repeats until the activation signal traverses the network        
        for w, b in zip(self.weights, self.biases):
            activ = self.sigmoid(w @ activ + b)
            # ! w MUST be a matrix, otherwise the operation will throw an error
            # NOTE: is this even an issue? one could wonder whether having a network with a single input neuron is even useful
        return activ