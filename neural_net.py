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
    
    def sigmoid_prime(self, z):
        """The derivative of the sigmoid activation function

        Args:
            z (int): values for which the function is computed

        Returns:
            int: result of the function
        """        
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
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
    
    def backprop(self, activ, expected):
        """The backpropagation algorithm for propagating the output layer error through all the layers of the network. 

        Args:
            activ (int): activation of the output layer
            expected (int): expected activations based on the labels of training data

        Returns:
            int: a gradient field for both weights and biases to be used for adjusting them
        """
        # setting up the matrices for storing the gradient field of weights and biases
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        
        activs = [activ] # stores the activations of each neuron per layer
        z = [] # stores the weighted inputs in each neural layer

        # using the output layer activation of the network, derive the activations of all other neurons
        for w, b in zip(self.weights, self.biases):
            # calculate the z of the current layer
            weighted_in = np.dot(w, activ) + b
            z.append(weighted_in)

            activ = self.sigmoid(weighted_in)
            activs.append(activ)

        # now, activations of all the neurons are known

        cost_deriv = activs[-1] - expected
        
        # using equation #1 of backpropagation
        # calculate the error of the output layer
        error = cost_deriv * self.sigmoid_prime(z[-1])

        # relating the error to part. deriv. of b
        grad_b[-1] = error
        grad_w[-1] = np.dot(error, activs[-2].T) # transposed to allow proper matrix mult      

        # propagate the error backward
        for l in range(2, self.num_layers):
            error = np.dot(self.weights[-l+1].T, error) * self.sigmoid_prime(z[-l])

            grad_b[-l] = error
            grad_w[-l] = np.dot(error, activ[-l - 1].T)
        
        return (grad_b, grad_w)