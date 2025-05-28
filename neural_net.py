import numpy as np


# Input data parameters
image_res = 28*28


# Neural network abstraction
class Network(object):

    def __init__(self, struct):     
        """Neural network initialization constructor

        Args:
            sizes (array[int]): an array integer that represents the layer sizes of the network
                                - for instance, [10,20,10] would produce a 3-layer neural network with a first layer of 10 neurons,
                                  second layer of 20 neurons, and third layer of 10 neurons.

        Returns:
            Network: a network object for storing neural network's parameters
        """        
        self.num_layers = len(struct)
        self.layer_sizes = struct
        self.weights = [np.random.randn(y, x) for y, x in zip(struct[:-1], struct[1:])]
        self.biases =  [np.random.randn(x, 1) for x in struct[1:]]

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
        
        sig = self.sigmoid(z)        
        return sig * (1 - sig)
    
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
    
    def backprop(self, batch, expected):
        """The backpropagation algorithm for propagating the output layer error through all the layers of the network and outputting gradient fields for weights
        and biases of the network.

        Args:
            batch (int): activation matrix of the input layer. Each element of batch is a single training data unit that consists of activations for the neuron
            of the input layer. Can accept multiple input activations stored in a single matrix at once. 
            expected (int): expected activations matrix based on the labels of training data

        Returns:
            grad_w (int): gradient field for weights
            grad_b (int): gradient field for biases
        """
        # setting up the matrices for storing the gradient fields of weights and biases
        # each entry within
        grad_w = [np.zeros((*w.shape, len(batch))) for w in self.weights]
        grad_b = [np.zeros((*b.shape, len(batch))) for b in self.biases]

        activs = [batch] # stores the activations of each neuron per layer
        z = [] # stores the weighted inputs in each neural layer

        # feed the input image into the input layer and push it through 
        for w, b in zip(self.weights, self.biases) :
    
            # need to expand b to apply to all training data units
            b = np.broadcast_to(b, (b.shape[0], batch.shape[1]))

            # calculate the z of the current layer
            weighted_in = w.T @ batch + b
            z.append(weighted_in)

            batch = self.sigmoid(weighted_in)
            activs.append(batch)

        # now, activations of all the neurons are known

        cost_deriv = activs[-1] - expected
        
        # using equation #1 of backpropagation
        # calculate the error of the output layer
        error = cost_deriv * self.sigmoid_prime(z[-1])

        # relating the error to part. deriv. of b
        grad_b[-1] = error
        grad_w[-1] = np.einsum('lb,kb->lkb', error, activs[-2]) # transposed to allow proper matrix mult      

        # propagate the error backward while evaluating the fields
        for l in range(2, self.num_layers):
            error = self.weights[-l+1].T @ error * self.sigmoid_prime(z[-l])

            grad_b[-l] = error
            grad_w[-l] = np.einsum('lb,kb->lkb', error, activs[-l - 1])
        
        # summing all the batch gradient field to output one unified gradient field
        unified_w = []
        unified_b = []
        for w, b in zip(grad_w, grad_b):
            unified_w.append(np.einsum("lkb->lk", w) / len(batch))
            unified_b.append(np.einsum("lb->l", b) / len(batch))

        grad_w = unified_w
        grad_b = unified_b

        return (grad_w, grad_b)