import numpy as np

class crossentropy_cost():
        """Cross-entropy cost function calculation wrapper

        Returns:
            cross_entropy: an object with two functions for computing cross-entropy
        """
        @staticmethod
        def name():
             return "Cross-entropy Cost Function"
        
        @staticmethod
        def calc(activ, expected):
            a = activ; y = expected
            return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

        @staticmethod
        def delta(z, a, expected):
            return a - expected
        
class quadratic_cost():
    """A simple quadratic cost function calculation wrapper.
    """
    
    @staticmethod
    def name():
         return "Quadratic Cost Function"
    @staticmethod 
    def calc(activ, expected):
        """Returns the quadratic cost of the network output

        Args:
            activ (list[list[int]]): activations of network layers
            expected (list[int]): expected outputs of the network

        Returns:
            list[int]: the loss of each neuron in the output layer
        """        
        a = activ; y = expected
        return 0.5*np.linalg.norm(a-y)**2
    
    def delta(self, weighted_in, activs, expected):
        """The derivative of the cost function that returns the error of each neuron used for deriving the gradients necessary for adjusting the parameters of
        the network

        Args:
            weighted_in (list[list[int]]): _description_
            activs (list[list[int]]): the activation values of network layers
            expected (list[int]): expected values of the network

        Returns:
            list[int]: the error of each neuron in the output layer
        """                

        z = weighted_in; a = activs; y = expected
        return (a - y) * self.sigmoid_prime(z)
    
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
         
    