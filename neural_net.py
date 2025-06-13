import numpy as np
import random
import time

from colorama import Fore, Style
from typing import Optional

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
        self.weights = [np.random.randn(y, x) for y, x in zip(struct[1:], struct[:-1])]
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
    
    def backprop(self, training_sample, expected):
        """The backpropagation algorithm for propagating the output layer error through all the layers of the network and outputting gradient fields for weights
        and biases of the network.

        Args:
            training_sample (np.ndarray): activation matrix of the input layer with shape (input_layer_size, batch_size). Each column is a training data unit.
            expected (np.ndarray): expected activations matrix based on the labels of training data with shape (output_layer_size, batch_size).

        Returns:
            tuple:
                grad_w (list[np.ndarray]): gradient field for weights, each with shape matching the corresponding weight matrix.
                grad_b (list[np.ndarray]): gradient field for biases, each with shape matching the corresponding bias vector.
        """
        # setting up the matrices for storing the gradient fields of weights and biases
        # each entry within
        batch_size = training_sample.shape[1]
        grad_w = [np.zeros((*w.shape, batch_size)) for w in self.weights]
        grad_b = [np.zeros((*b.shape, batch_size)) for b in self.biases]

        activs = [training_sample] # stores the activations of each neuron per layer
        z = [] # stores the weighted inputs in each neural layer

        # feed the input image into the input layer and push it through 
        for w, b in zip(self.weights, self.biases) :
    
            # need to expand b to apply to all training data units
            b = np.broadcast_to(b, (b.shape[0], batch_size))

            # calculate the z of the current layer
            weighted_in = w @ training_sample + b
            z.append(weighted_in)

            training_sample = self.sigmoid(weighted_in)
            activs.append(training_sample)

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
            unified_w.append(np.einsum("lkb->lk", w) / batch_size)
            unified_b.append(np.einsum("lb->l", b)   / batch_size)

        grad_w = unified_w
        grad_b = unified_b

        return (grad_w, grad_b)
    
    def evaluate(self, test_data):
        """Evaluate network output for an input image given the currently set weights and biases, and compare the obtained result against the associated label.
        If the test data is composed of multiple test samples, the function also computes the accuracy of the network.

        Args:
            test_data (list[tuple(list[int], int)]): a list of tuples consisting of image testing data and associated labels

        Returns:
            tuple:
                total_pass (int): number of images correctly identified from the test data pool.
                accuracy (float): percent accuracy of the network at identifying the correct label of a testing sample.
        """      
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        total_pass   = sum(int(x == y) for (x, y) in test_results)
        accuracy     = total_pass/len(test_data)

        return total_pass, accuracy

    def stochasticGD(self, epochs: int, batch_size: int, learning_rate: float, 
                     training_data: list[tuple[list[int],list[int]]],
                     test_data: Optional[list[tuple[list[list[int]], list[int]]]] = None) -> None:
        """Performs *stochastic gradient descent* based on the parameter gradient fields obtained from the backpropagation algorithm and **updates the weights and 
        biases of the network** based on the descent.

        Args:
            epochs (int): the number of training epochs to perform
            batch_size (int): the size of a single mini-batch
            training_data (list[tuple[list[int], list[int]]]): at list of tuples consisting of the training data and associated labels
            learning_rate (int): the rate at which gradient descent should be performed. Mathematically referred to as eta.
            test_data (list[tuple[int]], optional): Testing data to evaluate the network against. 
                At the end of each epoch, the network's accuracy is evaluated and printed to the terminal. Defaults to None.
        """        

        # warn the user about unused data in case of a suboptimal batch size selection
        unused_data = len(training_data) % batch_size
        if unused_data > 0:
            print(Fore.RED + "WARNING:" + Style.RESET_ALL + " due to the selected batch size, " + str(unused_data)
                + " units of training data have not been used.")

        epoch_accuracies = []
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            # simulates picking random training data samples for mini-batching
            random.shuffle(training_data)

            # generating the mini-batches
            mini_batches = []
            divs = len(training_data) // batch_size # integer division defaults to the floor, which is what we need
            for i in range(divs): 
                mini_batches.append(training_data[batch_size * i: batch_size * (i + 1)])

            for mini_batch in mini_batches:
                # separating the images and labels for ideal feeding into backprop
                # also converting them into matrices where each image is a row
                # NOTE conventionally, each training sample is supposed to be a column so keep that in mind for the future
                training_images, training_labels = zip(*mini_batch)
                training_images = np.column_stack(list(training_images))
                training_labels = np.column_stack(list(training_labels))

                gradient_w, gradient_b = self.backprop(training_images, training_labels)

                # updating weights and biases based on gradient fields
                self.weights = [w - (learning_rate) * gw
                                for w, gw in zip(self.weights, gradient_w)]
                self.biases  = [b - (learning_rate) * gb.reshape(b.shape)
                                for b, gb in zip(self.biases, gradient_b)]
        
            # epoch complete, print message
            if test_data:
                epoch_end  = time.perf_counter()
                training_duration = epoch_end - epoch_start

                total_pass, accuracy = self.evaluate(test_data)

                print(f"Epoch {epoch + 1}")
                print(f"├─ Correct: {total_pass}/{len(test_data)}")
                print(f"├─ Accuracy: {accuracy*100:.2f}%")
                print(f"└─ Training Duration: {training_duration:.2f}s")
                print("\n")

                epoch_accuracies.append(accuracy)

            else:
                print(f"Epoch {epoch} complete!")

        # once all the epochs were computed, find the epoch with the highest accuracy
        best_epoch = np.argmax(epoch_accuracies)
        best_acc   = epoch_accuracies[best_epoch]
        
        print("---------------------------------------------")
        print(f"-> Highest accuracy of {best_acc*100:.2f}% achieved in Epoch {best_epoch+1}")
