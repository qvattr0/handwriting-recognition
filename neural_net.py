import numpy as np
import random
import h5py
import copy
import time
import os

from colorama import Fore, Style
from typing import Optional

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
                - grad_w (list[np.ndarray]): gradient field for weights, each with shape matching the corresponding weight matrix.
                - grad_b (list[np.ndarray]): gradient field for biases, each with shape matching the corresponding bias vector.
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
                - total_pass (int): number of images correctly identified from the test data pool.
                - accuracy (float): percent accuracy of the network at identifying the correct label of a testing sample.
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
        # resetting compute time because it is meant to measure the training duration for a single
        self.compute_time = 0
        
        # saving the hyper-parameters for later use
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trainingpool_size = len(training_data)

        # warn the user about unused data in case of a suboptimal batch size selection
        unused_data = len(training_data) % batch_size
        if unused_data > 0:
            print(Fore.RED + "WARNING:" + Style.RESET_ALL + " due to the selected batch size, " + str(unused_data)
                + " units of training data have not been used.")
        
        param_history    = []
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
        
            # epoch complete, perform final tasks and print message
            epoch_end  = time.perf_counter()
            training_duration = epoch_end - epoch_start
            self.compute_time += training_duration

            if test_data:
                total_pass, accuracy = self.evaluate(test_data)

                print(f"Epoch {epoch + 1}")
                print(f"├─ Correct: {total_pass}/{len(test_data)}")
                print(f"├─ Accuracy: {accuracy*100:.2f}%")
                print(f"└─ Training Duration: {training_duration:.2f}s")
                print("\n")

                epoch_accuracies.append(accuracy)

            else:
                print(f"Epoch {epoch} complete!")

            # save the parameters computed this epoch
            param_history.append((copy.deepcopy(self.weights), copy.deepcopy(self.biases)))

        # once all the epochs were computed, find the epoch with the highest accuracy
        best_epoch = np.argmax(epoch_accuracies)
        best_acc   = epoch_accuracies[best_epoch]
        
        # save the best accuracy for future use
        self.best_acc = best_acc

        # change the parameter-state of the network to that of the best epoch
        self.weights, self.biases = param_history[best_epoch]

        
        print("--------------------------------------------------")
        print(f"-> Highest accuracy of {best_acc*100:.2f}% achieved in Epoch {best_epoch+1}")

    def save_data(self, label: str, notes: str, simple_save=False) -> None:
        """Saves the parameters and run properties of the network into an HDF file. The following network properties are saved:
        - Layer structure
        - Weights
        - Biases
        - Epochs used for training
        - Batch size
        - Learning rate
        - Training pool size
        - Network accuracy
        - Compute time

        Also has a simple save mode that just saves the weights and biases for quickly transferring network parameters between systems. Simple save mode saves
        the data either in the local `./data` folder or in the root directory of the script.

        Args:
            label (str): name to be used when saving this run to the file       
            notes (str): various annotations to be saved along with the data
            simple_save (bool): activates the simple save mode that creates a HDF file only with weights and biases
        """
        print("\n\n")
        if simple_save is True:
            print(" Saving data...")
            print(Fore.YELLOW + "[Simple Mode]" + Style.RESET_ALL)

            # check if the data directory exists
            if os.path.isdir("./data"):
                print(" ./data directory found")
                path = f"./data/{label}"
            else:
                print(" ./data directory not found, saving in root")
                path = label
            
            # saving the data
            with h5py.File(path, "w") as f:
                params = f.create_group('parameters')
                params.create_dataset('layer structure', data=self.layer_sizes)
                for i, (self.weights, self.biases) in enumerate(zip(self.weights, self.biases)):
                    layer = params.create_group(f"layer{i}")
                    layer.create_dataset('weights', data=self.weights)
                    layer.create_dataset('biases',  data=self.biases)

                # formatting the notes attribute
                f.create_dataset('notes', data=notes)

                print("󱣪 Data saved!")
                return None


        with h5py.File('./data/networks_repo.h5', 'r+') as f:
            print('󰳻 Saving data...')
            print(Fore.YELLOW + " [In networks_repo.h5]" + Style.RESET_ALL)
            # synthesizing the label
            run_names = list(f.keys())
            last_run  = int(run_names[-1][-1])

            label = label + f"-run{last_run + 1}"
            grp_run = f.create_group(label, track_order=True)

            grp_param = grp_run.create_group('parameters')
            grp_param.create_dataset('layer structure', data=self.layer_sizes)
            for i, (self.weights, self.biases) in enumerate(zip(self.weights, self.biases)):
                layer = grp_param.create_group(f"layer{i}")
                layer.create_dataset('weights', data=self.weights)
                layer.create_dataset('biases',  data=self.biases)
            
            metrics = grp_run.create_group('metrics')
            metrics.create_dataset('epochs', data=self.epochs)
            metrics.create_dataset('batch size', data=self.batch_size)
            metrics.create_dataset('learning rate', data=self.learning_rate)
            metrics.create_dataset('training pool size', data=self.trainingpool_size)
            metrics.create_dataset('best accuracy', data=self.best_acc)
            metrics.create_dataset('compute time', data=self.compute_time)

            grp_run.create_dataset('notes', data=notes)

            print("󱣪 Data saved!")
    
    def load_data(self, source: str, label: Optional[str] = None):
        """Loads the parameters of a neural network from a HDF file that stores information about previous runs or a single run.

        Upon loading in a network may also change the internal network structure assigned during the creation of the network object.

        Args:
            source (str): the path to the file with the network
            label  (str): label of the run if you're inputting from a source with multiple network entries
        """

        with h5py.File(source, 'r') as f:
            # choosing the proper depth depending on whether we're using a repo or singleton save file
            netconfig = f[label] if label else f

            # jump to parameters group
            params = netconfig["parameters"] # type: ignore
            assert isinstance(params, h5py.Group)

            # initializing containers and extracting data from save file
            self.weights = []
            self.biases  = []

            # warn the user if the initialized layer structure of the network is different to that of network that is being loaded in
            if params["layer structure"] != self.layer_sizes:
                print(Fore.RED + ' WARNING:' + Style.RESET_ALL + 'the layer structure of the loaded in network does not match the structure with which this network was initialized. Overwriting the initialized structure.')
                self.layer_sizes = params.require_dataset('layer structure', list[int], int)
            
            for layer in params.values():
                
                self.weights.append(layer['weights'][...])
                self.biases.append(layer['biases'][...])

        print("󱣪 Network Loaded!")
        return None        