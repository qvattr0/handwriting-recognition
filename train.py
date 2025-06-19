from neural_net import Network
import loader
import os

training_data, testing_data = loader.load_data()

# ============================== Parameter Space ============================= #
net_structure = [784, 20, 10]
epochs = 30
batch_size = 10
learning_rate = 3.0
network_gen = """
Mark. 1: 1st generation MLP neural network that deploys randomized parameter initialization.
"""
# ============================================================================ #

net = Network(net_structure)
net.stochasticGD(epochs, batch_size, learning_rate, training_data, testing_data)
net.save_data("mk1-v1.0.0-run1", network_gen)

