from neural_net import Network
import loader

training_data, testing_data = loader.load_data()

# ============================== Parameter Space ============================= #
net_structure = [784, 30, 10]
epochs = 30
batch_size = 10
learning_rate = 3.0
network_gen = """
Mark. 1: 1st generation MLP neural network that deploys randomized parameter initialization.
"""
# ============================================================================ #

net = Network(net_structure)


#net.stochasticGD(30, 10, 3.0, training_data, testing_data)

