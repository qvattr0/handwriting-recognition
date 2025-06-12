from neural_net import Network
import loader

net = Network([784, 30, 10])

training_data, testing_data = loader.load_data()

net.stochasticGD(30, 1000, 3.0, training_data, testing_data)
print("Neural network loaded!")

