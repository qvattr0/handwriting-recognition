# Software Structure
- Consider also having a separate python script dedicated purely for running the model by using the weights produced by the training script


# Improvements
- [ ] Find a way to better determine initial weights and biases
  * Could implement a validation stage that uses a subset of the training data to roughly determine a decent starting selection of starting weights and biases
- [ ] Is it possible to optimize learning rate selection?
- [ ] Investigate alternative activation functions that may be more suitable for the task