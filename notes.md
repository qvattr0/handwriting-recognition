# Software Structure
- Consider also having a separate python script dedicated purely for running the model by using the weights produced by the training script

## Issues

### Issue Archive
- ~~The accuracy of the model is abysmal. Another issue seems to be the fact that having larger batch sizes results in even worse performance while the latter should be the case. There may be a link between the starting the degree of accuracy and the batch_size. Figure out what's going on.~~
  - The issue was indeed associated with the batch size usage in calculations. Averaging was mistakenly done twice, resulting in compounding inaccuracy effect as a result of dividing the unified weights and biases matrix twice instead of just once.
- ~~The amount of accuracy increase per epoch seems to decrease as well based on the size of the batch (larger, lower). What the hell is going on?~~ See prior item
- ~~Increasing the number of neurons in the hidden layer increases compute time along with decreasing accuracy. Investigate.~~
  - Apparently increasing the number of hidden layer neurons does indeed result in lower accuracy due to causing more overfitting to occur. However, this effect stops occurring once the amount of parameters exceeds the number of training samples (that means a hidden neuron layer in the thousands of neurons sizes); unfortunately, testing that would be extremely restrictive, so I'd rather not.

# Improvements
- [ ] Find a way to better determine initial weights and biases
  * Could implement a validation stage that uses a subset of the training data to roughly determine a decent starting selection of starting weights and biases
- [ ] Is it possible to optimize learning rate selection?
- [ ] Investigate alternative activation functions that may be more suitable for the task
- [ ] Determine if a better cost function can be used for training
- [x] Rename the `training_images` and `training_labels` to `sample` and `labels` for increased clarity
- [ ] Place version mode and notes into the attributes section of each run entry