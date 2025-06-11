# Software Structure
- Consider also having a separate python script dedicated purely for running the model by using the weights produced by the training script

## Issues
- ~~The accuracy of the model is abysmal. Another issue seems to be the fact that having larger batch sizes results in even worse performance while the latter should be the case. There may be a link between the starting the degree of accuracy and the batch_size. Figure out what's going on.~~
  - The issue was indeed associated with the batch size usage in calculations. Averaging was mistakenly done twice, resulting in compounding inaccuracy effect as a result of dividing the unified weights and biases matrix twice instead of just once.
- ~~The amount of accuracy increase per epoch seems to decrease as well based on the size of the batch (larger, lower). What the hell is going on?~~ See prior item

# Improvements
- [ ] Find a way to better determine initial weights and biases
  * Could implement a validation stage that uses a subset of the training data to roughly determine a decent starting selection of starting weights and biases
- [ ] Is it possible to optimize learning rate selection?
- [ ] Investigate alternative activation functions that may be more suitable for the task
- [x] Rename the `training_images` and `training_labels` to `sample` and `labels` for increased clarity