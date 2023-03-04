# CNN



First, the required packages and hyper-parameters are imported, and the device is selected (either CPU or GPU). Next, the dataset is loaded and normalized using the torchvision package. The images are separated into five classes: airplanes, cars, dogs, faces, and keyboards.

Then, the CNN model is defined using the nn.Module class, which includes several layers of convolution and pooling, followed by a fully connected layer. This model is then used to create a loss function and an optimizer using the cross-entropy loss and the Adam optimizer, respectively.

The model is trained for 20 epochs, and the test accuracy is printed at the end of each epoch. The batch size is set to 16, and the learning rate is set to 0.00001. The seed is set to 666 to ensure reproducibility, and the CuDNN library is also set to be deterministic when running on a GPU.


We define a train function that trains a PyTorch neural network using the given hyperparameters batch_size, n_epochs, and l_rate. The function first prints the hyperparameters, sets up the training and validation data loaders using DataLoader and SubsetRandomSampler, and initializes variables for plotting the loss during training.

The training loop then runs for the specified number of epochs, with a separate loop for each mini-batch of data. Within each mini-batch loop, the input and label tensors are moved to the appropriate device (CPU or GPU), the optimizer's gradients are zeroed, the model is forward propagated, the loss is computed, the gradients are computed by backpropagation, and the optimizer is stepped to update the model parameters. Running loss and total training loss are recorded during the loop, and the running loss is printed every 10% of the mini-batches.

After each epoch, the total validation loss is computed by looping through the validation data loader with the no_grad() context to save memory and computation. If the total validation loss is lower than the previous best error, the model is saved to a file. The average training and validation loss for the epoch are recorded and returned at the end of the function.

Then we instantiate a ConvNet model, and calls the train function with the previously defined hyperparameters to train the model on the provided data. The returned training and validation loss history can be used to visualize the learning progress of the model.


we load the saved model that has the best validation loss. Then, we load the test images and creates a data loader for the test set. After that, we define a function called dataset_accuracy() that takes a neural network model, a data loader, and a string name as input. This function calculates the overall accuracy of the model on the given data loader and prints the accuracy and error rate. The function also returns two lists of correctly and incorrectly classified images with their corresponding labels and predictions.

Next, we define another function called accuracy_per_class() that takes a neural network model and a data loader as input. This function calculates the accuracy per class by creating a confusion matrix and printing the accuracy and error rate for each class. The function returns the confusion matrix.

We call the dataset_accuracy() and accuracy_per_class() functions on the loaded model and test data loader to calculate and print the overall accuracy and accuracy per class, respectively. The correctly and incorrectly classified images are also returned by the dataset_accuracy() function.

**Confusion Matrix**
This section defines a function plot_confusion_matrix that takes in a confusion matrix cm, a list of class labels classes, and some optional parameters such as normalize, title, and cmap. The function generates a plot of the confusion matrix with each cell representing the number of times the predicted class and true class match.

**Correctly and Incorrectly Classified Images**
This section defines a function imshow that takes in an image tensor, unnormalizes it, converts it to a numpy array, and displays it using matplotlib. The section then displays one correctly classified and one incorrectly classified image along with their true and predicted classes.

**Transfer Learning**
This section downloads the ImageNet class labels and defines a list of pretrained models - ResNet18, AlexNet, and VGG16. It then loads a small subset of images from the test set with correct normalization for classification and a larger subset of non-normalized images for display. The section then loops through each pretrained model and generates predictions for the loaded images. The top 5 predicted classes and their probabilities are displayed for each image along with the name of the pretrained model used to generate the predictions.