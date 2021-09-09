import numpy as np
import mnist
import nn_numpy as nn

# Prepare training labels in array form
train_labels = [np.zeros((10, 1)) for _ in range(50000)]
mnist_train_labels = mnist.train_labels()
for i in range(50000):
    train_labels[i][mnist_train_labels[i]] = 1

# Prepare training images
train_images = [image.reshape(-1, 1)/255 for image in mnist.train_images()]

# Prepare datasets
training_data = list(zip(train_images[:50000], train_labels[:50000]))
validation_data = (np.asarray([im.ravel() for im in train_images[50000:]]), mnist_train_labels[50000:])

# Initialise network structure
net = nn.Network([nn.Linear([784, 500]), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear([500, 10]), nn.ReLU()])

# Train network
progress = net.sgd(training_data, 30, 5, 0.05, validation_data=validation_data) 