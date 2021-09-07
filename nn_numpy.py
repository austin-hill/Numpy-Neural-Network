import numpy as np
import random

class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)+1
        self.layers = layers
    
    # Stochastic Gradient Descent
    def sgd(self, training_set, epochs, m, eta, reg, validation_data):
        n = len(training_set)
        progress = np.zeros(epochs, dtype=int)

        for j in range(epochs):
            random.shuffle(training_set)
            mini_batches = [training_set[k:k+m] for k in range(0, n, m)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta, reg, n)
            if validation_data:
                progress[j] = self.evaluate(validation_data)
                print('Epoch {}: {}/{}'.format(j+1, progress[j], len(validation_data)))
            else:
                print('Epoch {} complete'.format(j+1))

        return progress
    
    # Calculate gradients and update weights
    def update_batch(self, mini_batch, eta, reg, n):
        a = np.asarray([ims.ravel() for ims,_ in mini_batch])
        labels = np.asarray([labs.ravel() for _,labs in mini_batch])
        m = len(mini_batch)

        # Forward pass
        for layer in self.layers:
            a = layer.forward_batch(a, m)

        # Backward pass
        delta = a - labels # We are using a cross entropy cost function
        for layer in self.layers[-2::-1]:
            delta = layer.backprop(delta)
        for layer in self.layers:
            layer.update_weights(eta, reg, m, n)
    
    def forward(self, a):
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def evaluate(self, validation_data):
        test_results = [(np.argmax(self.forward(t[0])), t[1]) for t in validation_data]
        return sum(int(x==y) for (x, y) in test_results)

class Linear(object):
    '''Linear (fully connected) layer'''
    def __init__(self, size):
        self.biases = np.random.randn(size[1], 1)
        self.weights = np.random.randn(size[1], size[0])/np.sqrt(size[0])
    
    def forward(self, a):
        return np.dot(self.weights, a)+self.biases
    
    def forward_batch(self, a, m):
        self.last_input = a
        self.last_z = a @ self.weights.T + np.squeeze(np.array([self.biases for _ in range(m)]), axis=2)
        return self.last_z

    def backprop(self, delta):
        nb = delta.T
        nw = delta.T @ self.last_input
        delta = delta @ self.weights
        nb = np.asarray([sum(j) for j in nb]).reshape(-1,1)
        self.last_nb, self.last_nw = nb, nw
        return delta
    
    def update_weights(self, eta, reg, m, n):
        self.biases -= (eta/m)*self.last_nb
        self.weights = (1-eta*reg/n)*self.weights-(eta/m)*self.last_nw


class Sigmoid(object):
    '''Sigmoid loss function'''
    def forward(self, z):
        return self.sigmoid(z)
    
    def forward_batch(self, z, _):
        self.last_z = z
        return self.sigmoid(self.last_z)
    
    def backprop(self, delta):
        return delta * self.sigmoid_prime(self.last_z)
    
    def update_weights(self, eta, reg, m, n):
        pass

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))