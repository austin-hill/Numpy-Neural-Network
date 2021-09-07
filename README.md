# Overview
This is a neural network written only using numpy. The aim of this project was to gain a proper understanding of the mathematics behind neural networks. It scored an accuracy of 98% on the validation data of the MNIST dataset when trained for 30 epochs.
# Dependencies
```
pip install numpy mnist
```
# Sources
This network is loosely based on the mathematics and algorithms in Michael Nielsen's book, http://neuralnetworksanddeeplearning.com/. However, I have used a batch based approach here to ensure reasonable performance, and used a modular network structure so that I can add features to it easily in the future.