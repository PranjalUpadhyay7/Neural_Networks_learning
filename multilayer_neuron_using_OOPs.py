import numpy as np
np.random.seed(0)


X=[[1, 14,-13,-2],
    [-5, 3, 7 , 0],
    [13, 5, 8, -4]]


class Layer_Dense:
    def __init__(self , number_of_input , number_of_neurons ):
        self.weights = np.random.randn(number_of_input , number_of_neurons)
        self.biases= np.random.randint(0, 10, size=(1, number_of_neurons))
    def forward(self , input_data):
        self.output= np.dot(input_data, self.weights ) + self.biases

layer1= Layer_Dense(4, 4)
layer1.forward(X)
layer1_output= layer1.output
print(layer1_output, "\n")
layer2= Layer_Dense(4,10)
layer2.forward(layer1_output)
layer2_output= layer2.output
print(layer2_output)
