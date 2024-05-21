# the basic approach of making activation function ReLU
X=[[1, 14,-13,-2],
    [-5, 3, 7 , 0],
    [13, 5, 8, -4]]
output1=[[],[],[]]
output2=[[],[],[]]

for i in range(len(X)):
    for j in range(len(X[0])):
        
        if X[i][j]>0 :
            
            output1[i].append(X[i][j])
        else:
            output1[i].append(0)
print(output1)

# Activation fuction using max() function

for i in range(len(X)):
    for j in range(len(X[0])):
        output2[i].append(max(0, X[i][j]))
print(output2)


#class of activation functions 

import numpy as np



class Layer_Dense:
    def __init__(self , number_of_input , number_of_neurons ):
        self.weights = np.random.randn(number_of_input , number_of_neurons)
        self.biases= np.random.randint(0, 10, size=(1, number_of_neurons))
    def forward(self , input_data):
        self.output= np.dot(input_data, self.weights ) + self.biases


class ActivationFunction:
    def forward(self, input):
        self.output= np.maximum(0, input)

layer1= Layer_Dense(4, 6)
layer1.forward(X)
Activation1= ActivationFunction()
Activation2= ActivationFunction()
Activation2.forward(X)
print(Activation2.output)

Activation1.forward(layer1.output)
print(layer1.output)
print(Activation1.output)