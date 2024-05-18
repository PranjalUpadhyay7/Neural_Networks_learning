import numpy as np

# basic method to make 2 layer neurons , take output of first neuron and give it as input to the second neuron layer.
# also don't forget to maintain the weights2 shape corresponding to the shape of the output of the first neuron layer.
input_data=[[1, 14,-13,-2],
            [-5, 3, 7 , 0],
            [13, 5, 8, -4]]
weights1=[[7.9, 8.3, 5, -15],
         [-4 , 0.03, 4.7, 2.5],
         [1.7, 5.8, -4, -1]]
biases1=[7, 3 , 5]

layer1_output= np.dot(input_data , np.array(weights1).T) + biases1

weights2=[[0.2,0.04,0.56],
          [-0.4, 1.3,-2.1],
          [0.45,0.25, -0.15]]
biases2=[6,6,4]

layer2_output=np.dot(layer1_output , np.array(weights2).T) + biases2

print(layer2_output)