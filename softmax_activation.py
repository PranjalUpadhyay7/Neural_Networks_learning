# we are using the softmax activation function to use the concepts of probability
import math
import numpy as np
E = math.e

X=[[5, 14,-13,-2],
    [-5, 3, 7 , 0],
    [13, 5, 8, -4]]
output1=[[],[],[]]
output2=[[],[],[]]
# Basic approach to implement Softmax

for i in range(len(X)):
    for j in range(len(X[0])):
        
        output1[i].append((E**X[i][j]))
norm_base= []
for i in range(len(X)):
    norm_base.append(sum(output1[i]))
for i in range(len(X)):
    for j in range(len(X[0])):
        
        output2[i].append((output1[i][j]/norm_base[i]))

# print(output1)
print(output2)

# we are using numpy to implement the softmax activation function
output3=[]
output3=np.exp(X)

output4= output3/np.sum(output3, axis=1, keepdims=True)
print(output4)
#class of softmax function 

class softmax_activation:
    def forward(self, input):
        output5= np.exp(input- np.max(input, axis=1 , keepdims=True))
        output6 =output5/np.sum(output5, axis=1, keepdims=True)
        self.out=output6

softmax1= softmax_activation()
softmax1.forward(X)
print(softmax1.out)