import numpy as np


input_data=[[1, 14,-13,-2],
            [-5, 3, 7 , 0],
            [13, 5, 8, -4]]
weights=[[7.9, 8.3, 5, -15],
         [-4 , 0.03, 4.7, 2.5],
         [1.7, 5.8, -4, -1]]
biases=[7, 3 , 5]
output_data= np.dot( input_data , np.array(input_data).T) + biases
print(output_data)


# we are using transpose to use matrix multiplication for multiplying the input to respective weights and the output of the matrix 
# multiplication will be 3*3 matix as 3 input batches are inserted to pass through the 3 neurons . The biases will then be added to 
# the output matrix .