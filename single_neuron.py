input_data=[1,2,3,4,5]
weights=[4,5,6,7,8]
bias=9
neuron_output= 0
for i in range(len(input_data)):
    neuron_output += input_data[i] * weights[i]

neuron_output += bias
print(neuron_output)