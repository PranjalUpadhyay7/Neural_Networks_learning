input_data= [2.26,1.79 , 1.87 ]
weights= [[-3 ,1 ,6],
          [9,5,4],
          [-7,11,69]]
biases= [0.03, 0.1 , 3]
output_neurons=[]
for bais , weight in zip( biases , weights):
    output_single_neuron=0
    for input_neuron_data , neuron_weight in zip(input_data , weight):
        output_single_neuron+= input_neuron_data*neuron_weight
    output_single_neuron+=bais
    output_neurons.append(output_single_neuron)


print(output_neurons)

