# TensorFlow OOP

This is an object oriented approach to Tensorflow

# Getting Started

Initialize the neural net:

input_layer = 2

hidden_1 = 4

hidden_2 = 4

hidden_3 = 4

output_layer = 2

nn = NeuralNetwork(input_layer, hidden_1, hidden_2, hidden_3, output_layer, 0.05, 'grad', 'reduce_sum') 

# Training

nn.train([inputs], [target], epochs)

# Predictions

pred = nn.predict([inputs])
