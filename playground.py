from neuralNet import *

input_layer = 2
hidden_1 = 4
hidden_2 = 4
hidden_3 = 4
output_layer = 2

nn = NeuralNetwork(input_layer, hidden_1, hidden_2, hidden_3, output_layer, 0.0005, 'adam', 'reduce_sum1')

def createAndData(i):
	if(i == 0):
		return [[0, 0], [0, 1]]
	elif i == 1:
		return [[1, 0], [0, 1]]
	elif i == 2:
		return [[0, 1], [0, 1]]
	else:
		return [[1, 1], [1, 0]]

for j in range(0, 100):
	for i in range(0, 4):
		data = createAndData(i)
		inputs = data[0]
		target = data[1]
		print inputs
		print target
		nn.train([inputs], [target], 40)

for i in range(0, 4):
	data = createAndData(i)
	inputs = data[0]
	print "inputs", inputs
	pred = nn.predict([inputs])
	for i in range(len(pred)):
		pred[i] = round(pred[i])
	print "predictions:",  pred

