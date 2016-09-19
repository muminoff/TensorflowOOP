import tensorflow as tf

class NeuralNetwork:

	def __init__(self, input_layer, hidden_1, hidden_2, hidden_3, output_layer, learning_rate, opt_func):
		self.input_node = input_layer
		self.n_classes = output_layer
		self.n_nodes_hl1 = hidden_1
		self.n_nodes_hl2 = hidden_2
		self.n_nodes_hl3 = hidden_3

		self.x = tf.placeholder('float', [None, self.input_node])
		self.y = tf.placeholder('float', [None, self.n_classes])

		self.prediction = self.neural_network_model(self.x)
		self.cost = 0.5 * tf.reduce_sum(tf.sub(self.prediction, self.y) * tf.sub(self.prediction, self.y))

		if opt_func == "adam":
			self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
		else:
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)

		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())

	def neural_network_model(self, data):

		hidden_1_layer = {
			'weights': tf.Variable(tf.random_normal([self.input_node, self.n_nodes_hl1])),
			'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))
		}

		hidden_2_layer = {
			'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
			'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))
		}

		hidden_3_layer = {
			'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
			'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3]))
		}

		output_layer = {
			'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
			'biases': tf.Variable(tf.random_normal([self.n_classes]))
		}

		l1 = tf.nn.softplus(tf.matmul(self.x, hidden_1_layer['weights']) + hidden_1_layer['biases'])

		l2 = tf.nn.softplus(tf.matmul(l1, hidden_2_layer['weights']) + hidden_2_layer['biases'])

		l3 = tf.nn.softplus(tf.matmul(l2, hidden_3_layer['weights']) + hidden_3_layer['biases'])

		output = tf.nn.softplus(tf.matmul(l3, output_layer['weights']) + output_layer['biases'])

		return output

	def train(self, input_data, output_data, epochs):
		hm_epochs = epochs
		for epoch in range(hm_epochs):
			epoch_loss = 0
			epoch_x = input_data
			epoch_y = output_data
			_, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: epoch_x, self.y: epoch_y})
			epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

	def predict(self, input_data):
		p = self.sess.run(self.prediction, feed_dict={self.x: input_data})
		return p[0]