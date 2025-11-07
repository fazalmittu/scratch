from load_data import load_mnist, one_hot_encode
import numpy as np

class NN():
	def __init__(
		self,
		X_train, 
		y_train,
		X_test,
		y_test,
		epochs=200,
		learning_rate=0.1,
		verbose=True
	):
		self.X_train = X_train / 255
		self.y_train = y_train
		self.X_test = X_test / 255
		self.y_test = y_test
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.trained = False
		self.verbose = verbose

		self.W1 = np.random.rand(10, self.X_train.shape[0])	- 0.5
		self.b1 = np.random.rand(self.y_train.shape[0], 1) - 0.5
		self.W2 = np.random.rand(10, self.y_train.shape[0]) - 0.5
		self.b2 = np.random.rand(self.y_train.shape[0], 1) - 0.5


		if verbose:
			print('X_train: ' + str(X_train.shape))
			print('y_train: ' + str(y_train.shape))
			print('X_test:  ' + str(X_test.shape))
			print('y_test:  ' + str(y_test.shape))

			print('W1: ' + str(self.W1.shape))
			print('b1: ' + str(self.b1.shape))
			print('W2:  ' + str(self.W2.shape))
			print('b2:  ' + str(self.b2.shape))

	def relu(self, x: np.ndarray):
		return np.maximum(x, 0)

	def d_relu(self, x: np.ndarray):
		return x > 0

	def softmax(self, x: np.ndarray):
		return np.exp(x) / sum(np.exp(x))

	def forward_prop(
		self,
		X: np.ndarray,
		y: np.ndarray
	):
		H1 = self.W1.dot(X) + self.b1
		H1_nonlinear = self.relu(H1)
		H2 = self.W2.dot(H1_nonlinear) + self.b2
		H2_nonlinear = self.softmax(H2)

		return H1, H1_nonlinear, H2, H2_nonlinear

	def back_prop(
		self,
		H1: np.ndarray, 
		H1_nonlinear: np.ndarray,
		H2: np.ndarray,
		H2_nonlinear: np.ndarray,
		X: np.ndarray, 
		y: np.ndarray, 
	):
		y_size = y.shape[1]
		output_diff = H2_nonlinear - y
		W2_diff = 1/y_size * output_diff.dot(H1_nonlinear.T)
		b2_diff = 1/y_size * np.sum(output_diff)
		second_layer_diff = self.W2.T.dot(output_diff) * self.d_relu(H1)
		W1_diff = 1/y_size * second_layer_diff.dot(X.T)
		b1_diff = 1/y_size * np.sum(second_layer_diff)

		return W1_diff, b1_diff, W2_diff, b2_diff

	def update_weights(
		self,
		W1_diff, 
		b1_diff, 
		W2_diff, 
		b2_diff
	):
		self.W1 = self.W1 - W1_diff * self.learning_rate
		self.b1 = self.b1 - b1_diff * self.learning_rate
		self.W2 = self.W2 - W2_diff * self.learning_rate
		self.b2 = self.b2 - b2_diff * self.learning_rate

	def get_predictions(self, y_hat: np.ndarray):
		return np.argmax(y_hat, 0)

	def get_accuracy(self, predictions: np.ndarray, y: np.ndarray):
		# return np.sum(predictions == y) / y.shape[1]
		return np.sum(predictions == np.argmax(y, axis=0)) / y.shape[1]

	def gradient_descent(self):
		for i in range(self.epochs):
			H1, H1_nonlinear, H2, H2_nonlinear = self.forward_prop(self.X_train, self.y_train)
			W1_diff, b1_diff, W2_diff, b2_diff = self.back_prop(H1, H1_nonlinear, H2, H2_nonlinear, self.X_train, self.y_train)

			self.update_weights(W1_diff, b1_diff, W2_diff, b2_diff)

			if i % 50 == 0:
				print(f"Epoch: {i}")
				predictions = self.get_predictions(H2_nonlinear)
				accuracy = self.get_accuracy(predictions, self.y_train)
				print(f"Accuracy: {accuracy}")

		print(f"Epoch {self.epochs}")
		predictions = self.get_predictions(H2_nonlinear)
		accuracy = self.get_accuracy(predictions, self.y_train)
		print(f"Accuracy: {accuracy}")

		self.trained = True

		return self.W1, self.b1, self.W2, self.b2

if __name__ == "__main__":

	X_train, y_train, X_test, y_test = load_mnist()

	# reshape (60000, 28, 28) where it's (#num_samples, x, y)
	# -> (60000, 784)
	# then transpose -> (784, 60000)

	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] ** 2)).T
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] ** 2)).T

	y_train = one_hot_encode(y_train)
	y_test = one_hot_encode(y_test)

	nn = NN(
		X_train=X_train,
		y_train=y_train,
		X_test=X_test,
		y_test=y_test,
		epochs=500,
		learning_rate=0.5,
		verbose=True
	)

	W1, b1, W2, b2 = nn.gradient_descent()