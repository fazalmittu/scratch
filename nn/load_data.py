from keras.datasets import mnist
import numpy as np

def load_mnist():
	#loading the dataset
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

if __name__ == "__main__": 

	X_train, y_train, X_test, y_test = load_mnist()

	y_train = one_hot_encode(y_train)

	print(y_train[0])
	print(y_train[0].shape)

	# printing the shapes of the vectors 
	print('X_train: ' + str(X_train.shape))
	print('y_train: ' + str(y_train.shape))
	print('X_test:  ' + str(X_test.shape))
	print('y_test:  ' + str(y_test.shape))

