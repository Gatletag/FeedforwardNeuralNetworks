import numpy as np

class Perceptron():
    def __init__(self, num_inputs):
        self.weights = 2 * np.random.random((num_inputs, 1)) - 1

    # Sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # forward pass
    def forward_pass(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

    # error calculation based on predicted values and true values
    def error_calculation(self, true_outputs, predicted_outputs):
        error = true_outputs - predicted_outputs
        return error

    # train the perceptror over several iterations
    # requires an input set with the output true labels
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate):
        for iteration in range(number_of_training_iterations):
            sum_error =0
            for input, true_output in zip(training_set_inputs, training_set_outputs):
                output = self.forward_pass(input)

                error = self.error_calculation(true_output,output)

                # multiplying by input as it is the contribution it has on the error.
                self.weights[:,0] += learning_rate*error[0]*input
                sum_error+=error[0]
            print(sum_error)


if __name__ == "__main__":

    perceptron = Perceptron(3)

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T

    perceptron.train(training_set_inputs, training_set_outputs, 10000,0.1)

    print("New synaptic weights after training: ")
    print(perceptron.weights)

    # Test the neural network with a new situation.
    print("Considering new situation [1, 0, 0] -> ?: ")
print(perceptron.forward_pass(np.array([0, 0, 1])))
print(perceptron.forward_pass(np.array([1, 1, 1])))
print(perceptron.forward_pass(np.array([1, 0, 1])))
print(perceptron.forward_pass(np.array([0, 1, 0])))
