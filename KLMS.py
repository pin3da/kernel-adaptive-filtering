import numpy as np


class KLMS():

    def __init__(
        self,
        num_params,
        first_input=None,
        first_output=None,
        learning_step=0.5,
        sigma=0.5
    ):
        if first_input is not None:
            self.inputs = [first_input]
        else:
            self.inputs = [np.zeros(num_params)]
        if first_output is not None:
            self.weights = [first_output * learning_step]
        else:
            self.weights = [0]
        self.learning_step = learning_step
        self.sigma = sigma
        self.error = None

    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)

    def predict(self, new_input):
        estimate = 0
        for i in range(0, len(self.weights)):
            addition = self.weights[i] * self.kernel(self.inputs[i], new_input)
            estimate += addition
        return estimate

    def update(self, new_input, expected):
        self.error = expected - self.predict(new_input)
        self.inputs.append(new_input)
        new_weights = self.learning_step * self.error
        self.weights.append(new_weights)

    def name(self):
        return 'KLMS'
