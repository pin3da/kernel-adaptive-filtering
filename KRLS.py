import numpy as np


class KRLS():

    def __init__(
        self,
        first_input,
        first_output,
        reg_param=0.5,
        sigma=0.5
    ):
        self.reg_param = reg_param
        self.sigma = sigma
        self.Q = np.array(
            [1 / reg_param * self.kernel(first_input, first_input)])
        self.weights = np.array([self.Q * first_output])
        self.inputs = [first_input]

    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)

    def predict(self, new_input):
        answer = 0
        h = np.array([
            self.kernel(new_input, old_input) for old_input in self.inputs
        ]).reshape(1, len(self.inputs))
        return h.dot(self.weights)

    def update(self, new_input, expected):
        h = np.array([
            self.kernel(new_input, old_input) for old_input in self.inputs
        ]).reshape(len(self.inputs), 1)
        ht = h.T
        z = self.Q.dot(h)
        zt = z.T
        r = self.reg_param + self.kernel(new_input, new_input) - zt.dot(h)
        Q_size = len(z) + 1
        new_Q = np.zeros((Q_size, Q_size))
        new_Q[0:Q_size - 1, 0:Q_size - 1] = self.Q * r + z.dot(zt)
        new_Q[0:Q_size - 1, Q_size - 1:Q_size] = -z
        new_Q[Q_size - 1:Q_size, 0:Q_size - 1] = -zt
        new_Q[Q_size - 1][Q_size - 1] = 1
        self.Q = new_Q
        error = expected - ht.dot(self.weights)
        new_weights = np.zeros((Q_size, 1))
        new_weights[0:Q_size - 1, 0:1] = self.weights - z * (1 / r) * error
        new_weights[Q_size - 1][0] = (1 / r) * error
        self.weights = new_weights
        self.inputs.append(new_input)

    def name(self):
        return 'KRLS'
