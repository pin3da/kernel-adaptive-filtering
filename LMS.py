import numpy as np


class LMS:

    def __init__(self, num_params, learning_step):
        self.weights = np.ones(num_params)
        self.learning_step = learning_step

    def predict(self, new_input):
        return self.weights.dot(new_input)

    def update(self, new_input, desired_output):
        prediction_error = desired_output - self.predict(new_input)
        self.weights += self.learning_step * prediction_error * new_input


def kernel(x, y):
    dist = x - y
    dist = dist.dot(dist)
    return np.exp(-dist * 0.1)


class KLMS:
    def __init__(self, x, learning_step):
        self.alpha = [learning_step * x]
        self.u = [x]
        self.learning_step = learning_step

    def predict(self, x):
        ans = 0
        for i in range(len(self.alpha)):
            ans += self.alpha[i] * kernel(self.u[i], x)
        return ans

    def update(self, x, y):
        error = y - self.predict(x)
        self.alpha.append(self.learning_step * error)
        self.u.append(x)
