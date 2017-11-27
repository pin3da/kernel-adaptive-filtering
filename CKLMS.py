import numpy as np


class CKLMS():

    def __init__(
        self,
        first_input,
        first_output,
        learning_step=0.5,
        sigma=0.5,
        sigma_cor=0.5
    ):
        self.inputs = [first_input]
        self.errors = [first_output]
        self.learning_step = learning_step
        self.sigma = sigma
        self.sigma_cor = sigma_cor

    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)

    def predict(self, new_input):
        estimate = 0
        for i in range(0, len(self.inputs)):
            term = np.exp(
                -(self.errors[i] * self.errors[i]) /
                (2 * self.sigma_cor * self.sigma_cor))
            ker_eval = self.kernel(self.inputs[i], new_input)
            estimate += (
                self.learning_step * term * self.errors[i] * ker_eval)
        return estimate

    def update(self, new_input, new_output):
        self.errors.append(new_output - self.predict(new_input))
        self.inputs.append(new_input)

    def name(self):
        return 'CKLMS'
