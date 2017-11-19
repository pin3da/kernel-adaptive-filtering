import numpy as np
import KLMS from KLMS


class QKLMS(KLMS):

    def __init__(
        self,
        num_params,
        first_input=None,
        first_output=None,
        learning_step=0.5,
        min_distance=1
        sigma=0.5
    ):
        super().__init__(first_input, first_output, learning_step, sigma)
        self.min_distance = min_distance

    def update(self, new_input, expected):
        self.error = expected - self.predict(new_input)
        distance = np.absolute(np.linalg.norm(new_input, self.inputs[-1]))
        if distance < self.min_distance:
            self.weights[-1] += self.learning_step * self.error
        else:
            new_weights = self.weights[-1] + self.learning_step * self.error
            self.inputs.append(new_input)
            self.weights.append(new_weights)
