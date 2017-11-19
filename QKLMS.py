import numpy as np
from KLMS import KLMS


class QKLMS(KLMS):

    def __init__(
        self,
        num_params,
        first_input=None,
        first_output=None,
        learning_step=0.5,
        min_distance=1,
        sigma=0.5
    ):
        super().__init__(first_input, first_output, learning_step, sigma)
        self.min_distance = min_distance

    def update(self, new_input, expected):
        self.error = expected - self.predict(new_input)
        current_dist = 1e10
        current_index = None
        for i in range(0, len(self.inputs)):
            distance = np.linalg.norm(new_input - self.inputs[i])
            if distance < self.min_distance and distance < current_dist:
                current_dist = distance
                current_index = i
        if current_index is not None:
            self.weights[current_index] += self.learning_step * self.error
        else:
            new_weights = self.learning_step * self.error
            self.inputs.append(new_input)
            self.weights.append(new_weights)

    def name(self):
        return 'QKLMS'
