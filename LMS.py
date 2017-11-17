import numpy as np
import math


class LMS():

    def __init__(self, num_params, learning_step):
        self.weights = np.ones(num_params)
        self.learning_step = learning_step

    def predict(self, new_input):
        return self.weights.dot(new_input)

    def update_weights(self, new_input, desired_output):
        prediction_error = desired_output - self.predict(new_input)
        self.weights += self.learning_step * prediction_error.dot(new_input)


# Sine
input_data = np.arange(0, 2 * math.pi, math.pi / 10)
output_data = np.sin(input_data)

lms = LMS(1, 0.1)

for i in range(0, len(input_data)):
    print('Prediction: ' + lms.predict(input_data[i]).__str__())
    print('Wanted: ' + output_data[i].__str__())
    print('Current Weight:' + lms.weights.__str__())
    print('------')
    lms.update_weights(input_data[i], output_data[i])
