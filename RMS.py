import numpy as np
import math


class RMS():

    def __init__(self, num_params):
        self.num_params = num_params
        self.weights = np.ones(num_params)
        self.previous_inputs = []

    def predict(self, new_input):
        return self.weights.dot(new_input)

    def calc_time_avg_corr(self, new_input):
        R = np.eye(self.num_params)
        for input_data in self.previous_inputs:
            row = input_data.reshape(self.num_params, 1)
            col = input_data.reshape(1, len(input_data))
            R += row.dot(col)
        new_row = new_input.reshape(self.num_params, 1)
        new_col = new_input.reshape(1, self.num_params)
        R += new_row.dot(new_col)
        return R

    def calc_gain_vector(self, new_input):
        R = self.calc_time_avg_corr(new_input)
        return np.linalg.inv(R).dot(new_input)

    def update_weights(self, new_input, desired_output):
        prediction_error = desired_output - self.predict(new_input)
        k = self.calc_gain_vector(new_input)
        self.weights += k.dot(prediction_error)


# Sine
input_data = np.arange(0, 2 * math.pi, math.pi / 10)
output_data = np.sin(input_data)

rms = RMS(1)

for i in range(0, len(input_data)):
    print('Prediction: ' + rms.predict(input_data[i]).__str__())
    print('Wanted: ' + output_data[i].__str__())
    print('Current Weight:' + rms.weights.__str__())
    print('------')
    rms.update_weights(input_data[i], output_data[i])
