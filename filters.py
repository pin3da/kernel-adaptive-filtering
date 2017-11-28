import numpy as np


class Kernel:
    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)


class CKLMS(Kernel):
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


class KAPA1(Kernel):
    def __init__(self, u_0, d_0, K, eta, sigma):
        self.a = [eta * d_0]
        self.u = [u_0]
        self.d = [d_0]
        self.eta = eta
        self.K = K
        self.sigma = sigma

    def predict(self, x):
        ans = 0
        for i in range(len(self.a)):
            ans += self.a[i] * self.kernel(self.u[i], x)
        return ans

    def update(self, x, d):
        tmp = len(self.a)
        for i in range(max(0, tmp - self.K), tmp):
            y = self.predict(self.u[i])
            e = self.d[i] - y
            self.a[i] += self.eta * e

        error = d - self.predict(x)
        self.u.append(x)
        self.a.append(self.eta * error)
        self.d.append(d)

    def name(self):
        return 'KAPA1'


class KAPA2(Kernel):
    def __init__(
        self,
        first_input,
        first_output,
        learning_step,
        sigma,
        sample_size
    ):
        self.weights = [learning_step * first_output]
        self.inputs = [first_input]
        self.outputs = [first_output]
        self.learning_step = learning_step
        self.sigma = sigma
        self.sample_size = sample_size

    def predict(self, new_input):
        estimate = 0
        for i in range(0, len(self.weights)):
            estimate += self.weights[i] * self.kernel(self.inputs[i], new_input)
        return estimate

    def update(self, new_input, new_output):
        tmp = len(self.weights)
        errors = []
        for k in range(max(0, tmp - self.sample_size), tmp):
            errors.append(self.outputs[k] - self.predict(self.inputs[k]))
        err_vec = np.array(errors)

        tmp2 = min(tmp, self.sample_size)
        G = np.zeros((tmp2, tmp2))
        for i in range(max(0, tmp - self.sample_size), tmp):
            for j in range(max(0, tmp - self.sample_size), tmp):
                i_g = i - max(0, tmp - self.sample_size)
                j_g = j - max(0, tmp - self.sample_size)
                G[j_g][i_g] = self.kernel(self.inputs[i], self.inputs[j])
        G_inv = np.linalg.inv(G + 1e-5 * np.identity(tmp2))
        weight_deltas = G_inv.dot(err_vec).T * self.learning_step
        for i in range(max(0, tmp - self.sample_size), tmp):
            self.weights[i] += (
                weight_deltas[0][i - max(0, tmp - self.sample_size)])

        estimate = self.predict(new_input)
        error = new_output - estimate
        self.inputs.append(new_input)
        self.outputs.append(new_output)
        self.weights.append(self.learning_step * error)

    def name(self):
        return 'KAPA2'


class KLMS(Kernel):
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


class KRLS(Kernel):
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
            [1 / (reg_param + self.kernel(first_input, first_input))])
        self.weights = np.array([self.Q * first_output])
        self.inputs = [first_input]

    def predict(self, new_input):
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

        # reduce the regularization as we get more data
        self.reg_param *= 0.9

    def name(self):
        return 'KRLS'


class LMS:
    def __init__(self, num_params, learning_step):
        self.weights = np.ones(num_params)
        self.learning_step = learning_step

    def predict(self, new_input):
        return self.weights.dot(new_input)

    def update(self, new_input, desired_output):
        prediction_error = desired_output - self.predict(new_input)
        self.weights += self.learning_step * prediction_error * new_input

    def name(self):
        return 'LMS'


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


class RMS:
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


class APA1:
    def __init__(self, u, d, K, eta):
        self.U = np.array([u]).T
        self.D = np.array([d])
        self.w = np.zeros(len(u))  # initial guess
        self.K = K
        self.L = len(u)
        self.eta = eta

    def predict(self, new_input):
        return self.w.T.dot(new_input)

    def update(self, new_input, desired_output):
        tmp = min(len(self.U), self.K)
        U = np.zeros((self.L, tmp + 1))
        D = np.zeros(tmp + 1)

        U[:, :-1] = self.U[:, -tmp:]
        U[:, -1] = new_input.T
        D[: -1] = self.D[-tmp:]
        D[-1] = desired_output

        y = U.T.dot(self.w)
        e = D - y

        self.w += self.eta * U.dot(e)
        self.U = U
        self.D = D

    def name(self):
        return 'APA1'


class APA2:
    def __init__(self, u, d, K, eta, eps=1e-15):
        self.U = np.array([u]).T
        self.D = np.array([d])
        self.w = np.zeros(len(u))  # initial guess
        self.K = K
        self.L = len(u)

        self.eps = eps
        self.eta = eta

    def predict(self, new_input):
        return self.w.T.dot(new_input)

    def update(self, new_input, desired_output):
        tmp = min(len(self.U), self.K)
        U = np.zeros((self.L, tmp + 1))
        D = np.zeros(tmp + 1)

        U[:, :-1] = self.U[:, -tmp:]
        U[:, -1] = new_input.T
        D[: -1] = self.D[-tmp:]
        D[-1] = desired_output

        y = U.T.dot(self.w)
        e = D - y

        t = U.shape[1]
        norm_fact = np.linalg.pinv(self.eps * np.eye(t) + U.T.dot(U))

        self.w += self.eta * U.dot(norm_fact).dot(e)
        self.U = U
        self.D = D

    def name(self):
        return 'APA2'


def get_training_error(adap_filter, X, X_te, T, T_te, TD):
    N_tr = X.shape[0]
    N_te = X_te.shape[0]

    mse = []
    for i in range(N_tr):
        errors = []
        for j in range(N_te):
            errors.append(T_te[j] - adap_filter.predict(X_te[j]))
        errors = np.array(errors)
        mse.append(np.mean(errors ** 2))

        adap_filter.update(X[i], T[i])
    return mse
