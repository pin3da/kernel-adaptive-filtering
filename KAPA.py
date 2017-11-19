import numpy as np


class KAPA1:

    def __init__(self, u_0, d_0, K, eta, sigma):
        self.a = [eta * d_0]
        self.u = [u_0]
        self.d = [d_0]
        self.eta = eta
        self.K = K
        self.sigma = sigma

    def kernel(self, a, b):
        norm = np.linalg.norm(a - b)
        term = (norm * norm) / (2 * self.sigma * self.sigma)
        return np.exp(-1 * term)

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
