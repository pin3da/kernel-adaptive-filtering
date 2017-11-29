import time
import contextlib

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

from filters import (APA1, APA2, APA3, APA4, CKLMS, KAPA1, KAPA2, KAPA3, KAPA4,
                     KLMS, KRLS, LMS, QKLMS, get_training_error)


def gen_lorenz_data():
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def f(state, t):
        x, y, z = state  # unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0.0, 40.0, 0.01)

    states = odeint(f, state0, t)
    return states


def split_training_and_testing(states):
    # time embedding
    TD = 5

    # dataset size
    N_tr = 250
    N_te = 10

    # train data
    X = np.zeros((N_tr, TD))
    for k in range(N_tr):
        X[k, :] = states[k: k + TD]

    # test data
    X_te = np.zeros((N_te, TD))
    for k in range(N_te):
        X_te[k, :] = states[k + N_tr: k + TD + N_tr]

    # Desired signal
    T = np.zeros((N_tr, 1))
    for i in range(N_tr):
        T[i, 0] = states[i]

    # Desire signal testing
    T_te = np.zeros((N_te, 1))
    for i in range(N_te):
        T_te[i, 0] = states[i + N_tr]

    return X, X_te, T, T_te, TD


def plot_lorenz(states):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    plt.show()


@contextlib.contextmanager
def time_it(name):
    start = time.monotonic()
    print(name, end=' ')
    yield
    print('took %f seconds' % (time.monotonic() - start))


if __name__ == '__main__':
    states = gen_lorenz_data()
    # plot_lorenz(states)

    states = np.array([
        x for x, y, z in states
    ])

    X, X_te, T, T_te, TD = split_training_and_testing(states)
    filters = [
        LMS(TD, 0.0001),
        APA1(X[0], T[0], 10, 0.0001),
        APA2(X[0], T[0], 10, 0.02),
        APA3(X[0], T[0], 10, 0.0001, 0.01),
        APA4(X[0], T[0], 10, 0.005),
        # KLMS(TD, X[0], T[0], 0.034, 20),
        # CKLMS(X[0], T[0], 0.03, 10, 10),
        # QKLMS(TD, X[0], T[0], 0.03, 1, 10),
        # KAPA1(X[0], T[0], 10, 0.3, 5),
        # KAPA2(X[0], T[0], 0.2, 2.25, 10),
        # KAPA3(X[0], T[0], 0.1, 2.25, 10),
        # KAPA4(X[0], T[0], 0.2, 2.25, 10),
        # KRLS(X[0], T[0], 1, 1),
    ]
    for fi in filters:
        with time_it(fi.name()):
            err = get_training_error(fi, X, X_te, T, T_te, TD)
        plt.plot(err[50:], label=fi.name())

    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('iteration')
    plt.title('Lorenz oscillator')
    # plt.savefig('./compare-lorenz-2.png')
    plt.show()
