import matplotlib.pyplot as plt
import numpy

from LMS import LMS
from KLMS import KLMS
<<<<<<< HEAD
from KAPA import KAPA1
=======
from QKLMS import QKLMS
>>>>>>> Fixes QKLMS and adds it to tests.


def gen_test_data():
    # Original binary signal
    s = numpy.random.randint(0, 2, size=(1, 2500))
    s = 2 * s - 1

    # Nonlinear channel
    tmp = numpy.copy(s)
    tmp[1:] = tmp[0:-1]
    tmp[0] = 0
    x = s + 0.5 * tmp
    ns = 0.4 * numpy.random.normal(0, 1, s.shape[1])
    # non linearity
    r = x - 0.9 * (x ** 2) + ns

    # time embedding
    TD = 5
    # equalization time lag
    D = 2

    # data size
    N_tr = 300
    N_te = 10

    # train data
    X = numpy.zeros((N_tr, TD))
    for k in range(N_tr):
        X[k, :] = r[0, k: k + TD]

    # test data
    X_te = numpy.zeros((N_te, TD))
    for k in range(N_te):
        X_te[k, :] = r[0, k + N_tr: k + TD + N_tr]

    # Desired signal
    T = numpy.zeros((N_tr, 1))
    for i in range(N_tr):
        T[i, 0] = s[0, D + i]

    # Desire signal testing
    T_te = numpy.zeros((N_te, 1))
    for i in range(N_te):
        T_te[i, 0] = s[0, D + i + N_tr]

    return X, X_te, T, T_te, TD


def gen_sin_data():
    X = numpy.arange(0, 2 * numpy.pi, numpy.pi / 1000)
    T = numpy.sin(X)
    X_te = numpy.arange(0, 2 * numpy.pi, numpy.pi / 10)
    T_te = numpy.sin(X_te)
    return X, X_te, T, T_te, 1


def get_training_error(adap_filter, X, X_te, T, T_te, TD):
    N_tr = X.shape[0]
    N_te = X_te.shape[0]

    mse = []
    for i in range(N_tr):
        errors = []
        for j in range(N_te):
            errors.append(T_te[j] - adap_filter.predict(X_te[j]))
        errors = numpy.array(errors)
        mse.append(numpy.mean(errors ** 2))

        adap_filter.update(X[i], T[i])
    return mse


if __name__ == '__main__':

    X, X_te, T, T_te, TD = gen_test_data()
    filters = [
        LMS(TD, 0.01),
        KLMS(TD, X[0], T[0], 0.2, 2.25),
        QKLMS(TD, X[0], T[0], 0.2, 0.225, 2.25),
        KAPA1(X[0], T[0], 10, 0.2, 2.25)
    ]
    for fi in filters:
        err = get_training_error(fi, X, X_te, T, T_te, TD)
        plt.plot(err[20:], label=fi.name())

    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('iteration')
    # plt.savefig('./compare.png')
    plt.show()
