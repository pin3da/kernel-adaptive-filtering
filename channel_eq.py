import matplotlib.pyplot as plt
import numpy

from filters import (APA1, APA2, APA3, APA4, CKLMS, KAPA1, KAPA2, KAPA3, KAPA4,
                     KLMS, KRLS, LMS, QKLMS, get_training_error)


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

    # dataset size
    N_tr = 300
    N_te = 20

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


if __name__ == '__main__':
    # seed = numpy.random.randint(0, 1243464)
    seed = 337980  # this is to reproduce the results
    print('seed', seed)
    numpy.random.seed(seed)

    X, X_te, T, T_te, TD = gen_test_data()
    filters = [
        LMS(TD, 0.01),
        APA1(X[0], T[0], 10, 0.001),
        APA2(X[0], T[0], 10, 0.01),
        APA3(X[0], T[0], 10, 0.002, 0.1),
        APA4(X[0], T[0], 10, 0.01),
        KLMS(TD, X[0], T[0], 0.2, 2.25),
        CKLMS(X[0], T[0], 0.2, 2.25, 2),
        QKLMS(TD, X[0], T[0], 0.2, 0.225, 2.25),
        KRLS(X[0], T[0], 0.5, 2.25),
        KAPA1(X[0], T[0], 10, 0.2, 2.25),
        KAPA2(X[0], T[0], 0.2, 2.25, 10),
        KAPA3(X[0], T[0], 0.2, 2.25, 10),
        KAPA4(X[0], T[0], 0.2, 2.25, 10),
    ]
    for fi in filters:
        err = get_training_error(fi, X, X_te, T, T_te, TD)
        plt.plot(err[20:], label=fi.name())

    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('iteration')
    plt.title('Non linear channel equalization')
    # plt.savefig('./compare2.png')
    plt.show()
