import matplotlib.pyplot as plt
import numpy

from LMS import KLMS, LMS


def gen_test_data():
    # Original binary signal
    s = numpy.random.randint(0, 2, size=(1, 2500))
    s = 2 * s - 1

    # Nonlinear channel
    tmp = numpy.copy(s)
    tmp[1:] = tmp[0:-1]
    tmp[0] = 0
    x = s + 0.5 * tmp
    ns = numpy.random.normal(0, 1, s.shape[1])
    # non linearity
    r = x - 0.9 * (x ** 2) + ns

    # time embedding
    TD = 5
    # equalization time lag
    D = 2

    # data size
    N_tr = 1000
    N_te = 5

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
    lms = LMS(TD, 0.01)
    err1 = get_training_error(lms, X, X_te, T, T_te, TD)
    plt.plot(err1[20:], label='LMS')
    klms = KLMS(X[92], 0.2)
    err2 = get_training_error(klms, X, X_te, T, T_te, TD)
    plt.plot(err2[20:], label='KLMS')
    plt.legend()
    plt.ylabel('MSE')
    plt.xlabel('iteration')
    plt.show()