import numpy as np


def gradient_descent(A, x, lr, which='max', maxiter=100, tol=1e-8, save_history=False, subspace=None):
    assert which in ['max', 'min'], "Only 'max' and 'min' valid for `which`"

    if save_history:
        history = np.zeros(maxiter)

    if subspace is not None:
        assert len(subspace.shape) == 2, ("Subspace assumed to be a matrix of shape (d, m) where d = len(x), the "
                                          "dimension of the problem.")
        assert len(subspace) == len(x), ("Subspace assumed to be of shape (d, m) where d = len(x), the dimension of "
                                         "the problem.")

    sign = -1 if which == 'min' else 1

    Ax = A @ x

    rho = x @ Ax
    counter = 0
    for i in range(maxiter):
        old_rho = rho

        x += sign * lr * Ax

        if subspace is not None:
            x -= (x @ subspace) @ subspace.T

        x /= np.linalg.norm(x)

        Ax = A @ x
        rho = x @ Ax

        if abs(rho - old_rho) < tol:
            break

        if save_history:
            history[i] = rho
            counter += 1

    if save_history:
        return rho, x, history[:counter]

    return rho, x


def power_iteration(A, x, maxiter=100, tol=1e-8, save_history=False):
    if save_history:
        history = np.zeros(maxiter)

    Ax = A @ x
    rho = x @ Ax
    counter = 0
    for i in range(maxiter):
        old_rho = rho

        x[:] = Ax
        x /= np.linalg.norm(x)

        Ax = A @ x
        rho = x @ Ax

        if abs(rho - old_rho) < tol:
            break

        if save_history:
            history[i] = rho
            counter += 1

    if save_history:
        return rho, x, history[:counter]

    return rho, x


def nesterov_descent(A, x, lr, which='max', maxiter=100, tol=1e-8, save_history=False, subspace=None):
    assert which in ['max', 'min'], "Only 'max' and 'min' valid for `which`"
    if save_history:
        history = np.zeros(maxiter)

    if subspace is not None:
        assert len(subspace.shape) == 2, ("Subspace assumed to be a matrix of shape (d, m) where d = len(x), the "
                                          "dimension of the problem.")
        assert len(subspace) == len(x), ("Subspace assumed to be of shape (d, m) where d = len(x), the dimension of "
                                         "the problem.")

    lambda_old = 0
    lambda_new = _get_lambda_new(lambda_old)

    sign = -1 if which == 'min' else 1

    y = x.copy()
    old_y = np.zeros_like(y)

    Ax = A @ x

    rho = x @ Ax
    counter = 0
    for i in range(maxiter):
        old_rho = rho
        old_y[:] = y

        gamma = (1 - lambda_old) / lambda_new
        np.add(x, sign * lr * Ax, out=y)
        np.add((1 - gamma) * y, gamma * old_y, out=x)

        if subspace is not None:
            x -= (x @ subspace) @ subspace.T

        x /= np.linalg.norm(x)

        Ax[:] = A @ x
        rho = x @ Ax

        lambda_old = lambda_new
        lambda_new = _get_lambda_new(lambda_old)

        if abs(rho - old_rho) < tol:
            break

        if save_history:
            history[i] = rho
            counter += 1

    if save_history:
        return rho, x, history[:counter]

    return rho, x


def _get_lambda_new(lambda_old):
    lambda_new = (1 + np.sqrt(1 + 4 * lambda_old ** 2)) / 2
    return lambda_new
