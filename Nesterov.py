import numpy as xp
import warnings


class Nesterov:
    def __init__(self, A, x, lr=100, save_history=False, which='min', tol=1e-8):
        self.A = A
        self.x = x / xp.sqrt(xp.dot(x, x))
        self.Ax = self.A @ self.x
        self.rho = xp.dot(self.x, self.Ax)

        self.y = self.x.copy()
        self.y_prev = xp.zeros_like(self.x)

        self.lr = lr

        self.save_history = save_history
        self.history = None

        self.which = which

        self.tol = tol
        
        self.lambda_old = 0
        self.lambda_new = 0
        self.update_lambda_new()

    def get_extreme_eigvals(self, num_iters=1000, max_lr=100, alpha=0.99):
        assert not self.save_history, 'save_history must be false'

        max_flag, max_eig_val = self._get_which_eigval(num_iters, max_lr, 'max')
        if max_flag:
            warnings.warn('max-eigval has not converged')

        lr = alpha / (2 * max_eig_val)
        min_flag, min_eig_val = self._get_which_eigval(num_iters, lr, 'min')

        if min_flag:
            warnings.warn('min-eigval has not converged')

        return (min_flag, max_flag), min_eig_val, max_eig_val

    def _get_which_eigval(self, num_iters, lr, which):
        x_start, lr_start, which_start = self.x.copy(), self.lr, self.which

        self.lr = lr
        self.which = which

        flag, eig_val, _ = self.run(num_iters=num_iters)

        self._reset(x_start, lr_start, which_start)

        return flag, eig_val

    def _reset(self, x_start, lr_start, which_start):
        self.lr = lr_start
        self.which = which_start

        self.x = x_start / xp.sqrt(xp.dot(x_start, x_start))
        self.Ax = self.A @ self.x
        self.rho = xp.dot(self.x, self.Ax)

        self.y = self.x.copy()
        self.y_prev[:] = 0

        self.lambda_old = 0
        self.lambda_new = 0
        self.update_lambda_new()
        
    def update_lambda_new(self):
        self.lambda_new = (1 + xp.sqrt(1 + 4 * self.lambda_old ** 2)) / 2

    def _initialize_history(self, num_iters):
        if self.save_history:
            self.history = xp.zeros(num_iters)

    def _update_history(self, i, val):
        if self.save_history:
            self.history[i] = val

    def _truncate_history(self, i):
        if self.save_history:
            self.history = self.history[:i + 1].copy()

    def run(self, num_iters=1000):
        self._initialize_history(num_iters)
        not_converged = True
        for i in range(num_iters):
            gamma = (1 - self.lambda_old) / self.lambda_new

            self.y_prev[:] = self.y

            if self.which == 'min':
                xp.subtract(self.x, 2 * self.lr * self.Ax, out=self.y)
            else:
                xp.add(self.x, self.lr * self.Ax, out=self.y)
            xp.add((1 - gamma) * self.y, gamma * self.y_prev, out=self.x)

            self.x /= xp.sqrt(xp.dot(self.x, self.x))

            self.Ax[:] = self.A @ self.x

            new_rho = xp.dot(self.x, self.Ax)
            if abs(new_rho - self.rho) < self.tol and i > 0:  # Optimizer might not move on first iteration
                print(self.which, i, new_rho, self.rho)
                self.rho = new_rho
                self._update_history(i, self.rho)
                self._truncate_history(i)
                not_converged = False
                break

            self.rho = new_rho
                
            self.lambda_old = self.lambda_new
            self.update_lambda_new()
            
            self._update_history(i, self.rho)
            
        return not_converged, self.rho, self.x
