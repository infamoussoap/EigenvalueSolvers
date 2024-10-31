import numpy as xp


class ConjugateGradient:
    def __init__(self, A, x, alpha_start=1, tol=1e-7):
        self.A = A
        
        self.x = x / xp.sqrt(x @ x)
        self.rho = self.x @ self.A @ self.x
        
        self.alpha_start = alpha_start
        self.x_new = x.copy()

        self.tol = tol
        
    def run(self, num_iters=1000):
        r = self.A @ self.x - self.rho * self.x
        p = r
        for i in range(num_iters):
            flag, self.rho = self.backtrack_line_search(p)
            if flag:
                break
            
            r = self.A @ self.x - self.rho * self.x
            pA = p @ self.A
            
            a1, b1, c1 = pA @ p, 2 * pA @ r, r @ self.A @ r

            delta = b1 ** 2 - 4 * a1 * c1
            if delta < -1e-8 or abs(a1) < 1e-8:
                beta = 0
            else:
                beta = (-b1 + xp.sqrt(max(delta, 0))) / (2 * a1)
            p = r + beta * p
            
        return flag, i + 1, self.rho, self.x
    
    def backtrack_line_search(self, p):
        new_rho = xp.inf

        alpha = self.alpha_start
        counter = 0
        
        while new_rho > self.rho and alpha > self.tol:
            self.x_new[:] = self.x - alpha * p
            self.x_new /= xp.sqrt((self.x_new ** 2).sum())
            
            new_rho = self.x_new @ self.A @ self.x_new

            alpha *= 0.5

            counter += 1

        if alpha <= self.tol:
            return True, self.rho

        self.x[:] = self.x_new
        return False, new_rho