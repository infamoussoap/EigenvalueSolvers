import numpy as xp


class RMSProp:
    def __init__(self, A, x, alpha_start=1, tol=1e-7, beta=0.1):
        self.A = A
        
        self.x = x / xp.sqrt(x @ x)
        self.rho = self.x @ self.A @ self.x
        self.g2 = xp.zeros_like(x)
        
        self.alpha_start = alpha_start
        self.x_new = x.copy()
        
        self.beta = beta

        self.tol = tol
        
    def run(self, num_iters=1000):
        for i in range(num_iters):
            grad = self.A @ self.x - self.rho * self.x
            self.g2[:] = self.beta * self.g2 + (1 - self.beta) * grad ** 2
            
            flag, self.rho = self.backtrack_line_search(grad / xp.sqrt(self.g2))
            if flag:
                break
            
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
