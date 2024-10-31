import numpy as xp


class GD:
    def __init__(self, A, x, save_history=True):
        self.A = A
        
        self.x = {'min': x / xp.sqrt(x @ x),
                  'max': x / xp.sqrt(x @ x)}
        
        self.Ax = {key: self.A @ val for key, val in self.x.items()}
        
        self.grad = {key: xp.zeros_like(val) for key, val in self.x.items()}
        
        self.rho = {key: val @ self.A @ val for key, val in self.x.items()}
        
        self.save_history = save_history
        self.history = None
        
    def initialize_history(self, num_iters):
        if self.save_history:
            self.history = xp.zeros(num_iters)
            
    def update_history(self, i):
        if self.save_history:
            self.history[i] = self.rho['min']
        
    def run(self, num_iters=1000):
        self.initialize_history(num_iters)
        for i in range(num_iters):
            L = self.rho['max']
            lr = 1 / (2 * L)  # Optimum lr is 1 / lambda_max
            
            for key, val in self.grad.items():
                x = self.x[key]
                grad = self.grad[key]
                
                grad[:] = self.Ax[key] - self.rho[key] * x
                
                if key == 'min':
                    x[:] -= lr * grad
                else:
                    x[:] += lr * grad
                    
                x /= xp.sqrt(x @ x)
                
                self.Ax[key][:] = self.A @ x
                
                self.rho[key] = xp.dot(x, self.Ax[key])
                
            self.update_history(i)
            
        return False, i + 1, self.rho['min'], self.x['min']
