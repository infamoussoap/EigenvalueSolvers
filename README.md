Without loss of generality, we look at finding the maximum and minimum eigenvalues of a symmetric, semi-positive definite matrix $A$. This can be found by solving the constrained optimization problem

$$\min_x f(x)\ =\ \min_x x^TAx,\qquad\text{for}\qquad x^Tx\ =\ 1.$$

LOBPCG is a popular algorithm that uses gradient ascent (or descent) to find the maximum (or minimum) eigenvalues. LOBPCG uses a locally optimum approach in which each iteration performs a line search or a Rayleigh-Ritz method to guarantee each iteration increases (or decreases).

However, for a convex function $f$ with $L$-Lipshitz gradient, it is well known that the optimum learning rate of gradient descent and Nesterov accelerated gradient descent is $1/L$. Since $A$ is semi-positive definite, $f$ is convex with a $L=2\lambda_{\max}$-Lipshitz continuous gradient, where $\lambda_{\max}$ is the maximum eigenvalue of $A$. 

Once $\lambda_{\max}$ is found, the optimum learning rate for gradient descent is known, allowing us to find $\lambda_{\min}$. Moreover, $1/2\lambda_{\max}$ will guarantee that the function value decreases at each iteration, providing a locally optimum algorithm. Thus, we need only look at how to find $\lambda_{\max}$.

Once found, if $\lambda_{\min}>0$ (which can be assured by convergence properties of Nesterov descent), the problem becomes strongly convex. In turn, projected Nesterov descent can then be used to yield an exponentially converging algorithm for rest of the smallest eigenvalues.

