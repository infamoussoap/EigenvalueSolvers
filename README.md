# Fast solver to find the extreme eigenvalue
I present a simple code that allows one to compute the maximum and minimum eigenvalues of a positive definite matrix in Python.
The code is faster than `scipy.sparse.linalg.eigsh`, providing results that converge while `scipy.sparse.linalg.eigsh` doesn't.

Example code can be seen in the `Example.ipynb` file. The method that we used can be seen in `Eigenvalues.pdf`.

The following pictures are the results when trying to approximate the maximum and minimum eigenvalues of a positive definite $1000\times 1000$ matrix
with eigenvalues logarithmically from $10^{-4}$ to $10^4$. We remark that in this example, `scipy.sparse.linalg.eigsh` fails to converge
for the minimum eigenvalue.

![alt text](https://github.com/infamoussoap/EigenvalueSolvers/blob/main/convergence.png)
![alt text](https://github.com/infamoussoap/EigenvalueSolvers/blob/main/time_taken.png)