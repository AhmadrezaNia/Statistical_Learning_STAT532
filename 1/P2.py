import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from itertools import product

np.random.seed(0)


def check_condition(rho, r):
    return (6 * rho / (1 + rho**2)) > (1 / r + 2 * r)


def generate_X_Y(rho, r):
    X = np.array([[1, rho * r], [rho, r]])
    Y = np.dot(X, np.array([[-1/2], [1]]))
    return X, Y

# Solve the LASSO 
def lasso_regression(X, Y, lambda_val):
    lasso = Lasso(alpha=lambda_val, fit_intercept=False)
    lasso.fit(X, Y.ravel())
    return lasso.coef_

valid_pairs = []

# Randomly search for valid pairs of rho and r
while len(valid_pairs) < 16:
    rho = np.random.rand()
    r = np.random.rand()
    if check_condition(rho, r):
        valid_pairs.append((rho, r))    

fig, axs = plt.subplots(4, 4, figsize=(16, 16))  


lambda_values = np.logspace(-2, 2, 100)

# For each valid pair (rho, r), I compute the lasso path and plot it
for idx, (rho, r) in enumerate(valid_pairs):
    ax = axs[idx // 4, idx % 4]  # I Converted idx to 2D index for the subplot grid
    X, Y = generate_X_Y(rho, r)
    coefficients = []
    l1_norms = []
    
    for lambda_val in lambda_values:
        coef = lasso_regression(X, Y, lambda_val)
        coefficients.append(coef)
        l1_norms.append(np.linalg.norm(coef, 1))
    
    # Plot the coefficients 
    ax.plot(l1_norms, np.array(coefficients)[:, 0], label='Coefficient 1')
    ax.plot(l1_norms, np.array(coefficients)[:, 1], label='Coefficient 2')
    ax.set_title(r'$\rho={:.2f}, r={:.2f}$'.format(rho, r))
    ax.set_xlabel(r'$\|\hat{\beta}\|_1$')
    ax.legend()


for ax in axs.flat:
    ax.set(xlabel='||beta_hat||_1', ylabel='Coefficients')
    ax.label_outer()  

plt.tight_layout()
plt.show()
