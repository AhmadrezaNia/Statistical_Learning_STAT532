from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(0)  
X = np.random.randn(100, 200)  # design matrix
beta = np.concatenate((np.ones(30), np.zeros(170)))  # coefficients

def compute_Y(X, beta, sigma):
    noise = np.random.normal(0, sigma, X.shape[0])
    return X @ beta + noise


sigma_values = np.logspace(-2, 2, 50)
optimal_lambdas = []


for sigma in sigma_values:
    Y = compute_Y(X, beta, sigma)
    lasso_cv = LassoCV(cv=5, fit_intercept=False)
    lasso_cv.fit(X, Y)
    optimal_lambdas.append(lasso_cv.alpha_)


plt.figure(figsize=(10, 6))
plt.plot(sigma_values, optimal_lambdas, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Sigma values')
plt.ylabel('Optimal Lambda')
plt.title('Trend of Optimal Lambda as Sigma Varies')
plt.grid(True)
plt.show()
