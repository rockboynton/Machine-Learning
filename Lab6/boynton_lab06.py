# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Continued Cost Function Fun
# 
# Rock Boynton | CS 4850
# 
# ## Introduction
# 
# In this lab, we will get more practice writing cost functions, implement the
# linear regression machine learning algorithm and explore the effect of feature
# selection.
#
# The two cost functions we will be implementing are gaussian and multivariate
# linear regression
#
# 
# We will then run the following experiments on the linear regression model to make
# predictions on an advertising dataset
# 
# 1. Numerical Approximation
#
# 2. Normal Solutions
# 
# ## Summary of Results
# 
#
# ---

# %% [markdown]
# ## Setup
# 1. Load in the Advertisement.csv dataset and identify the column pertaining to the response
# variable and the feature variables.
#
# 2. Rescale each of the features such that the values range between 0 and 1.
# 
# 3. Add a feature to the dataset to include information about the baseline/DC term. This feature
# should be a constant among all observations.
#
# 4. At the end of this step you should have one variable (numpy matrix) that holds the feature
# matrix and one variable (numpy vector) that holds the response.

# %%
from optim import Optimizer
from cost_functions import GaussianCostFunction, LinearCostFunction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# load data
X = np.loadtxt('Advertising.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
y = np.loadtxt('Advertising.csv', delimiter=',', skiprows=1, usecols=[4])
features = X.copy()
# normalize features
X = X / X.max(axis=0)

# normalize response
# y = y / y.max(axis=0)
    
# Add baseline feature in column 0
X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
# %% [markdown]
# ## Experiment 1: Numerical Approximation

# %% 
cost = LinearCostFunction(X, y)

step_size = 0.1
max_iter = 5000
tol = 1e-8
delta = 1e-5

optimizer = Optimizer(step_size, max_iter, tol, delta)
initial_params = np.zeros(X.shape[1])
optimized_params, iters = optimizer.optimize(cost, initial_params)
print(f'Found min at {optimized_params} starting at {initial_params} in {iters} iterations of optimization algorithm.')
y_pred = np.sum(X * optimized_params, axis=1)
# y_true = np.sum(X, axis=1)
plt.scatter(y_pred, y)
plt.plot( [0,25],[0,25] )
plt.show()

# %% [markdown]
# ## Experiment 2: Normal Solutions

# %% 
# 1. Solve for the coefficients of a univariate linear model for each of the
#    features individually.
for feature in range(1, X.shape[1]):
    f = X[:, (0, feature)]
    beta = np.dot(np.dot(np.linalg.inv(np.dot(f.T, f)), f.T), y)
    print(beta)
    y_pred = np.dot(f, beta)
    x = np.linspace(0, 1, num=200)
    plt.scatter(X[:, feature], y)
    plt.plot(x, beta[0] + beta[1] * x)
    plt.title(f'Feature {feature}')
    plt.show()
plt.scatter(pred_y, y)
plt.plot( [0,25],[0,25], label='identity line' )
plt.title('True v Predicted Response')
plt.show()

# 2. Solve for the coefficients of a multivariate linear model using all of the features together.
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print(theta)
pred_y = np.dot(X, theta)
x = np.linspace(0, 1, num=200)
for feature in range(1, X.shape[1]):
    plt.scatter(X[:, feature], y)
    plt.plot(x, theta[0] + theta[feature] * x)
    plt.title(f'Feature {feature}')
    plt.show()
plt.scatter(pred_y, y)
plt.plot( [0,25],[0,25], label='identity line' )
plt.title('True v Predicted Response')
plt.show()

# %% [markdown]
# ## Questions
# 
# 1. For experiment 1, how long did it take your optimizer to converge to a solution? Does this
# seem reasonable?
#    It took 2469 iterations of optimization algorithm. which seems like a a lot
#    of iterations. But when you factor in that there are several parameters, I
#    could see how it might take longer
# 2. For experiment 1, what do you think would happen if you didn’t rescale your values?
# (maybe you didn’t at first!?)
# 3. For experiment 1, why did we have you plot the line of identiy? How do you interpret this,
# and what does it mean if you data is above this line? Below the line?
# 4. For experiment 2, what did you observe when you solved for the coefficients individually
# versus all together? If you noticed any differences, what do you think caused this? Can
# you explain this with what you know about optimization?
# 5. Can you interpret the coefficient values from this experiment? What does their
# magnitude/sign tell you?
