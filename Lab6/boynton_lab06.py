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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# load data
features = np.loadtxt('Advertising.csv', delimiter=',', skiprows=1, usecols=[1, 2, 3])
response = np.loadtxt('Advertising.csv', delimiter=',', skiprows=1, usecols=[4])

# normalize features
features = features / features.max(axis=0)
    
# Add baseline feature in column 0
features = np.hstack((np.ones(features.shape[0]).reshape(-1, 1), features))
# %%
