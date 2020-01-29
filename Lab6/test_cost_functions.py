import unittest

import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error

import cost_functions
import optim

class TestLinearCostFunction(unittest.TestCase):
    def solve_iris_regression(self):
        """
        Fits a linear regression model for the Iris data set
        using the normal equation.
        """
        iris = datasets.load_iris()
        
        # use sepal length as features
        X = iris.data[:, :1]
        
        # add a columns of 1s so that model has an intercept term
        X = np.hstack([X, np.ones((iris.data.shape[0], 1))])
        
        # predict the petal length
        y = iris.data[:, 2]
        
        # solve for params with the normal equation
        betas = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        pred_y = np.dot(X, betas)
        
        return X, y, betas, pred_y

    def test_predict(self):
        """
        Tests the predict method using known model
        params
        """
        X, y, betas, exp_y = self.solve_iris_regression()
        
        # use cost function predict method
        cost = cost_functions.LinearCostFunction(X, y)
        obs_y = cost._predict(X, betas)
        mse = mean_squared_error(obs_y, exp_y)
        
        self.assertEqual(obs_y.shape, y.shape)
        self.assertAlmostEqual(mse, 0.0)
        
    def test_error(self):
        """
        Tests the mean-squared error implementation
        """
        X, y, _, pred_y = self.solve_iris_regression()
        
        # use cost function predict method
        cost = cost_functions.LinearCostFunction(X, y)
        obs_mse = cost._mse(y, pred_y)
        exp_mse = mean_squared_error(y, pred_y)
        
        self.assertAlmostEqual(obs_mse, exp_mse)
        
    def test_cost(self):
        """
        Tests the cost function.
        """
        X, y, betas, pred_y = self.solve_iris_regression()
        exp_mse = mean_squared_error(y, pred_y)
        
        # use cost function predict method
        cost_func = cost_functions.LinearCostFunction(X, y)
        cost = cost_func.cost(betas)
        
        self.assertAlmostEqual(cost, exp_mse)
        
    def test_optim(self):
        """
        Test optimizing the cost function.
        """
        X, y, betas, normal_pred_y = self.solve_iris_regression()
        exp_mse = mean_squared_error(y, normal_pred_y)
        
        cost_func = cost_functions.LinearCostFunction(X, y)
        optimizer = optim.Optimizer(0.01, 10000, 1e-4, 1e-4)
        
        initial_params = np.zeros(betas.shape)
        optim_params, n_iters = optimizer.optimize(cost_func, initial_params)
        print(n_iters)
        
        print(betas)
        print(optim_params)
        
        gd_pred_y = cost_func._predict(X, optim_params)
        obs_mse = mean_squared_error(y, gd_pred_y)
        self.assertAlmostEqual(exp_mse, obs_mse, 2)
        
class TestGaussianCostFunction(unittest.TestCase):
    def solve_model(self):
        """
        Fits the Gaussian model using estimates for
        the mean and variance.
        """
        data = np.loadtxt("gaussdist.csv", delimiter=",")
        
        y = data[:, 0]
        X = data[:, 1]

        # expected params
        # mu/mean, sigma/std
        mu = 3.0
        sigma = 1.0
        exp_params = np.array([mu, sigma])
        
        exp = ((X - mu) / sigma) ** 2
        pred_y = np.exp(-0.5 * exp) / (sigma * np.sqrt(2.0 * np.pi))
        
        return X, y, exp_params, pred_y

    def test_predict(self):
        """
        Tests the predict method using known model
        params
        """
        X, y, params, exp_y = self.solve_model()
        
        # use cost function predict method
        cost = cost_functions.GaussianCostFunction(X, y)
        obs_y = cost._predict(X, params)
        
        self.assertEqual(exp_y.shape, obs_y.shape)
        
    def test_error(self):
        """
        Tests the mean-squared error implementation
        """
        X, y, _, pred_y = self.solve_model()
        
        # use cost function predict method
        cost = cost_functions.GaussianCostFunction(X, y)
        obs_mse = cost._mse(y, pred_y)
        exp_mse = mean_squared_error(y, pred_y)
        
        self.assertAlmostEqual(obs_mse, exp_mse)
        
    def test_cost(self):
        """
        Tests the cost function.
        """
        X, y, params, pred_y = self.solve_model()
        exp_mse = mean_squared_error(y, pred_y)
        
        # use cost function predict method
        cost_func = cost_functions.GaussianCostFunction(X, y)
        cost = cost_func.cost(params)
        
        self.assertAlmostEqual(cost, exp_mse)
        
    def test_optim(self):
        """
        Test optimizing the cost function.
        """
        X, y, exp_params, exp_pred_y = self.solve_model()
        exp_mse = mean_squared_error(y, exp_pred_y)
        
        cost_func = cost_functions.GaussianCostFunction(X, y)
        optimizer = optim.Optimizer(0.1, 10000, 1e-4, 1e-4)
        
        # use a mean of 0, variance of 1 to start
        initial_params = np.array([0.0, 1.0])
        
        optim_params, n_iters = optimizer.optimize(cost_func, initial_params)
        
        print(n_iters)
        print(exp_params)
        print(optim_params)
        
        self.assertAlmostEqual(optim_params[0], exp_params[0], 1)
        self.assertAlmostEqual(optim_params[1], exp_params[1], 1)
  
if __name__ == "__main__":
    unittest.main()   