import numpy as np


class Optimizer:
    """
    Implements Gradient Descent using numerical differentiation for calculating the gradient.
    """
    def __init__(self, step_size, max_iter, tol, delta):
        """
        Max_iter -- maximum number of iterations to run
        step_size -- also known as lambda
        tol -- tolerance
        delta -- perturbation to use in numerical differentiation
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        
    
    def optimize(self, cost_func, starting_params):
        """
        Finds parameters that optimize the given cost function.
        
        This method should implement your iterative algorithm for updating your parameter estimates.
        Use an updated estimate of the gradient to update the parametes.
        
        Give consideration for what the exit conditions of this loop should be.
        
        Returns a tuple of (optimized_param, iters)
        """ 
        current_params = starting_params
        iters = 0
        while iters < self.max_iter:
            # compute the gradient, ∇𝑓(𝑥⃑), at current model parameters.
            gradient = self._gradient(cost_func, current_params)
            
            # update 𝑥⃑𝑘+1 = 𝑥⃑𝑘 - 𝛾∇𝑓(𝑥⃑)
            optimized_params = self._update(current_params, gradient)

            # if change in gradient is < tolerance, stop – else go to step 3.
            if self._calculate_change(current_params, optimized_params) < self.tol:
                break

            iters += 1

        return optimized_params, iters
        
    
    def _calculate_change(self, old, new):
        """
        Calculates the change between the old and new parameters.
        Returns a scalar.
        """
        return np.square(new - old).sum()
        
    
    def _gradient(self, cost_func, params):
        """
        Numerically estimates the gradient (first derivative) of the cost function
        at param.
        
        First-order numerical differentiation
        df/dx = [ f(x + delta) - f(x) ] / delta
        
        Should return the gradient at the calculated point
        """
        gradient = np.zeros(params.size)
        for i in range(gradient.size):
            partial = np.copy(params)
            partial[i] += self.delta
            gradient[i] = (cost_func.cost(partial) - cost_func.cost(params)) / self.delta
            
        return gradient
        
            
    def _update(self, param, gradient):
        """
        Updates the param vector using the Gradient Descent algorithm.                
        
        Returns the new parameters.  (Do not modify input)
        """
        # 𝑥⃑𝑘+1 = 𝑥⃑𝑘 + 𝛾∇𝑓(𝑥⃑)
        return param - self.step_size * gradient
        