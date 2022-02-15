import numpy as np

def cost_function( x, y, theta0, theta1 ):
    """Compute the squared error cost function

    Inputs:
    x        vector of length m containing x values
    y        vector of length m containing y values
    theta_0  (scalar) intercept parameter
    theta_1  (scalar) slope parameter

    Returns:
    cost     (scalar) the cost
    """
    
    h_x = theta1 * x + theta0
    cost = (1/2) * sum([val**2 for val in (y-h_x)])
    
    return cost


def gradient(x, y, theta0, theta1, step_size):
    """Compute the partial derivative of the squared error cost function

    Inputs:
    x          vector of length m containing x values
    y          vector of length m containing y values
    theta_0    (scalar) intercept parameter
    theta_1    (scalar) slope parameter

    Returns:
    d_theta_0  (scalar) Partial derivative of cost function wrt theta_0
    d_theta_1  (scalar) Partial derivative of cost function wrt theta_1
    """

    
    # variable for h_theta(x)
    h_x = (theta1 * x) + theta0

    # Evaluate the derivatives
    d_theta1 = 2 * sum(x * (h_x-y))
    d_theta0 = 2 * sum(h_x-y)

    # update theta0, theta1
    theta1 = theta1 - (step_size * d_theta1)
    theta0 = theta0 - (step_size * d_theta0)
    
    
    return theta0, theta1 # return is a tuple