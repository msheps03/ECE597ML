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

    ##################################################
    # TODO: write code here to compute cost correctly
    ##################################################
    
    y_pred = theta1 * x + theta0
    cost = (1/2) * sum([val**2 for val in (y-y_pred)])
    
    return cost


def gradient(x, y, theta0, theta1, step_size, previous_cost=0):
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

    ##################################################
    # TODO: write code here to compute partial derivatives correctly
    ##################################################
    
    done = False
    previous_cost = -1
    close = 1e-6
    
    n = len(x)
    # Making predictions, h(x)
    y_predicted = (theta1 * x) + theta0

    # Calculating the current cost
    current_cost = cost_function(x, y, theta0, theta1)

    # check if convergence, close is some value close to zero
    if previous_cost and abs(previous_cost-current_cost) <= close:
        done = True

    # Calculating the gradients
    d_theta1 = (2) * sum(x * -y+y_predicted)
    d_theta0 = (2) * sum(-y+y_predicted)

    # Updating weights and bias
    theta1 = theta1 - (step_size * d_theta1)
    theta0 = theta0 - (step_size * d_theta0)
    
    
    
    return theta1, theta0, done, current_cost # return is a tuple