from scipy.optimize import minimize
import numpy as np

# Define the objective function
def objective(x):
    return x[0]**2 + (x[1] - 3)**2

# Define the gradient of the objective function
def gradient_objective(x):
    return np.array([2*x[0], 2*(x[1] - 3)])

# Define the constraints functions
def constraint1(x):
    return x[1]**2 - 2*x[0]

def constraint2(x):
    return (x[1] - 1)**2 + 5*x[0] - 15

# Define the gradients of the constraints
def gradient_constraint1(x):
    return np.array([-2, 2*x[1]])

def gradient_constraint2(x):
    return np.array([5, 2*(x[1] - 1)])

# Define the Sequential Quadratic Programming (SQP) function
def mysqp(f, df, g1, dg1, g2, dg2, x0, opt):
    # Set initial conditions
    x = x0  # Set current solution to the initial guess
    W = np.eye(len(x))  # Initialize Hessian to identity
    mu = np.zeros(2)  # Initialize Lagrange multipliers to zero
    eps = opt['eps']
    
    # Optimization loop
    while True:
        # Solve the QP subproblem
        # This is a simplification using 'minimize' as a QP solver
        res = minimize(lambda p: 0.5 * np.dot(p, W.dot(p)) + np.dot(df(x), p),
                       np.zeros_like(x), 
                       method='SLSQP', 
                       constraints=[{'type': 'ineq', 'fun': lambda p: -g1(x + p)}, 
                                    {'type': 'ineq', 'fun': lambda p: -g2(x + p)}])
        s = res.x  # Step direction
        x_new = x + s  # New candidate
        
        # Check for convergence (simplistic approach)
        if np.linalg.norm(s) < eps:
            break
        
        # Update the Hessian approximation using BFGS
        y = df(x_new) - df(x)  # Change in gradients
        W = W - (W @ np.outer(s, s) @ W) / (s @ W @ s) + np.outer(y, y) / (y @ s)
        
        # Update x for the next iteration
        x = x_new

    return x

# Optimization settings
opt_settings = {'eps': 1e-6}

# Set the initial guess
x0 = np.array([1, 1])

# Run optimization
solution = mysqp(objective, gradient_objective, constraint1, gradient_constraint1, constraint2, gradient_constraint2, x0, opt_settings)

print("Optimized solution:", solution)
