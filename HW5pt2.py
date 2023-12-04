# HW5 Problem 2

from bayes_opt import BayesianOptimization

# Define the objective function
def objective_function(x1, x2):
    return (4 - 2.1*x1**2 + (x1**4)/3) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2

# Bounded region of parameter space
pbounds = {'x1': (-3, 3), 'x2': (-2, 2)}

# Bayesian optimization
optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)

print(optimizer.max)
