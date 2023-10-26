import torch

def solve(s, d, lr=0.01):
    """
    Update state using the reduced gradient.
    s: current state
    d: reduced gradient
    """
    with torch.no_grad():
        s -= lr * d
    return s

def GRG_optimizer(lr=0.01, max_iter=100):
    x1 = torch.tensor(0.5, requires_grad=True, dtype=torch.float32)
    x2 = torch.tensor(0.5, requires_grad=True, dtype=torch.float32)
    x3 = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    for k in range(max_iter):
        f = x1**2 + x2**2 + x3**2
        h1 = x1**2/4 + x2**2/5 + x3**2/25 - 1
        h2 = x1 + x2 - x3

        h1.retain_grad()
        h2.retain_grad()

        # Zero the gradients
        if x1.grad: x1.grad.zero_()
        if x2.grad: x2.grad.zero_()
        if x3.grad: x3.grad.zero_()

        f.backward(retain_graph=True)

        grad_x1 = x1.grad
        grad_h1 = h1.grad
        grad_h2 = h2.grad
        
        if grad_x1 is None or grad_h1 is None or grad_h2 is None:
            continue

        d1 = grad_x1 - grad_h1 * grad_x1 / grad_h1 - grad_h2 * grad_x1 / grad_h2
        d2 = x2.grad - grad_h1 * x2.grad / grad_h1 - grad_h2 * x2.grad / grad_h2
        d3 = x3.grad - grad_h1 * x3.grad / grad_h1 - grad_h2 * x3.grad / grad_h2

        # Update the state variables using the solve function
        x1 = solve(x1, d1, lr)
        x2 = solve(x2, d2, lr)
        x3 = solve(x3, d3, lr)

    return x1.item(), x2.item(), x3.item()

solution = GRG_optimizer()
print(f"Optimal solution: x1={solution[0]}, x2={solution[1]}, x3={solution[2]}")
