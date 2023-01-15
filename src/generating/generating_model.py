# Disable gradients for the network.
# Set your input tensor as a parameter requiring grad.
# Initialize an optimizer wrapping the input tensor.
# Backprop with some loss function and a goal tensor
# ...
# Profit!
import torch


class GeneratingModel():
    def __init__(self, n, k) -> None:
        pass

    def go_up(self):
        pass

    def generate_samples(self):
        pass


f = torch.nn.Linear(10, 5)
f.requires_grad_(False)
x = torch.nn.Parameter(torch.rand(10), requires_grad=True)
optim = torch.optim.SGD([x], lr=1e-1)
mse = torch.nn.MSELoss()
y = torch.ones(5)  # the desired network response

num_steps = 5  # how many optim steps to take
for _ in range(num_steps):
    loss = mse(f(x), y)
    loss.backward()
    optim.step()
    optim.zero_grad()
