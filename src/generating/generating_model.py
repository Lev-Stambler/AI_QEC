# Disable gradients for the network.
# Set your input tensor as a parameter requiring grad.
# Initialize an optimizer wrapping the input tensor.
# Backprop with some loss function and a goal tensor
# ...
# Profit!
import torch
from CPC.generate_random import random_cpc


class GeneratingModel():
    def __init__(self, n_bits, n_checks, deg_bits, deg_phase, deg_cc) -> None:
        self.n_bits = n_bits
        self.n_checks = n_checks
        self.deg_phase = deg_phase
        self.deg_bits = deg_bits
        self.deg_cc = deg_cc

    def generate_origin_sample(self, scoring_model, physical_error_rates, num_steps=10):
        scoring_model.requires_grad_(False)
        # Optimization originally from https://stackoverflow.com/questions/67328098/how-to-find-input-that-maximizes-output-of-a-neural-network-using-pytorch
        mse = torch.nn.MSELoss()
        optim = torch.optim.SGD([bit_adj, phase_adj, check_adj], lr=1e-1)
        physical_error_rates.requires_grad_(False)
        _, bit_adj, phase_adj, check_adj = random_cpc(self.n_bits, self.n_checks,
            self.deg_phase, self.deg_bit, self.deg_cc)
        # TODO: starting?
        for _ in range(num_steps):
            # Optimize towards a 0 error rate
            # TODO: maybe doing 0 here makes things too aggressive... lets see
            loss = mse(scoring_model(bit_adj,
                phase_adj, check_adj, physical_error_rates), 0.0)
            loss.backward()
            # hmmm... we are doing some stepping here, but we have to make hard decisions eventually
            # Maybe this is where RL would be better...
            optim.step()
            optim.zero_grad()

        # Revert the model back
        scoring_model.requires_grad_(True)
        print("OPTIMIZED", bit_adj, phase_adj, check_adj)
        hard_decision = lambda f_tensor: (f_tensor >= 0.5).type(torch.int16)
        return hard_decision(bit_adj), hard_decision(phase_adj), hard_decision(check_adj)

    def mutate_origin_sample(self):
        pass

