# Disable gradients for the network.
# Set your input tensor as a parameter requiring grad.
# Initialize an optimizer wrapping the input tensor.
# Backprop with some loss function and a goal tensor
# ...
# Profit!
import random
import torch
import utils
from CPC.generate_random import random_cpc
from CPC.cpc_code import get_classical_code_cpc


class GeneratingModel():
    def __init__(self, device, p_skip_mutation=1., p_random_mutation=0.0) -> None:
        self.device = device
        self.p_random_mutation = p_random_mutation
        self.p_skip_mutation = p_skip_mutation
    
    def generate_sample(self, scoring_model, physical_error_rates, num_steps=40, starting_code=None, mutate=True):
        bit_adj = phase_adj = check_adj = None
        if starting_code == None:
            _, bit_adj, phase_adj, check_adj = random_cpc()
        else:
            bit_adj = starting_code[0]
            phase_adj = starting_code[1]
            check_adj = starting_code[2]

        bit_adj = utils.numpy_to_parameter(bit_adj, self.device)
        phase_adj = utils.numpy_to_parameter(phase_adj, self.device)
        check_adj = utils.numpy_to_parameter(check_adj, self.device)
        physical_error_rates = utils.numpy_to_parameter(physical_error_rates, self.device)

        scoring_model.requires_grad_(False)
        physical_error_rates.requires_grad_(False)
        # Optimization originally from https://stackoverflow.com/questions/67328098/how-to-find-input-that-maximizes-output-of-a-neural-network-using-pytorch
        mse = torch.nn.MSELoss()
        optim = torch.optim.SGD([bit_adj, phase_adj, check_adj], lr=1e-1)
        # physical_error_rates.requires_grad_(False)
        goal_tensor = torch.tensor([[1.0]]).to(self.device).type(utils.get_numb_type())

        for _ in range(num_steps):
            # Optimize towards a 0 error rate
            loss = mse(scoring_model(bit_adj,
                phase_adj, check_adj, physical_error_rates), goal_tensor)
            loss.backward()
            # hmmm... we are doing some stepping here, but we have to make hard decisions eventually
            # Maybe this is where RL would be better...
            optim.step()
            optim.zero_grad()

        # Revert the model back
        scoring_model.requires_grad_(True)
        # print("OPTIMIZED", bit_adj, phase_adj, check_adj)
        hard_decision = lambda f_tensor: (f_tensor >= 0.5).squeeze(0).type(torch.int16).cpu().numpy()
        bit_adj, phase_adj, check_adj = hard_decision(bit_adj), hard_decision(phase_adj), hard_decision(check_adj)
        if mutate and random.random() > self.p_skip_mutation and self.p_random_mutation > 0.:
            self.mutate(
                bit_adj, phase_adj, check_adj
            )
        # print("AAA", bit_adj)
        pc = get_classical_code_cpc(bit_adj, phase_adj, check_adj)
        # print("PCC", pc, bit_adj)
        return pc, bit_adj, phase_adj, check_adj

    def mutate(self, bit_adj, phase_adj, check_adj):
        """
        Perform random "mutations" by randomly flipping bits in the parity check matrices
        according to `self.p_random_mutation`
        """
        p_edge_flip = self.p_random_mutation
        for data_bit in range(bit_adj.shape[0]):
            for check in range(bit_adj.shape[1]):
                if random.random() < p_edge_flip:
                   bit_adj[data_bit, check] = 1 - bit_adj[data_bit, check] 
                if random.random() < p_edge_flip:
                   phase_adj[data_bit, check] = 1 - phase_adj[data_bit, check] 
        for check_1 in range(check_adj.shape[0]):
            for check_2 in range(check_1):
                if random.random() < p_edge_flip:
                    check_adj[check_1, check_2] = 1- check_adj[check_1, check_2]
                    check_adj[check_2, check_1] = 1- check_adj[check_2, check_1]

