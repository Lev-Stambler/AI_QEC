import torch


class ScoringInitDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, error_prob_sample: function, random_code_sample: function, n_samples):
        self.error_prob = error_prob_sample
        self.random_code = random_code_sample
        self.n_samples = n_samples
        super(ScoringInitDataset, self).__init__()

    def __getitem__(self, index):
        return (self.random_code(), self.error_prob())

    def __len__(self):
        return self.n_samples
