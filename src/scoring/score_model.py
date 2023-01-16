
"""
@editing-author: Lev Stambler, levstamb@gmail.com

Original authors: Yoni Choukroun, choukroun.yoni@gmail.com
Error Correction Code Transformer
https://arxiv.org/abs/2203.14966
"""
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformer_util import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward
############################################################


class ScoringTransformer(nn.Module):
    def __init__(self, n_bits, n_checks, h, d_model, N_dec, device, dropout=0):
        super(ScoringTransformer, self).__init__()
        ####
        self.device = device
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model*4, dropout)

        n = (n_bits + n_checks) * 2
        k = n - n_checks
        self.n_checks = n_checks

        self.bit_adj_size = self.phase_adj_size = n_bits * n_checks

        self.check_check_adj = n_checks * n_checks

        inp_transformer_size = n * (n - k) + n
        self.src_embed = torch.nn.Parameter(torch.empty(
            (inp_transformer_size, d_model)))

        self.decoder = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(d_model, 1)])

        self.out_fc = nn.Linear(inp_transformer_size, 1)
        self.out_activation = nn.Sigmoid()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Only allow 1 batch size for now b/c I am feeling lazy rn
    def forward(self, bit_adj, phase_adj, check_adj, error_probabilities):
        # Addition of 0,1 bits mod 2 acts as an XOR
        phase_check = (((bit_adj.transpose(-2, -1) @ phase_adj) %
                       2) + check_adj) % 2
        pc = torch.concat([phase_adj.transpose(-2, -1), phase_check,
                          bit_adj.transpose(-1, -2), torch.eye(self.n_checks).unsqueeze(0).repeat(bit_adj.shape[0], 1, 1).to(self.device)], axis=-1)
        # Modified
        emb = torch.cat(
            [pc.flatten(start_dim=1), error_probabilities], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb)
        return self.out_activation(self.out_fc(self.oned_final_embed(emb).swapaxes(-1, -2)).squeeze(-1))

    def loss(self, error_rate_pred, real_error_rate):
        loss = F.mse_loss(error_rate_pred, real_error_rate)
        return loss
