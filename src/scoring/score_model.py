
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
    def __init__(self, n, k, h, d_model, N_dec, dropout=0):
        super(ScoringTransformer, self).__init__()
        ####
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_model*4, dropout)

        self.pc_size = n * (n - k)

        self.src_embed = torch.nn.Parameter(torch.empty(
            (self.pc_size + n, d_model)))

        self.decoder = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(d_model, 1)])

        self.out_fc = nn.Linear(self.pc_size + n, 1)
        self.out_activation = nn.Sigmoid()

        self.n = n
        self.k = k
        # TODO: renable mask
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Only allow 1 batch size for now b/c I am feeling lazy rn
    def forward(self, p_check_mat, error_probabilities):
        # Modified
        emb = torch.cat([p_check_mat.flatten(start_dim=-2), error_probabilities], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        emb = self.decoder(emb)
        return self.out_activation(self.out_fc(self.oned_final_embed(emb).swapaxes(-1, -2)).squeeze(-1))

    def loss(self, error_rate_pred, real_error_rate):
        loss = F.mse_loss(error_rate_pred, real_error_rate)
        # print(f"Error rate predicted {error_rate_pred}, real error rate {real_error_rate}, loss {loss}")
        return loss
