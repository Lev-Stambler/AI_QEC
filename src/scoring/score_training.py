import math
from global_params import params
from torch.utils.data import DataLoader
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import logging
import torch
import time
from scoring.score_dataset import ScoringDataset
import utils
import scoring.score_model as scoring_model
##################################################################
##################################################################
# logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def calc_std_dev(items: list):
    avg = sum(items) / len(items)
    return math.sqrt(sum([(avg - i) ** 2 for i in items]) / (len(items) - 1))


def train(model: scoring_model.ScoringTransformer, device, train_loader, optimizer, epoch, LR, plot_loss=None, plotting_period=500):
    model.train()
    cum_loss = cum_samples = 0
    best_p = 0
    t = time.time()
    train_std_dev = 0
    past_preds = []
    for batch_idx, (bit_adj, phase_adj, check_adj, error_distr, error_rate) in enumerate(
            train_loader):
        if cum_samples % plotting_period == 0:
            if cum_samples != 0:
                train_std_dev = calc_std_dev(past_preds)
            past_preds = []

        error_rate_pred = model(bit_adj.to(device).type(torch.float32), phase_adj.to(device).type(
            torch.float32), check_adj.to(device).type(torch.float32), error_distr.to(device).type(torch.float32))

        if cum_samples % plotting_period == 0:
            if plot_loss is None:
                print(
                    f"Training and on round {cum_samples}. Train delta err: {error_rate_pred.mean().item()  - error_rate.mean().item()}. Real success rate: {error_rate.mean().item()}")

        for p in error_rate.tolist():
            if p > best_p:
                best_p = p

        past_preds.append(error_rate_pred.mean().item())
        loss = model.loss(error_rate_pred, error_rate.unsqueeze(
            0).to(device).type(utils.get_numb_type()))
        if plot_loss is not None:
            plot_loss.update({'Train Delta Err': abs(error_rate_pred.mean().item() - error_rate.mean().item()), 'Train Loss': loss.item(),
                              'Train std dev': train_std_dev, 'Real Success Rate': error_rate.mean().item(), 'Predicated': error_rate_pred.mean().item()})
            plot_loss.send()  # draw, update logs, etc

        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###

        cum_loss += loss.item() * bit_adj.shape[0]

        cum_samples += bit_adj.shape[0]
        if (batch_idx+1) % 1_000 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, best_p


##################################################################
def get_save_dir(prefix, epoch):
    return f"data/{params['params_prefix']}_{prefix}_code_({params['n_data_qubits']},{params['n_data_qubits'] - params['n_check_qubits']})_epoch_{epoch}"


def main_training_loop(data_dir_prefix,
                       model, error_prob_sample, random_code_sample, save_path,
                       n_score_training_samples, epochs: int, plot_loss=None,
                       epoch_start=1):
    """
    Train the scoring model.
    Here we assume that the random_code_sample function always returns the same parity check
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    lr = 1e-6
    batch_size = 1
    # Use a new random code after 32 runs, we do not want this to be too high as we are
    # trying to learn a **general** decoder
    workers = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_size = n_score_training_samples
    logging.info(model)
    logging.info(
        f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')

    #################################
    best_loss = float('inf')
    for epoch in range(epoch_start, epochs + 1):
        train_dataloader = DataLoader(ScoringDataset(error_prob_sample, random_code_sample, load_save_dir=get_save_dir(data_dir_prefix, epoch), raw_dataset_size=train_size), batch_size=int(batch_size),
                                      shuffle=True, num_workers=workers)
        loss, best_p = train(model, device, train_dataloader, optimizer,
                     epoch, LR=scheduler.get_last_lr()[0], plot_loss=plot_loss)
        print("Stepping with scheduler")
        scheduler.step()
        print("Done stepping")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), save_path)
            print("Saving Model at Epoch", epoch)
        print(f"Epoch {epoch} finished, loss: {loss}, best_p: {best_p}")
