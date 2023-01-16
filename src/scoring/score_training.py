import math
from global_params import params
from torch.utils.data import DataLoader
from IPython.display import display, clear_output
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


def train(model: scoring_model.ScoringTransformer, device, train_loader, optimizer, epoch, LR, plot_loss=None, plotting_period=10):
    model.train()
    cum_loss = cum_samples = 0
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
        past_preds.append(error_rate_pred.mean().item())
        loss = model.loss(error_rate_pred, error_rate.unsqueeze(
            0).to(device).type(utils.get_numb_type()))
        if plot_loss is not None:
            plot_loss.update({'Train Delta Err': abs(error_rate_pred.mean().item() - error_rate.mean().item()), 'Train Loss': loss.item(),
                              'Train std dev': train_std_dev, 'Real Error Rate': error_rate.mean().item()})
            plot_loss.send()  # draw, update logs, etc

        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###

        cum_loss += loss.item() * bit_adj.shape[0]

        cum_samples += bit_adj.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples


##################################################################

def test(model: scoring_model.ScoringTransformer, device, test_loader):
    model.eval()
    test_loss_list, cum_samples_all = [], []
    t = time.time()
    # TODO: mov
    with torch.no_grad():
        for batch_idx, (bit_adj, phase_adj, check_adj, error_distr, error_rate) in enumerate(
                test_loader):
            test_loss = cum_count = 0.
            if cum_count % 100:
                print(
                    f"Doing sample {cum_count} out of {params['n_score_training_samples']}")
            error_rate_pred = model(bit_adj.to(device).type(torch.float32), phase_adj.to(
                device).type(torch.float32), check_adj.to(device).type(torch.float32), error_distr.to(device).type(torch.float32))
            loss = model.loss(error_rate_pred, error_rate.to(
                device).type(utils.get_numb_type()))

            test_loss += loss.item() * loss

            cum_count += bit_adj.shape[0]
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['Loss: {:.2e}'.format(elem) for elem
             in
             test_loss_list]))
    logging.info(
        f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list

##################################################################
##################################################################
##################################################################


def main_training_loop(model, error_prob_sample, random_code_sample, save_path, plot_loss=None, skip_testing=False):
    """
    Train the scoring model.
    Here we assume that the random_code_sample function always returns the same parity check
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    lr = 1e-4
    epochs = params['n_score_epochs']
    batch_size = 1
    # Use a new random code after 32 runs, we do not want this to be too high as we are
    # trying to learn a **general** decoder
    workers = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_size = params['n_score_training_samples']
    logging.info(model)
    logging.info(
        f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    test_batch_size = 1
    test_size = params['n_score_testing_samples']

    # TODO: scoring data loader...
    train_dataloader = DataLoader(ScoringDataset(error_prob_sample, random_code_sample, dataset_size=train_size), batch_size=int(batch_size),
                                  shuffle=True, num_workers=workers)
    test_dataloader = DataLoader(ScoringDataset(error_prob_sample, random_code_sample, dataset_size=test_size),
                                 batch_size=int(test_batch_size), shuffle=False, num_workers=workers)

    #################################
    # TODO: increase the batch size so loss is a better metric for saving
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        loss = train(model, device, train_dataloader, optimizer,
                     epoch, LR=scheduler.get_last_lr()[0], plot_loss=plot_loss)
        print("Stepping with scheduler")
        scheduler.step()
        print("Done stepping")
        # TODO: reenable
        if loss < best_loss:
            best_loss = loss
            torch.save(model, save_path)
            print("Saving Model at Epoch", epoch)
        if not skip_testing:
            test_loss_list = test(
                model, device, test_dataloader)
            print("Losses for test of", test_loss_list)
        # clear_output(wait=True)
        print(f"Epoch {epoch} finished, loss: {loss}")
