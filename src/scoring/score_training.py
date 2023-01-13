import math
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


def train(model: scoring_model.ScoringTransformer, device, train_loader, optimizer, epoch, LR, plot_loss=None):
    model.train()
    cum_loss = cum_samples = 0
    t = time.time()
    for batch_idx, (H, error_distr, error_rate) in enumerate(
            train_loader):
        error_rate_pred = model(H.to(device), error_distr.to(device))
        loss = model.loss(error_rate_pred, error_rate.unsqueeze(0).to(device))
        if plot_loss is not None:
            plot_loss.update({'Train Delta Err': math.abs(error_rate_pred.mean().item() - error_rate.mean().item()), 'Train Loss': loss.item()})
            plot_loss.send()  # draw, update logs, etc
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###

        cum_loss += loss.item() * H.shape[0]

        cum_samples += H.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples


##################################################################

def test(model: scoring_model.ScoringTransformer, device, test_loader_list):
    model.eval()
    test_loss_list, cum_samples_all = [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = cum_count = 0.
            while True:
                (H, error_dist, error_rate) = next(iter(test_loader))
                error_rate_pred = model(H.to(device), error_dist.to(device))
                loss = model.loss(error_rate_pred, error_rate)

                test_loss += loss.item() * loss

                cum_count += H.shape[0]
                # cum count before 1e5
                if cum_count >= 1e4:
                    break
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


def main_training_loop(model, error_prob_sample, random_code_sample, save_path, plot_loss=None):
    """
    Train the scoring model.
    Here we assume that the random_code_sample function always returns the same parity check
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    lr = 1e-4
    epochs = 1000
    batch_size = 1
    # Use a new random code after 32 runs, we do not want this to be too high as we are
    # trying to learn a **general** decoder
    workers = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # We want a train size of about 400 * batch_size
    train_size = batch_size * 400
    logging.info(model)
    logging.info(
        f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    test_batch_size = 1
    test_size = test_batch_size * 20

    # TODO: scoring data loader...
    train_dataloader = DataLoader(ScoringDataset(error_prob_sample, random_code_sample, dataset_size=train_size), batch_size=int(batch_size),
                                  shuffle=True, num_workers=workers)
    test_dataloader_list = [DataLoader(ScoringDataset(error_prob_sample, random_code_sample, dataset_size=test_size),
                                       batch_size=int(test_batch_size), shuffle=False, num_workers=workers)]
    #################################
    # TODO: increase the batch size so loss is a better metric for saving
    best_loss = float('inf')
    last_save_epoch = -1
    for epoch in range(1, epochs + 1):
        loss = train(model, device, train_dataloader, optimizer,
                     epoch, LR=scheduler.get_last_lr()[0], plot_loss=plot_loss)
        scheduler.step()
        if loss < best_loss and epoch - last_save_epoch > 15:
            best_loss = loss
            last_save_epoch = epoch
            torch.save(model, os.path.join(save_path, 'best_model'))
            print("Saving Model at Epoch", epoch)
        if epoch % 300 == 0 or epoch in [1, epochs]:
            test_loss_list = test(
                model, device, test_dataloader_list)
            print("Losses for test of", test_loss_list)
        # clear_output(wait=True)
        print(f"Epoch {epoch} finished, loss: {loss}")
