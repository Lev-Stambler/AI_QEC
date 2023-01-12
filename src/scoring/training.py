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
import scoring.model as scoring_model
##################################################################
##################################################################
# logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def train(model: scoring_model.ScoringTransformer, device, train_loader, optimizer, epoch, LR, shuffle_code_round):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    for batch_idx, (H, error_distr, error_rate) in enumerate(
            train_loader):
        if batch_idx % shuffle_code_round == 0:
            train_loader.dataset.shuffle_rand_code()
        error_rate_pred = model(H.to(device), error_distr.to(device))
        loss = model.loss(error_rate_pred, error_rate.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###

        cum_loss += loss.item() * loss.shape[0]

        cum_samples += loss.shape[0]
        if (batch_idx+1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model: scoring_model.ScoringTransformer, device, test_loader_list, EbNo_range_test, min_FER=10):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (parity_check, pc_adj, jm, x, z, y, magnitude,
                 syndrome) = next(iter(test_loader))
                z_mul = (y * utils.bin_to_sign(x))
                z_pred = model(parity_check.to(device), pc_adj.to(device),
                               magnitude.to(device), syndrome.to(device))
                loss = model.loss(-z_pred,
                                          z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                cum_count += loss.shape[0]
                # cum count before 1e5
                if (min_FER > 0 and test_fer > min_FER and cum_count > 10) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        print(
                            f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                    else:
                        print(
                            f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                    break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(
                f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
    logging.info(
        f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list

##################################################################
##################################################################
##################################################################


model = None


def main(n, k, deg_row, save_path,):
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #################################
    h = 4  # changed from 8...
    lr = 1e-4
    epochs = 1000
    batch_size = 1
    # Use a new random code after 32 runs, we do not want this to be too high as we are
    # trying to learn a **general** decoder
    rand_code_shuffle_len = 8
    workers = 1
    N_dec = 3  # CHanged from 6
    d_model = 40  # default is 32 but we are adding parity check info...
    adj_size = (n - k) * deg_row
    model = scoring_model.ScoringTransformer(n, k, h, d_model, N_dec,
                            adj_size,  dropout=0).to(device)
    model = torch.load(os.path.join(save_path, 'best_model'))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # We want a train size of about 400 * batch_size
    train_size = batch_size * \
        math.floor(400 / rand_code_shuffle_len) * rand_code_shuffle_len

    logging.info(model)
    logging.info(
        f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    test_batch_size = 1

		# TODO: scoring data loader...
    train_dataloader = DataLoader(ScoringDataset(), batch_size=int(batch_size),
                                  shuffle=True, num_workers=workers)
    test_dataloader_list = [DataLoader(ScoringDataset(),
                                       batch_size=int(test_batch_size), shuffle=False, num_workers=workers) for ii in range(len(std_test))]
    #################################
    # TODO: increase the batch size so loss is a better metric for saving
    best_loss = float('inf')
    last_save_epoch = -1
    for epoch in range(1, epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], shuffle_code_round=rand_code_shuffle_len)
        scheduler.step()
        if loss < best_loss and epoch - last_save_epoch > 15:
            best_loss = loss
            last_save_epoch = epoch
            torch.save(model, os.path.join(save_path, 'best_model'))
            print("Saving Model at Eopch", epoch)
        if epoch % 300 == 0 or epoch in [1, epochs]:
            test_loss_list, test_loss_ber_list, test_loss_fer_list = test(
                model, device, test_dataloader_list, EbNo_range_test)
            print("Losses for test of", test_loss_list,
                  test_loss_ber_list, test_loss_fer_list)
        # clear_output(wait=True)
        print(f"Epoch {epoch} finished, loss: {loss}, ber: {ber}, fer: {fer}")


GLOBAL_N = 50
GLOBAL_K = 15
GLOBAL_DEG_ROW = 5
main(GLOBAL_N, GLOBAL_K, GLOBAL_DEG_ROW, 'model_out')
