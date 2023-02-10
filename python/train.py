import numpy as np
import torch
from nerf_model import NeRF
from nerf_loss import NeRFLoss
from constants import RESULT_DIR
import os

if __name__ == "__main__":
    _dataset = np.load(f'{RESULT_DIR}/dataset.npz')
    dataset = {'o': _dataset['o'], 'd': _dataset['d'], 'C': _dataset['C']}

    n_epoch = 1
    batch_size = 2048
    max_step = 100

    nerf = NeRF(t_n=0., t_f=2.5, c_bg=(1, 1, 1))
    loss_func = NeRFLoss(nerf)

    optimizer = torch.optim.Adam(
        loss_func.parameters(),
        lr=3e-4, betas=(0.9, 0.999), eps=1e-7)

    loss_func.cuda('cuda:0')

    n_sample = dataset['o'].shape[0]

    save_dir = f"{RESULT_DIR}/train/"
    os.makedirs(save_dir, exist_ok=True)

    step = 0

    for e in range(1, n_epoch + 1):
        print(f'epoch: {e}')
        perm = np.random.permutation(n_sample)
        sum_loss = 0.
        sum_loss_print = 0.

        PRINT_INTERVAL = 100

        for i in range(0, n_sample, batch_size):
            o = dataset['o'][perm[i:i + batch_size]]
            d = dataset['d'][perm[i:i + batch_size]]
            C = dataset['C'][perm[i:i + batch_size]]

            loss = loss_func(o, d, C)
            sum_loss += loss.item() * o.shape[0]
            sum_loss_print += loss.item()

            if (i / batch_size + 1) % PRINT_INTERVAL == 0:
                sum_loss_print /= PRINT_INTERVAL
                print(f"{i} / {n_sample} ({100 * i / n_sample:5.1f}) {sum_loss_print:.4f}")
                sum_loss_print = 0

            loss_func.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(nerf.state_dict(), f"{save_dir}/nerf_model.pt")
            step += 1
            if step >= max_step:
                break

        print('sum loss: {}'.format(sum_loss / n_sample))

        # save state.
        torch.save(nerf.state_dict(), f"{save_dir}/nerf_model.pt")
        torch.save(loss_func.state_dict(), f"{save_dir}/loss_func.pt")
        torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")

        if step >= max_step:
            break
