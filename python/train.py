import numpy as np
import torch
from nerf_model import NeRF
from nerf_loss import NeRFLoss
from constants import RESULT_DIR
import os
import time

if __name__ == "__main__":
    _dataset = np.load(f'{RESULT_DIR}/dataset.npz')
    dataset = {'o': _dataset['o'], 'd': _dataset['d'], 'C': _dataset['C']}

    N_EPOCH = 1
    BATCH_SIZE = 2048
    MAX_STEP = 1000
    PRINT_INTERVAL = 100

    nerf = NeRF(t_n=0., t_f=2.5, c_bg=(1, 1, 1))
    loss_func = NeRFLoss(nerf)

    optimizer = torch.optim.Adam(
        loss_func.parameters(),
        lr=3e-4, betas=(0.9, 0.999), eps=1e-7)

    loss_func.cuda('cuda:0')

    n_sample = dataset['o'].shape[0]

    save_dir = f"{RESULT_DIR}/train/"
    os.makedirs(save_dir, exist_ok=True)

    f = open(f"{save_dir}/train_loss.tsv", 'w')
    header_str = "time\tepoch\tstep\tepoch_rate\tloss\n"
    f.write(header_str)
    print(header_str, end="")

    step = 0
    start_time = time.time()

    for e in range(1, N_EPOCH + 1):
        perm = np.random.permutation(n_sample)
        sum_loss = 0.
        sum_loss_print = 0.

        for i in range(0, n_sample, BATCH_SIZE):
            step += 1
            o = dataset['o'][perm[i:i + BATCH_SIZE]]
            d = dataset['d'][perm[i:i + BATCH_SIZE]]
            C = dataset['C'][perm[i:i + BATCH_SIZE]]

            loss = loss_func(o, d, C)
            sum_loss += loss.item() * o.shape[0]
            sum_loss_print += loss.item()

            if (i / BATCH_SIZE + 1) % PRINT_INTERVAL == 0:
                sum_loss_print /= PRINT_INTERVAL
                elapsed_sec = int(time.time() - start_time)
                elapsed_min = elapsed_sec // 60 % 60
                elapsed_hou = elapsed_sec // 3600
                elapsed_str = f"{elapsed_hou:02d}:{elapsed_min:02d}:{elapsed_sec % 60:02d}"
                epoch_rate = 100 * (i + BATCH_SIZE) / n_sample
                print_str = f"{elapsed_str}\t{e}\t{step}\t{epoch_rate:5.1f}\t{sum_loss_print:.5f}\n"
                f.write(print_str)
                f.flush()
                print(print_str, end="")
                sum_loss_print = 0

            loss_func.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(nerf.state_dict(), f"{save_dir}/nerf_model.pt")
            if step >= MAX_STEP:
                break

        # save state.
        torch.save(nerf.state_dict(), f"{save_dir}/nerf_model.pt")
        torch.save(loss_func.state_dict(), f"{save_dir}/loss_func.pt")
        torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")

        if step >= MAX_STEP:
            break
