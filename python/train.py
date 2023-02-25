import numpy as np
import torch
import torch.nn.functional as F
from nerf_model import NeRF
from constants import RESULT_DIR
import os
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_step", type=int, default=1e9)
    args = parser.parse_args()

    dataset = np.load(f'{RESULT_DIR}/dataset.npz')
    o_np = dataset['o']
    d_np = dataset['d']
    C_np = dataset['C']

    N_EPOCH = 1
    BATCH_SIZE = 2048
    MAX_STEP = args.max_step
    PRINT_INTERVAL = 100

    nerf = NeRF(t_n=0., t_f=2.5)
    nerf.to("cuda")

    optimizer = torch.optim.Adam(
        nerf.parameters(),
        lr=2e-4, betas=(0.9, 0.999), eps=1e-7)

    n_sample = o_np.shape[0]

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
        sample_num = 0

        for i in range(0, n_sample, BATCH_SIZE):
            step += 1
            o = o_np[perm[i:i + BATCH_SIZE]]
            d = d_np[perm[i:i + BATCH_SIZE]]
            C = C_np[perm[i:i + BATCH_SIZE]]
            o = torch.tensor(o, device=nerf.device())
            d = torch.tensor(d, device=nerf.device())
            C = torch.tensor(C, device=nerf.device())

            C_c, C_f = nerf.forward(o, d)
            loss = F.mse_loss(C_c, C) + F.mse_loss(C_f, C)
            sum_loss += loss.item() * o.shape[0]
            sample_num += o.shape[0]

            if (i / BATCH_SIZE + 1) % PRINT_INTERVAL == 0:
                sum_loss /= sample_num
                elapsed_sec = int(time.time() - start_time)
                elapsed_min = elapsed_sec // 60 % 60
                elapsed_hou = elapsed_sec // 3600
                elapsed_str = f"{elapsed_hou:02d}:{elapsed_min:02d}:{elapsed_sec % 60:02d}"
                epoch_rate = 100 * (i + BATCH_SIZE) / n_sample
                print_str = f"{elapsed_str}\t{e}\t{step}\t{epoch_rate:5.1f}\t{sum_loss:.5f}\n"
                f.write(print_str)
                f.flush()
                print(print_str, end="")
                sum_loss = 0
                sample_num = 0
                # save state.
                torch.save(nerf.state_dict(), f"{save_dir}/nerf_model.pt")
                torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()
            if step >= MAX_STEP:
                break

        if step >= MAX_STEP:
            break
