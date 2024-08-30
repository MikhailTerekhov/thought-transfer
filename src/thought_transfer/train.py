import sys
import argparse
from io import BytesIO
from datetime import datetime
from itertools import chain
from pathlib import Path

import wandb
import numpy as np
import webdataset as wds
import torch as t
from tqdm import tqdm

sys.path.append(Path(__file__).resolve().parent.as_posix())
from sparse_recoder import SparseRecoder


def unpack(sample):
    text = sample["text.txt"]
    resid_gpt2 = np.load(BytesIO(sample["resid_gpt2.npy"]))
    resid_opt = np.load(BytesIO(sample["resid_opt.npy"]))
    return text, resid_gpt2, resid_opt


def get_input_output(resid1, resid2, mode):
    if mode == 'sae1':
        data_in = resid1
        data_out = resid1
    elif mode == 'sae2':
        data_in = resid2
        data_out = resid2
    elif mode == 'translate':
        data_in = resid1
        data_out = resid2
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return data_in, data_out


def fix_batch_iter(dataset_iter, batch_size, mode):
    batch_in = []
    batch_out = []
    cum_size = 0
    num_processed = 0
    for sample in dataset_iter:
        text, resid1, resid2 = unpack(sample)
        data_in, data_out = get_input_output(resid1, resid2, mode)
        # first dim is 1
        data_in = t.tensor(data_in[0], device='cuda')
        data_out = t.tensor(data_out[0], device='cuda')
        batch_in.append(data_in)
        batch_out.append(data_out)
        num_processed += 1
        cum_size += data_in.shape[0]

        if cum_size >= batch_size:
            concat_in = t.cat(batch_in, dim=0)
            concat_out = t.cat(batch_out, dim=0)
            yield_in = concat_in[:batch_size]
            yield_out = concat_out[:batch_size]

            rest_in = concat_in[batch_size:]
            rest_out = concat_out[batch_size:]
            batch_in = [rest_in]
            batch_out = [rest_out]
            cum_size = rest_in.shape[0]

            yield num_processed, (yield_in, yield_out)

    if cum_size > 0:
        concat_in = t.cat(batch_in, dim=0)
        concat_out = t.cat(batch_out, dim=0)
        yield_in = concat_in
        yield_out = concat_out
        yield num_processed, (yield_in, yield_out)


def main(args):
    wandb.init(project='sparse-recoder')

    dataset_iter = chain(*[iter(wds.WebDataset(d)) for d in args.data])

    text, resid1, resid2 = unpack(next(dataset_iter))
    data_in, data_out = get_input_output(resid1, resid2, args.mode)
    dim_in = data_in.shape[-1]
    dim_out = data_out.shape[-1]
    dim_hidden = max(dim_in, dim_out) * args.expansion

    print(f"dim_in: {dim_in}, dim_out: {dim_out}, dim_hidden: {dim_hidden}")
    print(text)

    model = SparseRecoder(d_in=dim_in, d_out=dim_out, d_hidden=dim_hidden)
    model = model.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)

    batch_iter = fix_batch_iter(dataset_iter, args.batch_size, args.mode)
    prev_i = 0
    pbar = tqdm(total=args.num_samples)
    for i, (batch_in, batch_out) in batch_iter:
        recons = model(batch_in)
        loss = t.nn.functional.mse_loss(recons, batch_out)
        wandb.log({'loss': loss.item()})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.update(i - prev_i)
        prev_i = i


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse Recoder')
    parser.add_argument('data', type=str, nargs='+', help='dataset path')
    parser.add_argument('--expansion', type=float, default=8, help='expansion factor')
    parser.add_argument('--num_samples', type=int, default=None, help='number of samples (only for time est)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--mode', type=str, default='sae1', choices=['sae1', 'sae2', 'translate'])
    args = parser.parse_args()

    main(args)