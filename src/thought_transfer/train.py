import argparse
from io import BytesIO
from datetime import datetime

import numpy as np
import webdataset as wds

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


def main(args):
    dataset = wds.WebDataset(args.data)
    dataset_iter = iter(dataset)

    text, resid1, resid2 = unpack(next(dataset_iter))
    data_in, data_out = get_input_output(resid1, resid2, args.mode)
    dim_in = data_in.shape[-1]
    dim_out = data_out.shape[-1]
    dim_hidden = max(dim_in, dim_out) * args.expansion

    model = SparseRecoder(d_in=dim_in, d_out=dim_out, d_hidden=dim_hidden)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse Recoder')
    parser.add_argument('data', type=str, help='dataset path')
    parser.add_argument('--expansion', type=float, default=8, help='expansion factor')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--mode', type=str, default='sae1', choices=['sae1', 'sae2', 'translate'])
    args = parser.parse_args()

    main(args)