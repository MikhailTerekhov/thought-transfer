import sys
import argparse
from pathlib import Path
from io import BytesIO
import json

import torch as t
import numpy as np
import transformer_lens as tl
import webdataset as wds
import datasets
import nltk


cached_value = None


def cache_it(resid, hook):
    global cached_value
    cached_value = resid.detach()
    # print(hook)
    return resid


def main(args):
    model1 = tl.HookedTransformer.from_pretrained(args.model1)
    model2 = tl.HookedTransformer.from_pretrained(args.model2)

    nltk.download('punkt')
    nltk.download('punkt_tab')

    wiki = datasets.load_dataset("wikipedia", "20220301.en")

    train_frac = 0.8
    test_frac = 1.0 - train_frac
    train_mask = np.random.rand(len(wiki["train"])) < train_frac
    test_mask = ~train_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="../data/wikipedia")
    parser.add_argument("--model1", type=str, default="opt-125m")
    parser.add_argument("--model2", type=str, default="gpt2_small")
    args = parser.parse_args()
