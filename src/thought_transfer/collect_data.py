import sys
import argparse
from pathlib import Path
from datetime import datetime
from io import BytesIO
import json

import torch as t
import numpy as np
import transformer_lens as tl
import webdataset as wds
import datasets
import nltk
from tqdm import tqdm


cached_value = None


def cache_it(resid, hook):
    global cached_value
    # cached_value = t.clone(resid.detach())
    cached_value = resid.detach()
    # print(hook)
    return resid


def get_middle_hook(model):
    nlayers = model.W_K.shape[0]
    mid_layer = nlayers // 2
    return f'blocks.{mid_layer}.hook_resid_post'


def main(args):
    model1 = tl.HookedTransformer.from_pretrained(args.model1)
    model2 = tl.HookedTransformer.from_pretrained(args.model2)

    hook1 = get_middle_hook(model1)
    hook2 = get_middle_hook(model2)

    nltk.download('punkt')
    nltk.download('punkt_tab')

    data_dir = Path(args.save)
    wiki = datasets.load_dataset("wikipedia", "20220301.en",
                                 cache_dir=(data_dir.parent / "cache").as_posix())

    inds = np.load("../data/wiki_split.npz")
    train_inds = inds["train_inds"]

    print(f"num train inds: {len(train_inds)}")

    test_inds = inds["test_inds"]

    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    data_fname = f"../data/wiki_sentences_{args.start_ind}_{args.end_ind}_{timestamp}.tar"
    print(f"writing to {data_fname}", flush=True)
    writer = wds.TarWriter(data_fname)
    log_fname = f"../outputs/log_{timestamp}.txt"
    with open(log_fname, "w") as log_file:
        print(f"logging to {log_fname}", flush=True)
        num_sampled = 0
        num_mismatched = 0
        for ii, i in tqdm(enumerate(train_inds[args.start_ind:args.end_ind]), total=args.end_ind - args.start_ind):
            text = wiki["train"][int(i)]["text"]
            sent = nltk.sent_tokenize(text)

            print(f"{ii} before check len", file=log_file, flush=True)
            if len(sent) <= args.num_sentences:
                continue
            print(f"{ii} after check len", file=log_file, flush=True)
            random_chunk = np.random.randint(0, len(sent) - args.num_sentences)
            chunk = sent[random_chunk:random_chunk + args.num_sentences]
            chunk = " ".join(chunk)
            tokens1 = model1.to_str_tokens(chunk)
            tokens2 = model2.to_str_tokens(chunk)
            print(f"{ii} after composing the sent", file=log_file, flush=True)

            if len(tokens1) > args.max_tokens or len(tokens2) > args.max_tokens:
                continue

            print(f"{ii} token check", file=log_file, flush=True)

            if tokens1[1:] != tokens2[1:]:
                num_mismatched += 1

            num_sampled += 1

            model1.run_with_hooks(chunk, return_type="loss", fwd_hooks=[(hook1, cache_it)])
            resid1 = cached_value.detach().cpu().numpy()
            print(f"{ii} gpt2", file=log_file, flush=True)
            model2.run_with_hooks(chunk, return_type="loss", fwd_hooks=[(hook2, cache_it)])
            resid2 = cached_value.detach().cpu().numpy()
            print(f"{ii} opt", file=log_file, flush=True)

            print(f"res sum: {np.sum(resid1)}, {np.sum(resid2)}")


            # print(resid_gpt2.shape, resid_opt.shape)

            writer.write({
                "__key__": f"{num_sampled:08d}",
                "text.txt": chunk,
                "resid_gpt2.npy": resid1,
                "resid_opt.npy": resid2
            })

            print(f"{ii} written", file=log_file, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="../data/wikipedia")
    parser.add_argument("--model1", type=str, default="opt-125m")
    parser.add_argument("--model2", type=str, default="gpt2-small")
    parser.add_argument("--num_sentences", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--start_ind", type=int, default=10_000)
    parser.add_argument("--end_ind", type=int, default=30_000)
    args = parser.parse_args()

    main(args)
