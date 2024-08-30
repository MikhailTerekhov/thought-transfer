import torch as t
from torch.nn.init import kaiming_normal_


# sanity check: without topk easily overfits on id data
class SparseRecoder(t.nn.Module):
    def __init__(self, d_in, d_out, d_hidden, k=5, add_topk=True):
        super(SparseRecoder, self).__init__()
        self.config = {'d_in': d_in, 'd_out': d_out, 'd_hidden': d_hidden, 'k': k, 'add_topk': add_topk}
        self.encoder = t.nn.Linear(d_in, d_hidden)
        self.decoder = t.nn.Linear(d_hidden, d_out)
        self.k = k
        self.add_topk = add_topk

        kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        kaiming_normal_(self.decoder.weight)

    def encode(self, x):
        encoded = self.encoder(x)
        if self.add_topk:
            topk_vals, topk_inds = t.topk(encoded, self.k, dim=1)
            new_encoded = t.zeros_like(encoded)
            new_encoded.scatter_(1, topk_inds, topk_vals)
        else:
            new_encoded = encoded
        return new_encoded

    def forward(self, x):
        enc = self.encode(x)
        decoded = self.decoder(enc)
        return enc, decoded
