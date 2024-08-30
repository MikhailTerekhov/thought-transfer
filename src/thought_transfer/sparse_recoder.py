import torch as t


class SparseRecoder(t.nn.Module):
    def __init__(self, d_in, d_out, d_hidden, add_topk=True):
        super(SparseRecoder, self).__init__()
        self.encoder = t.nn.Linear(d_in, d_hidden)
        self.decoder = t.nn.Linear(d_hidden, d_out)
        self.add_topk = add_topk

    def forward(self, x):
        encoded = self.encoder(x)
        if self.add_topk:
            # TODO add topk
            pass
        decoded = self.decoder(encoded)
        return decoded
