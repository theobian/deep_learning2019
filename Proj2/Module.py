import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np


class Module():
    def __init__(self, *args):
        raise NotImplementedError

    def get_parameters(self):
        return []

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def zero_grad(self):
        pass


class ReLU(Module):
    def __init__(self):
        self.id ='Relu'

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, dl_dx):
        mask = (x > 0).float()
        dw = torch.mul(torch.mul(x,mask).view(-1,1),dl_dx)
        return dw


class Tanh(Module):
    def __init__(self):
        self.id = 'Tanh'

    def forward(self, x):
        return x.tanh()

    def backward(self, x, dl_dx):
        return 4 * (x.view(-1,1).exp() + x.view(-1,1).mul(-1).exp()).pow(-2) * dl_dx


class MSELoss(Module):
    def __init__(self):
        self.id = 'MSELoss'

    def forward(self, v, t):
        return (v.view(2,1) - t).pow(2).sum()

    def backward(self, v, t):
        return 2 * (v.view(2,1) - t)


class FullyConnectedLayer(Module):

    def __init__(self, n_in, n_out, std):
        self.id = 'Fully Connected Layer'

        self.w = FloatTensor(n_out, n_in).normal_(std)
        self.b = FloatTensor(n_out).normal_(std)

        self.dw = FloatTensor(n_out,n_in).zero_()
        self.db = FloatTensor(n_out).zero_()

        self.parameters = [[self.w, self.dw], [self.b, self.db]]

    def get_parameters(self):
        return self.parameters


    def forward(self,x):
        s = self.w.mv(x.view(-1)) + self.b
        return s


    def backward(self, x, dl_ds):
        self.dw.add_(dl_ds.view(-1, 1).mm(x.view(1, -1)))
        self.db.add_(dl_ds.view(-1))
        dx_previous = self.w.t().mm(dl_ds)
        return dx_previous


    def zero_grad(self):
            self.dw.zero_()
            self.db.zero_()
