import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np


################################################################################
################################################################################
################################################################################
class Module():
    def __init__(self, id):
        if(id == None):
            self.id = 'None'
        else :
            self.id = id

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError

    def zero_grad(self):
        pass

    def optimize(self, *args):
        pass

################################################################################
################################################################################
################################################################################
class Relu(Module):
    def __init__(self):
        Module.__init__(self, 'Relu')

    def forward(self, s0):
        return np.maximum(0, s0)

    def backward(self, x, dl_dx):
        mask = (x > 0).float()
        x = torch.mul(x, mask).view(-1, 1)
        return torch.mul(x, dl_dx)

    def param(self):
        return None

################################################################################
class Tanh(Module):
    def __init__(self):
        Module.__init__(self, 'Tanh')

    def forward(self,s0):
        return s0.tanh()

    def backward(self, x, dl_dx):
        return 4 * (x.view(-1,1).exp() + x.view(-1,1).mul(-1).exp()).pow(-2) * dl_dx
        # 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    def param(self):
        return None


################################################################################
################################################################################
################################################################################
class MSELoss(Module):
    def __init__(self):
        Module.__init__(self, 'MSE')

    def forward(self, v, t):
        return (v.view(2,1) - t).pow(2).sum()
        # return (v.view(-1,1) - t).pow(2).sum()
        # return (v - t).pow(2).sum()

    def backward(self, v, t):
        return 2 * (v.view(2,1) - t)
        # return 2 * (v - t)

    def param(self):
        return None


################################################################################
################################################################################
################################################################################
class Linear(Module):
    def __init__(self, input, output, std, w = None, b = None):
        Module.__init__(self, 'Linear')
        if(w != None):
            self.w = FloatTensor(output,input).normal_(0, std)
        self.w = FloatTensor(output,input).normal_(0, std)
        if(b != None):
            self.b = FloatTensor(output).normal_(0, std)
        self.b = FloatTensor(output).normal_(0, std)
        self.dw = FloatTensor(output,input).zero_()
        self.db = FloatTensor(output).zero_()

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()

    def forward(self, x):
        s = self.w.mv(x.view(-1)) + self.b
        return s

    def backward(self, x, dl_ds):
        self.dw.add_(dl_ds.view(-1, 1).mm(x.view(1, -1)))
        self.db.add_(dl_ds.view(-1))
        dx_previous = self.w.t().mm(dl_ds)
        return dx_previous

    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]

    def optimize(self, eta):
        self.w -= eta * self.dw
        self.b -= eta * self.db

################################################################################
################################################################################
################################################################################
