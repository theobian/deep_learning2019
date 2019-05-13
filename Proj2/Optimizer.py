import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np

class Optimizer():
    def __init__(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, parameters, eta):
        self.eta = eta
        self.parameters = parameters

    def update(self):
        for w in self.parameters:
            w[0]-= self.eta * w[1]
