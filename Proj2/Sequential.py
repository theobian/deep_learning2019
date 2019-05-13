import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np

class Sequential():
    def __init__(self, *args):
        raise NotImplementedError

    def forward(self, args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError



class Linear(Sequential):
    def __init__(self, *args):
        self.sequence = args
        self.cache = []
        self.parameters = []
        for elt in self.sequence:
            if len(elt.get_parameters()) != 0:
                self.parameters += elt.get_parameters()

    
    def get_parameters(self):
        return self.parameters

    def forward(self, x):
        self.cache = []
        for i in range(len(self.sequence)):
            self.cache.append(x)
            x = self.sequence[i].forward(x)
        self.cache.append(x)
        return x

    def backward(self, *args):
        dx = args[0]
        for i, elt in reversed(list(enumerate(self.sequence))):
            dx = elt.backward(self.cache[i], dx)
        return dx

    def zero_grad(self):
        for elt in self.sequence:
            elt.zero_grad()
