import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np


################################################################################
################################################################################
################################################################################
class Sequential():
    def __init__(self, *args):
        self.elements = args
        self.cache = []

    def forward(self, input):
        self.cache = []
        self.cache.append(input)
        for i, elt in enumerate(self.elements):
            input = elt.forward(input);
            self.cache.append(input)
        return input

    def backward(self, input, grad):
        error = []
        error.append(grad)
        for i, elt in reversed(list(enumerate(self.elements))):
            grad = elt.backward(self.cache[i], grad)
            error.append(grad)
        error.reverse()
        return grad, error

    def zero_grad(self):
        for element in self.elements:
            element.zero_grad()

    def param(self):
        parameters = []
        for elt in self.elements:
            if elt.param() != None:
                parameters += elt.param()
        return parameters

    def optimize(self, eta):
        for elt in self.elements:
            elt.optimize(eta)


################################################################################
################################################################################
################################################################################
