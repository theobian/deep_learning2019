'''
Sequential
'''
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import math




################################################################################
################################################################################
################################################################################
class Sequential():
    '''
    input: self
    output: none
    within elements, each element is an instance of the Module class
    the cache contains an array of an input after it is passed through each forward pass
    standard instantiation
    '''
    def __init__(self, *args):
        self.elements = args
        self.cache = []


     
    '''
    input: self, data
    output: input after having gone through all Module instances' forward pass
    the forward pass computes an output through a list of Module instances
    the starting input, as well as each input after it is passed through a layer, is saved
    this list of saved inputs is necessary to compute the backward-propagatino gradients
    '''    
    def forward(self, input):
        self.cache = []
        self.cache.append(input)
        for i, elt in enumerate(self.elements):
            input = elt.forward(input);
            self.cache.append(input)
        return input



    '''
    input: self, data and model gradient -- weights and bias gradients
    output: gradient
    the backward pass computes an output through a list of Module instances
    the starting gradient as well as each gradient after it is passed through a layer, is saved
    the backpropagation can happen because a list of inputs at each layer/point has been cached
    this backpropagation algorithm in fine allows the update of weights
    '''
    def backward(self, input, grad):
        error = []
        error.append(grad)
        for i, elt in reversed(list(enumerate(self.elements))):
            grad = elt.backward(self.cache[i], grad)
            error.append(grad)
        error.reverse()
        return grad, error



    '''
    input: self
    output: none
    iterates over elements within the Sequential instance and sets their respective gradients to zero
    this is used after each model training iteration, to make sure all gradients are computed over one iteration
    '''
    def zero_grad(self):
        for element in self.elements:
            element.zero_grad()



    '''
    input: self
    output: parameters of the whole sequence as a list
    iterates over elements within the Sequential instance, and fetches their parameters
    this is used to get parameters (weights, biases, and their gradients) for optimization
    '''
    def param(self):
        parameters = []
        for elt in self.elements:
            if elt.param() != None:
                parameters += elt.param()
        return parameters



################################################################################
################################################################################
################################################################################
