'''
Module
Forward and Backward passes need to be defined explicitly
For Activation and Loss Functions, the backward pass is defined by the forward's gradient
'''
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import math




################################################################################
################################################################################
################################################################################
class Module():
    '''
    input: self, string identifier
    output: none
    set id to None by default
    standard instantiation
    '''
    def __init__(self, id):
        if(id == None):
            self.id = 'None'
        else :
            self.id = id


    '''
    has to be overridden
    '''
    def forward(self, *args):
        raise NotImplementedError


    '''
    has to be overridden
    '''
    def backward(self, *args):
        raise NotImplementedError


    '''
    has to be overridden
    '''
    def param(self):
        raise NotImplementedError


    '''
    does not have to be overridden
    should be overridden for Module instances that have parameter gradients
    '''
    def zero_grad(self):
        pass



################################################################################
################################################################################
################################################################################
class Linear(Module):
    '''
    input: input and output sizes for layer determination, weights and biases and standard deviation
    output: none
    weights and biases are by default initialized with the std provided according to a normal distribution
    gradients for weights and biases are set to zero
    standard instantiation
    '''
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

    

    '''
    sets gradients to zero
    '''    
    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()



    '''
    input: self, input
    output: input after forward pass
    this computes forward according to the formula: input * weight + bias
    the input is flattened into a vector and weights are multipled with it
    '''
    def forward(self, x):
        s = self.w.mv(x.view(-1)) + self.b
        return s



    '''
    input: input, gradient
    output: gradient from before forward pass
    this allows the backpropagation of errors
    the weight and biases are updated thanks to the loss function gradient parameter
    '''
    def backward(self, x, dl_dx):
        self.dw.add_(dl_dx.view(-1, 1).mm(x.view(1, -1)))
        self.db.add_(dl_dx.view(-1))
        dx_previous = self.w.t().mm(dl_dx)
        return dx_previous



    '''
    returns a list of parameters to use for optimization
    '''
    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]



################################################################################
################################################################################
################################################################################
class ReLU(Module):
    '''
    input: self
    output: none
    standard instantiation
    '''
    def __init__(self):
        Module.__init__(self, 'Relu')



    '''
    input: self, input
    output: input after forward pass
    '''
    def forward(self, x):
        return np.maximum(0, x)



    '''
    input: self, input, gradient
    output: backward pass propagation according to input
    allows the computation of error backward propagation
    '''
    def backward(self, x, dl_dx):
        mask = (x > 0).float()
        x = torch.mul(x, mask).view(-1, 1)
        return torch.mul(x, dl_dx)



    '''
    input: self
    output: None 
    differentiates this Module instance from those with parameters
    '''
    def param(self):
        return None



################################################################################
class Tanh(Module):
    '''
    input: self
    output: none
    standard instantiation
    '''
    def __init__(self):
        Module.__init__(self, 'Tanh')



    '''
    input: self, input
    output: input after forward pass
    '''
    def forward(self,s0):
        return s0.tanh()



    '''
    input: self, input, gradient
    output: backward pass propagation according to input
    allows the computation of error backward propagation
    '''
    def backward(self, x, dl_dx):
        return 4 * (x.view(-1,1).exp() + x.view(-1,1).mul(-1).exp()).pow(-2) * dl_dx
    


    '''
    input: self
    output: None 
    differentiates this Module instance from those with parameters
    '''
    def param(self):
        return None



################################################################################
################################################################################
################################################################################
class MSELoss(Module):
    '''
    input: self
    output: none
    standard instantiation
    '''
    def __init__(self):
        Module.__init__(self, 'MSE')



    '''
    input: self, input
    output: input after forward pass
    '''
    def forward(self, v, t):
        return (v.view(2,1) - t).pow(2).sum()



    '''
    input: self, input, gradient
    output: backward pass propagation according to input
    allows the computation of error backward propagation
    '''
    def backward(self, v, t):
        return 2 * (v.view(2,1) - t)
    


    '''
    input: self
    output: None 
    differentiates this Module instance from those with parameters
    '''
    def param(self):
        return None



################################################################################
################################################################################
################################################################################
