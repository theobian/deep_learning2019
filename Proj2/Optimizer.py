import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np

################################################################################
################################################################################
################################################################################
class Optimization():
    def __init__(self, *args):
        raise NotImplementedError


################################################################################
class SGD(Optimization):
    def __init__(self, parameters, eta):
        self.eta = eta
        self.parameters = parameters

    def step(self):
        for w in self.parameters:
            w[0]-= self.eta * w[1]


################################################################################
class SGDMomentum(Optimization):
	def __init__(self, parameters, eta, gamma=0.9):
		self.parameters = input_parameters
		self.eta = eta
		self.gamma = gamma

	def step(self, prev_update):
		for w in self.parameters:
			w[0] -= self.eta * w[1]
			self.eta += (self.gamma * prev_update)



################################################################################
################################################################################
################################################################################
