'''
Optimizer
'''
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import math



################################################################################
################################################################################
################################################################################
class Optimization():
	'''
	needs to be overridden
	'''
	def __init__(self, *args):
		raise NotImplementedError



################################################################################
class SGD(Optimization):
	'''
	input: self, parameters to be optimized, learning rate
	output: none
	standard instantiation
	'''
	def __init__(self, parameters, eta):
		self.eta = eta
		self.parameters = parameters



	'''
	input: self
	output: none
	Iterates over the parameters and updates them according to the following rule:
	param = gradient_param * learning_rate
	'''
	def step(self):
		for w in self.parameters:
			w[0]-= self.eta * w[1]



################################################################################
class SGDMomentum(Optimization):
	'''
	input: self, parameters to be optimized, learning rate, update rate/momentum
	output: none
	standard instantiation
	'''
	def __init__(self, parameters, eta, gamma = 0.9):
		self.parameters = input_parameters
		self.eta = eta
		self.gamma = gamma



	'''
	input: self, previous update vector
	output: none
	iterates over the parameters and updates them according to the following rule:
	param = gradient_param * learning_rate
	updates the learning rate according to the following rate:
	learning_rate = momentum * previous_update_vector
	'''
	def step(self, prev_update):
		for w in self.parameters:
			w[0] -= self.eta * w[1]
			self.eta += (self.gamma * prev_update)



################################################################################
################################################################################
################################################################################
