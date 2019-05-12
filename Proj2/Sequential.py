from torch import empty
import math
import numpy as np
from Module import *


#==============================================================================#
#==============================================================================#
#==============================================================================#
class Sequential():

	def __init__(self, *args):
		raise NotImplementedError


	def get_parameters(self):
		raise NotImplementedError


	def forward(self):
		raise NotImplementedError


	def backward(self):
		raise NotImplementedError


#==============================================================================#
#==============================================================================#
#==============================================================================#
class Linear(Sequential):

	def __init__(self, *args):
		self.sequence = args
		self.cache = []


	# could be implemented in the init method
	def get_parameters(self):
		parameters = []
		for elt in self.sequence:
			if len(elt.get_parameters()) != 0 :
				print(elt, len(elt.get_parameters()))
				parameters.append(elt.get_parameters())


	def zero_grad(self):
		[elt.zero_grad() for elt in self.sequence]


	def forward(self, input):
		self.cache = []
		self.cache.append(input)
		for i, elt in enumerate(self.sequence):
			print(input.size())
			input = elt.forward(input)
			self.cache.append(input)
		return input


	def backward(self, input):
		for i, elt in enumerate(self.sequence[::-1]):
			input = elt.backward(self.cache[i], input)
		return input


#==============================================================================#
#==============================================================================#
#==============================================================================#
