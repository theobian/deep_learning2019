from torch import empty
import math
import numpy as np
from torch import FloatTensor
import torch
#==============================================================================#
#==============================================================================#
#==============================================================================#
class Module(object):

	def __init__ (self, instance, *args):
		self.instance = instance

	def get_parameters(self):
		raise NotImplementedError


	def forward(self, *args):
		raise NotImplementedError


	def backward(self, *args):
		raise NotImplementedError


#==============================================================================#
#==============================================================================#
#==============================================================================#
class FullyConnectedLayer(Module):

	def __init__(self, input_size, output_size, std):
		super().__init__(self, 'FullyConnectedLayer')
		self.input_size = input_size
		self.output_size = output_size
		self.w = torch.empty(output_size, input_size).normal_(0, std)
		self.b = torch.empty(output_size, input_size).normal_(0, std)
		self.dl_dw = torch.empty(self.w.size())
		self.dl_db = torch.empty(self.b.size())

		# self.w = FloatTensor(output_size, input_size).normal_(0, std)
		# self.b = FloatTensor(output_size).normal_(0, std)
		# self.dl_dw = FloatTensor(output_size, input_size).zero_()
		# self.dl_db = FloatTensor(output_size).zero_()


	def get_parameters(self):
		return [[self.w, self.dl_dw], [self.b, self.dl_db]]


	def zero_grad(self):
		self.dl_dw.zero_()
		self.dl_db.zero_()


	def forward(self, input):
		print(len(input), len(self.w), len(self.b))
		return self.w.mv(input.view(-1)) + self.b


	def backward(self, x, dl_ds):
		self.dl_dw.add_(dl_ds.view(-1,1).mm(x.view(1,-1)))
		self.dl_db.add_(dl_ds)
		return self.w.t().mv(dl_ds)


#==============================================================================#
#==============================================================================#
#==============================================================================#
class Tanh(Module):

	def __init__(self):
		super().__init__(self, 'Tanh')


	def get_parameters(self):
		return []


	def forward(self, x):
		return x.tanh()


	def backward(self, x, dl_dx):
		return (4 * (x.exp() + x.mul(-1).exp()).pow(-2)) * dl_dx



class ReLU(Module):

	def __init__(self):
		super().__init__(self, 'ReLU')


	def get_parameters(self):
		return []


	def forward(self, x):
		return np.maximum(0, data)


	def backward(self, x, dl_dx):
		gradient = [1. * x[i] for i in range(len(x)) if x>0]
		return gradient.mul(dl_dx)


#==============================================================================#
#==============================================================================#
#==============================================================================#
class MSELoss(Module):

	def __init__(self):
		super().__init__(self, 'MSELoss')


	def get_parameters(self):
		return []


	def forward(self, v, t):
		return (v - t).pow(2).sum()


	def backward(self, v, t):
		return 2 * (v - t)


#==============================================================================#
#==============================================================================#
#==============================================================================#
