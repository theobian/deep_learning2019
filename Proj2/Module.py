from torch import empty
import math
import numpy as np



#  comment in

# don't need to override methods to have them empty.

# not sure we need to inherit from object... we're in Python3
# if you do, you should probably super
class Module(object):

	def __init__ (self, instance='Undefined'):
		self.instance = instance #string defining which type of module object is used

	def forward(self, *input):
		raise NotImplementedError

	def backward(self, *gradwrtoutput):
		raise NotImplementedError

# param is useless here. or should at least return physical location or something...
	def param(self):
		return []

#==============================================================================#
#==============================================================================#
#==============================================================================#

class Linear (Module):
	def __init__(self, input_size, output_size, init_std, w=None, b=None):
		super().__init__()

		#Gives the possibility to initialize w and b with fixed initial values
		if w is None:
			w=empty(output_size, input_size).normal(0, init_std)
		if b is None:
			b=empty(output_size).normal(0, init_std)

		self.w = w
		self.b = b

		self.dl_dw = empty(w.size())
		self.dl_db = empty(b.size())


	def zero_grad(self):
		"""Reset the derivative, used for each iteration of backprop"""
		self.dl_dw.zero()
		self.dl_db.zero()


	def forward(self, input_layer):
		"""Multiplies the entries by their respective weight
		and add the bias. Standar forward pass for a fully connected
		linear layer"""

		#TODO maybe add input_layer.view(-1) to reshape it as a precaution for
		#wrong inputs?
		return self.w.mv(input_layer) + self.b


	def param(self):
		"""Return the parameters with their corrispective derivative"""
		return [[self.w, self.dl_dw] [self.b, self.dl_db]]


	def backward(self, x, dl_ds):

		"""Returning the derivative of the previous layer"""
		dl_dx_antecedent = self.w.t().mv(dl_ds)

		"""Update the derivated weights and biases"""
		self.dl_dw.add_(dl_ds.view(-1,1).mm(x.view(1,-1)))
		self.dl_db.add_(dl_ds)i

		return dl_dx_antecedent

#==============================================================================#
#==============================================================================#
#==============================================================================#

class Tanh(Module):

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.tanh()

	def backward(self, x, dl_dx):
		#TODO CHECK IF NOT MULTIPLICATION WITH SOMETHING ELSE
		return (4 * (x.exp() + x.mul(-1).exp()).pow(-2))*dl_dx



class ReLU(Module):

	def __init__(self):
		super().__init__()

	def forward(self, x):
		#TODO change with anything but a for loop if possible
		# return np.maximum(0, data)
		for i, s in enumerate(x):
			if s < 0:
				x[i]=0
		return x

	def backward(self, x, dl_dx):
		gradients = 1. * (x > 0)
		return gradients * dl_dx


#==============================================================================#
#==============================================================================#
#==============================================================================#


class MSELoss (Module):
	"""Mean Square Error Loss metric to compare output to taget label"""

	def __init__(self):
		super().__init__()


	def forward(self, v, t):
		"""Compute the forward path inherited by Module mother class

		type v: torch.tensor
		param v: the output of the neural network

		type t: torch.tensor
		param t: the target"""

		return (v - t).pow(2).sum()



	def backward(self, v, t):
		"""Compute the forward path inherited by Module mother class

		type v: torch.tensor
		param v: the output of the neural network

		type t: torch.tensor
		param t: the target"""

		return 2 * (v - t)b
