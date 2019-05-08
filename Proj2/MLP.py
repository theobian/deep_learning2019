from mod import *
from Sequential_DL import *

import math
import numpy as np
import torch

class MLP(Module):
	"""Defines a Multi Layer perceptron with three hidden layers, each with 25 
	units and tanh activation"""
	
	def __init__(self):
		super().__init__()
		self.linear_1 = Sequential_DL.Sequential(2,25,0.1,'tanh')
		self.linear_2 = Sequential_DL.Sequential(25,25,0.1,'tanh')
		self.linear_3 = Sequential_DL.Sequential(25,25,0.1,'tanh')
		self.linear_4 =Sequential_DL.Sequential(25,2,1,'tanh')

	def zero_grad(self):
		"""Reset the derivative, used for each iteration of backprop"""
		linear_1.zero_grad()
		linear_2.zero_grad()
		linear_3.zero_grad()
		linear_4.zero_grad()


	def forward(self, input_data):
		x0 = input_data
		s1, x1 = linear_1.forward(x0)
		s2, x2 = linear_2.forward(x1) 
		s3, x3 = linear_2.forward(x2) 
		s4, x4 = linear_2.forward(x3)

		return x0, s1, s2, s3, s4, x1, x2, x3, x4

	def backward(self, x1, x2, x3, x4, dl_dx4): #TODO we could do better this maybe and not having to pass each layer
		"""Compute the backward path by concatenating the backward paths of 
			the modules composing the Sequential class

			type input_layer: torch.tensor size(input_size)
			param input_layer: the input tensor containing the values of
			each neuron

			type dl_dx4: torch tensor
			param dl_dx4: the derivative of the loss with respect to the output
			layer. Comes from the backward method of a loss class."""

			dl_ds4, dl_dx3 = linear_4.backward(x4, dl_dx4)
			dl_ds3, dl_dx2 = linear_4.backward(x4, dl_dx4)
			dl_ds2, dl_dx1 = linear_4.backward(x4, dl_dx4)
			dl_ds1, dl_dx0 = linear_4.backward(x4, dl_dx4)#TODO like this we compute the derivative of the input it's weird, I can change it if you think it might be dangerous or weird
			return dl_ds1, dl_ds2, dl_ds3, dl_ds4, dl_dx1, dl_dx2, dl_dx3, dl_dx4 








