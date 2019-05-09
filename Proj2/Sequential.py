from torch import empty
import math
import numpy as np
from Module import *


#  redo comments -- especially: not specific to MLP creation: made it so that any thing else can be implemented....
# also: the activation function and linear thing...
class Sequential(Module):
	def __init__(self, input_size, output_size, init_std, activation, w=None, b=None):
		super().__init__()
		"""Initialize the parameters for a multilayer perceptron

		type input_size: int
		param input_size:  number of input units, the dimension of
		the space in which the input datapoints lie.

		type output_size: int
		param output_size: number of output units, the dimension of
		the space in which the output datapoints lie.

		init std: double
		param init_std: the weights are initialized using a normal distribution
		centered in 0 and with std = init_std

		type w: torch.tensor size(output_size, input_size)
		param w: the weights connecting input to output layers

		type b: torch.tensor size(output_size)
		param b: the bias associated to each connection between input to output
		layers

		type activation: string
		param activation: specifies the activation function, choose between the
		ones described in Module.py"""

		self.linear = Linear(
			input_size = input_size,
			output_size = output_size,
			init_std = init_std,
			w = w,
			b = b)

		self.activation_selection = activation

		if (activation_selection = 'tanh'):
			self.activation = Tanh()

		if (activation_selection = 'relu'):
			self.activation = ReLU()

		self.input = input

		def forward(self, input_layer):
			"""Compute the forward path by concatenating the forward paths of
			the modules composing the Sequential class

			type input_layer: torch.tensor size(input_size)
			param input_layer: the input tensor containing the values of
			each neuron"""

			s = linear.forward(x0)
			x1 = activation.forward(s)

			return x0, s, x1

		def backward(self, input_layer, dl_dx1):
			"""Compute the backward path by concatenating the backward paths of
			the modules composing the Sequential class

			type input_layer: torch.tensor size(input_size)
			param input_layer: the input tensor containing the values of
			each neuron

			type dl_dx1: torch tensor
			param dl_dx1: the derivative of the loss with respect to the layer.
			Can come either from the backward method of a loss class or the
			backward method of a Sequential class"""

			dl_ds = activation.backward(dl_dx1)
			dl_dx = linear.backward(input_layer, dl_ds)

			return dl_ds, dl_dx
