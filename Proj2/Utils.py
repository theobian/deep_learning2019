import math

import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt


#==============================================================================#
#==============================================================================#
#==============================================================================#
def data_set_generation(nb_data_points):
	input = Tensor(nb_data_points, 2).uniform_(0, 1)
	target = input.pow(2).sum(1).sub((1/(2*math.pi))).sign().add(1).div(2).long()
	return input, target


def data_set_plot(input, target):
	cmap = []
	for i in range(len(target)):
		if(target[i]):
			cmap.append('blue')
		else:
			cmap.append('red')
	fig, ax = plt.subplots()
	plt.scatter(input[:,0], input[:,1], c = cmap)
	circle1=plt.Circle((0,0),np.sqrt(1/(2*math.pi)), color = 'black', fill = False)
	plt.gcf().gca().add_artist(circle1)
	plt.ylim(0, 1);
	plt.xlim(0, 1);
	plt.title('Dataset')
	plt.show()
	fig.savefig('Dataset.png')


#==============================================================================#
#==============================================================================#
#==============================================================================#
def train(model, criterion, optimizer, input, target, mini_batch_size, epochs):
	losses = []
	for e in range(epochs):
		sum_loss = []
		loss = 0
		for b in range(0, input.size(0), mini_batch_size):
			print(input.narrow(0, b, mini_batch_size))
			output = model.forward(input.narrow(0, b, mini_batch_size))
			print(len(output), len(target.narrow(0, b, mini_batch_size)))
			print(criterion.forward())
			loss = criterion.forward(output, target.narrow(0, b, mini_batch_size))
			sum_loss += loss.item()
			model.zero_grad()
			back_prop = criterion.backward(output, target.narrow(0, b, mini_batch_size))
			model.backward(back_prop)
			optimizer.update()
		print('epoch {} loss {} sum_loss {}').format(e+1, loss, sum_loss)
		losses.append(sum_loss)
	return losses


def evaluate(model, input, target, mini_batch_size):
	error = 0
	print(len(input))
	print(len(input[0]))
	for b in range(len(input[0])):
		output = model.forward(input.narrow(0, b, mini_batch_size))
		if np.argmax(output) != np.argmax(target.narrow(0, b, mini_batch_size)):
			error += 1
	return error


#==============================================================================#
#==============================================================================#
#==============================================================================#
