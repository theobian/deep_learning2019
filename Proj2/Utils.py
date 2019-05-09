import math

import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt


def data_set_generation(nb_data_points):
	"""
	Create a dataset made up of uniformly distributed pairs in [0,1]x[0,1]
	type nb_data_points: int
	param nb_data_points: the number of datapoints present in the dataset
	"""

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



# WILL NEED TO HAVE THIS MOVED TO HE MAIN/TEST.py file....
def sample_gen():
	nb_data_points = 1000
	input, target = data_set_generation(nb_data_points)
	return input, target

#==============================================================================#
#==============================================================================#
#==============================================================================#



# from class
def train_model(model, train_input, train_target, mini_batch_size):

    train_input, train_target = Variable(train_input), Variable(train_target)

    criterion = nn.MSELoss()

    eta = 1e-1

    for e in range(0, 25):
        sum_loss = 0

        # Use mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            # narrow(dimension, start, length) → Tensor
            output = model(train_input.narrow(0, b, mini_batch_size))
            # torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            # MSELoss creates a criterion that measures the MSE between each element in the input x and target y
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            # use torch.Tensor.item() to get a Python number from a tensor containing a single value
            sum_loss = sum_loss + loss.item()
            # zero_grad(): sets gradients of all model parameters to zero
            model.zero_grad()
            # back-propagated gradient computation
            loss.backward()

            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)


# from class
def evaluate(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors
