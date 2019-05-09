import math

import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt


def data_set_generation(nb_data_points):
	"""
	Create a dataset made up of uniformely distributed pairs in [0,1]x[0,1]
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
