import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def disc_generator (datapoint=1000, plot=True):
	"""Generate a disc dataset, which is not linearly separable

	type datapoint: int
	param datapoint: the number of point in the dataset"""

	#TODO better understand if this is right
	#TODO check if we can move the problem and center it in 0
	dataset = torch.empty(datapoint,2).uniform_(-1,1)
	label = torch.empty(datapoint,2).zero_()

	boundary = math.sqrt((2/math.pi))

	#Not sure if it's the right dimension, maybe try dim=1
	mask = torch.norm(dataset, dim=0)<boundary

	#TODO better check this part
	label = torch.cat([~mask.unsqueeze(1), mask.unsqueeze(1)], dim=1).float().t()

	if plot==True:
		fig, ax = plt.subplots()
		plt.plot(torch.masked_select(dataset[:,0],mask).numpy(), torch.masked_select(dataset[:,1],mask).numpy(),'b.', label='0')
		plt.plot(torch.masked_select(dataset[:,0],~mask).numpy(), torch.masked_select(dataset[:,1],~mask).numpy(),'r.', label='1')
		circle=plt.Circle((0,0), radius, color='k', fill=False)
		plt.legend()
		ax.add_artist(circle)
		plt.show()
		fig.savefig('Dataset.png')

	return dataset, label

	disc_generator(1000, True)

"""def generate_disc_set(nb, plot=True):
    Generate data set. 
    _ nb : number of points. 
    _ plot=bool : plots the data if it's True 
    # input is uniformly distributed in [−1, 1] × [−1, 1], size nbx2 = coordinates of a point
    train_set=torch.empty(nb,2).uniform_(-1,1)
    #train_set=train_set.sub_(train_set.mean())

    #label is 1 inside the disc of radius R and 0 outside
    radius= np.sqrt((2/np.pi))
    
    target=torch.empty(nb,2).zero_()
    mask = torch.norm(train_set, 2, 1) < radius #byte tensor : 1 if inside 
    target=torch.cat([~mask.unsqueeze(1), mask.unsqueeze(1)], dim=1).float().t()
    
    #plot the input 
    if plot==True:
        fig, ax = plt.subplots()
        plt.plot(torch.masked_select(train_set[:,0],mask).numpy(), torch.masked_select(train_set[:,1],mask).numpy(),'b.', label='Class1')
        plt.plot(torch.masked_select(train_set[:,0],~mask).numpy(), torch.masked_select(train_set[:,1],~mask).numpy(),'r.', label='Class2')
        circle=plt.Circle((0,0), radius, color='k', fill=False)
        plt.legend()
        ax.add_artist(circle)
        plt.show()
        fig.savefig('Dataset.png')
    
    return train_set, target





generate_disc_set(1000)"""





