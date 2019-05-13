import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import matplotlib.pyplot as plt



def data_set_generation(n):
    input = FloatTensor(n, 2).uniform_(0, 1)
    target = FloatTensor(n, 2).zero_()
    target = input.pow(2).sum(1).sub((1/(2*math.pi))).sign().add(1).div(2).long()

    return input, target


def generate_disc_set(nb):
    train_set=FloatTensor(nb,2).uniform_(-1,1)
    radius= np.sqrt((2/np.pi))
    target=FloatTensor(nb,2).zero_()
    mask = torch.norm(train_set, 2, 1) < radius #byte tensor : 1 if inside
    target=torch.cat([~mask.unsqueeze(1), mask.unsqueeze(1)], dim=1).float().t()
    return train_set, target


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



def train(model, train_input, train_target, epochs,  mini_batch_size, criterion, optimizer, verbatim):
    nb_samples=train_input.size(0)
    for e in range(epochs): # for every epoch

        lower_batch=0
        loss=0
        for batch in np.append(np.arange(0,nb_samples,nb_samples//mini_batch_size),nb_samples): #for every batches

            for sample in range(lower_batch,batch): #for every samples in the batch
                output = model.forward(train_input.narrow(0, sample, 1))
                loss += criterion.forward(output, train_target.narrow(1, sample, 1))
                model.zero_grad()
                model.backward(criterion.backward(output,train_target.narrow(1, sample, 1))) 
            optimizer.update()

            lower_batch=batch

        print('epoch', e+1,  'loss', loss)
        print(criterion.forward(output, train_target.narrow(1,1,1)))

def train__(model, train_input, train_target, epochs,  mini_batch_size, criterion, optimizer, verbatim):
    # batch_size = mini_batch_size
    # nb_samples=train_input.size(0)
    best_parameters = []
    min_loss = 0
    losses = []
    # X = train_input
    # Y = train_target
    # for e in range(epochs):
    #     lower_batch = 0
    #     loss = 0
    #     for b in np.arange(0, nb_samples, nb_samples//mini_batch_size):
    #         for i in range(mini_batch_size):
    #             for sample in range(lower_batch, i):
    #                 output = model.forward(train_input.narrow(0, sample, 1))
    #                 loss += criterion.forward(output, train_target.narrow(1, sample, 1))
    #                 model.zero_grad()
    #                 model.backward(criterion.backward(output,train_target.narrow(1, sample, 1)))
    #                 lower_batch = i
        
        # permutation = torch.randperm(X.size()[0])
        # # for i in range(mini_batch_size):
        # for i in range(0, X.size()[0], batch_size):
        #     # optimizer.zero_grad()
        #     # indices = permutation[i : i + batch_size]
        #     # batch_x, batch_y = X[indices], Y[indices]
        #     # outputs = model.forward(batch_x)
        #     # loss  = criterion.forward(outputs, batch_y)
        #     # criterion.backward())
        #     # optimizer.update()

        #     output = model.forward(train_input.narrow(0, 1, 1))
        #     loss += criterion.forward(output, train_target.narrow(1, 1, 1))
        #     model.zero_grad()
        #     model.backward(criterion.backward(output,train_target.narrow(1, 1, 1)))

        #     optimizer.update()

            

        # if( min_loss == 0 or min_loss > loss ):
        #     best_parameters = model.get_parameters()
        #     min_loss = loss

        # losses.append(loss.item())

        # if(verbatim):
        #     print('epoch', e+1,  'loss', loss.item())


    return min_loss, best_parameters, losses




def evaluate(model, test_input, test_target, verbatim):

    n_error = 0
    mask = []

    for i in range(len(test_target[0])):
        output = model.forward(test_input.narrow(0, i, 1))
        test_sample = test_input.narrow(0, i, 1)
        if np.argmax(output) != np.argmax(test_target.narrow(1, i, 1)):
            n_error += 1
            mask.append(0)
        else:
            mask.append(1)
    if(verbatim):
        print('error: {}%'.format( (n_error*100) / len(test_target[0]) ))

    # mask_test = []
    # for i in range(len(test_target[0])):
    #     output = model.forward(test_input.narrow(0 i, 1))
    #     test_sample = test_input.narrow(0, i, 1)
    #     mask_test = (1 if np.argmax(output) == np.argmax(test_target.narrow(1, i, 1)))
    #     return [word for word in lst if len(word) > 5]
    #     n_error = len(test_target[0]) - mask.sum()


    return n_error, mask


def plot_results(x, y, plot_label, color):
    fig, ax = plt.subplots()
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.plot(x, y, color = color, label = plot_label)
    plt.legend()

    radius= np.sqrt((2/np.pi))
    circle=plt.Circle((0,0), radius, color='k', fill=False)
    ax.add_artist(circle)
    plt.show()
