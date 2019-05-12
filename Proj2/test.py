import math

import torch
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt


from Sequential import *
from Module import *
from Optimizer import *
from Utils import *


#==============================================================================#
#==============================================================================#
#==============================================================================#
def main():

    data_nb = 1000
    train_input, train_target = data_set_generation(data_nb)
    test_input, test_target = data_set_generation(data_nb)

    std = 0.1
    model = Linear(FullyConnectedLayer(2, 25, std), Tanh(), FullyConnectedLayer(25, 25, std), Tanh(), FullyConnectedLayer(25, 25, std), Tanh(), FullyConnectedLayer(25, 2, std), Tanh())

    learning_rate = 0.001
    criterion = MSELoss()
    parameters = model.get_parameters()
    optimizer = SGD(parameters, learning_rate)

    epochs = 25
    mini_batch_size = 25
    losses = train(model, criterion, optimizer, train_input, train_target, mini_batch_size, epochs)

    error = evaluate(model, train_input, train_target, mini_batch_size)


if __name__ == '__main__':
    main()





#==============================================================================#
#==============================================================================#
#==============================================================================#
