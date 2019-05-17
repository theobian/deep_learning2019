import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import matplotlib.pyplot as plt

################################################################################
################################################################################
################################################################################
def data_gen(nb):
    input = FloatTensor(nb, 2).uniform_(0, 1)
    target = FloatTensor(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).float()
    # print(target.size())
    return input, target

################################################################################
def dummy_data_gen(nb):
    input = FloatTensor(nb, 2).uniform_(-1, 1)
    target = FloatTensor(nb, 2).uniform_(-1, 1)
    target = input.sum(1).sign().add(1).div(2).float()
    return input, target

################################################################################
def data_plot(input, target):
    cmap = []
    for i in range(len(target)):
        if(target[i]):
            cmap.append('green')
        else:
            cmap.append('red')
    fig, ax = plt.subplots()
    plt.scatter(input[:,0], input[:,1], s=1,c = cmap)
    circle1 = plt.Circle((0,0),np.sqrt(1/(2*math.pi)), color = 'black', fill = False)
    plt.gcf().gca().add_artist(circle1)
    plt.ylim(0, 1);
    plt.xlim(0, 1);
    plt.title('Dataset')
    plt.show()
    fig.savefig('Dataset.png')

################################################################################
def dummy_data_plot(input, target):
    cmap = []
    for i in range(len(target)):
        if(target[i]):
            cmap.append('blue')
        else:
            cmap.append('red')
    fig, ax = plt.subplots()
    plt.scatter(input[:,0], input[:,1], c = cmap)
    x = np.linspace(-1, 1, 100)
    y = -x
    plt.plot(x, y, c = 'black')
    plt.ylim(-1, 1);
    plt.xlim(-1, 1);
    plt.title('Dataset')
    plt.show()
    fig.savefig('Dataset.png')

################################################################################
def data_norm(input, target):
    mean, std = input.mean(),input.std()
    input.sub_(mean).div_(std)
    input.sub_(mean).div_(std)
    #  can't use Variable because it keeps track of the gradient EVEN THOUGh it is now deprecated.
    # train_input, train_target = Variable(input), Variable(target)
    return input, target

################################################################################
def data_transform(input):
    # specific to THE DATA WE ARE USING
    transform = []
    transform.append(input[:, 0].reshape((-1, 1)))
    transform.append(input[:, 1].reshape((-1, 1)))
    t_data = transform[0].pow(2) + transform[1].pow(2)
    return t_data, transform

################################################################################
def data_reshape(input):
    X = input
    # adding a new dimension to X
    X1 = X[:, 0].reshape((-1, 1))
    X2 = X[:, 1].reshape((-1, 1))
    X3 = (X1**2 + X2**2)
    X = np.hstack((X, X3))

    # visualizing data in higher dimension
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection = '3d')
    # axes.scatter(X1, X2, X1**2 + X2**2, c = Y, depthshade = True)
    # plt.show()
    return X


################################################################################
################################################################################
################################################################################
def train_model_no_batch(model, train_input, train_target, loss_criterion, optimizer, eta, epochs):
    losses = []
    for e in range(epochs):
        sum_loss = 0
        for s in range(len(train_input)):
            output = model.forward(train_input.narrow(0, s, 1))
            loss = loss_criterion.forward(output, train_target.narrow(0, s, 1))
            sum_loss += loss.item()
            model.zero_grad()
            grad, grad_err = model.backward(output, loss_criterion.backward(output, train_target.narrow(0, s, 1)))
            optimizer.step()
            model.optimize(eta)
        losses.append(sum_loss)
        if(e%5 == 0 or e == 0 or e == epochs): print('epoch', e,'loss', sum_loss)
    weights = model.param()
    return losses, weights


################################################################################
def train_model_batch(model, train_input, train_target, loss_criterion, optimizer, epochs, mini_batch_size):
    losses = []
    weights = []
    for e in range(epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward((train_input.narrow(0, b, mini_batch_size)))
            loss = loss_criterion.forward((output, train_target.narrow(0, b, mini_batch_size)))
            model.zero_grad()
            grad, grad_err = model.backward(loss_criterion.backward(output, train_target.narrow(0, b, mini_batch_size)))
            sum_loss += loss.item()
            '''needs to access the model parameters themselves: the WEIGHTS!!!!'''
            # should implement batching within batching and update the optimizer only after a full batch
            optimizer.step()
            losses.append(sum_loss)
        print('epoch',e, 'loss', sum_loss)
    return losses, weights

################################################################################
def test_model_no_batch(model, test_input, test_target, loss_criterion):
    n_errors = 0
    n_correct = 0
    cx, cy, ix, iy = [], [], [], []
    l = []
    for s in range(len(test_input)):
        output = model.forward(test_input.narrow(0, s, 1))
        if(test_target.narrow(0, s, 1).item() != np.argmax(output)):
            n_errors += 1
            ix.append(test_input.narrow(0, s, 1)[0][0].item())
            iy.append(test_input.narrow(0, s, 1)[0][1].item())
            l.append(0)
            # print('err', 'l=0', test_target.narrow(0,s,1).item())
        else:
            n_correct += 1
            cx.append(test_input.narrow(0, s, 1)[0][0].item())
            cy.append(test_input.narrow(0, s, 1)[0][1].item())
            l.append(1)
            # print('cor', 'l=1', test_target.narrow(0,s,1).item())
    # print('errors', n_errors)
    # print('correct', n_correct)
    return n_errors, n_correct, ix, iy, cx, cy, l

################################################################################
def test_model_wip(model, test_input, test_target, loss_criterion):
    n_errors = 0
    n_correct = 0
    cx, cy, ix, iy = [], [], [], []
    l = []
    for s in range(len(test_input)):
        output = model.forward(test_input.narrow(0, s, 1))
        t = test_target.narrow(0, s, 1).item()
        i = test_input.narrow(0, s, 1)
        p = np.argmax(output)
        # print(t)
        # print(p)
        if(t == p):
            n_correct += 1
            cx.append(i[0][0].item())
            cy.append(i[0][1].item())
            l.append(1)
            # print('cor \n')
        else:
            n_errors += 1
            ix.append(i[0][0].item())
            iy.append(i[0][1].item())
            l.append(0)
            # print('mis \n')

    # print('errors', n_errors)
    # print('correct', n_correct)
    return n_errors, n_correct, ix, iy, cx, cy, l

################################################################################
def test_model_batch(model, test_input, test_target, loss_criterion, mini_batch_size):
    n_errors = 0
    predicted = np.zeros(2, len(test_target), float)
    for b in range(0, test_input.size(0), mini_batch_size):
        output = model.forward(test_input.narrow(0, b, mini_batch_size))
        for s in range(mini_batch_size):
            if test_target[b+s] != np.argmax(output[s]):
                n_errors += 1
                predicted[0].append(test_input[b+s], output[s])
            else:
                predicted[1].append(test_input[b+s], output[s])
    print(n_errors)
    print(n_correct)
    return n_errors, predicted

################################################################################
def evaluation():
    # loop over epochs to see if the loss goes down
    # loop over learning rates to see if out-of-range lr make No sense as they should
    pass


################################################################################
def plot_results(ix, iy, cx, cy):
    fig, ax = plt.subplots()
    plt.scatter(ix, iy, s = 1, c = 'red', label = 'Misclassified')
    plt.scatter(cx, cy, s = 1, c = 'green', label = 'Correctly Classified')
    circle1=plt.Circle((0,0),np.sqrt(1/(2*math.pi)), color = 'black', fill = False)
    plt.gcf().gca().add_artist(circle1)
    plt.ylim(-0.01, 1.1);
    plt.xlim(-0.01, 1.1);
    plt.legend(loc = 1)
    plt.title('Results')
    plt.show()
    fig.savefig('Results.png')

################################################################################
def dummy_plot_results(ix, iy, cx, cy):
    fig, ax = plt.subplots()
    plt.scatter(ix, iy, s = 1, c = 'red', label = 'Misclassified')
    plt.scatter(cx, cy, s = 1, c = 'green', label = 'Correctly Classified')
    x = np.linspace(-1, 1, 100)
    y = -x
    plt.plot(x, y, c = 'black')
    plt.ylim(-1, 1);
    plt.xlim(-1, 1);
    plt.legend(loc = 1)
    plt.title('Results')
    plt.show()
    fig.savefig('Results.png')

################################################################################
################################################################################
################################################################################
