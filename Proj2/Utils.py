'''
Utils
'''
import math
import torch
from torch import FloatTensor
from torch import LongTensor
import numpy as np
import matplotlib.pyplot as plt
import time



################################################################################
################################################################################
################################################################################
'''
input: number of data points to generate
output: binary input and corresponding class according to the following rule: 
target is 1 if point pair is within circle of radius R = sqrt(1/2pi)
''' 
def data_gen(n):
    input = FloatTensor(n, 2).uniform_(0, 1)
    target = FloatTensor(n, 2).uniform_(0, 1)
    target = input.pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).float()
    return input, target



################################################################################
'''
input: input and class to be plotted
output: none
plots the binary data as two different classes according to their labeled class
saves the figure
'''
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
################################################################################
################################################################################
'''
input: model to be trained; binary data and corresponding class; loss and optimization criterions for training, learning rate, epochs, and verbose parameter
output: total losses for each epoch, and model parameters (weights, biases and corresponding gradients) after the last epoch training
this is SGD type update;
iteratively calls: the model's forward pass, then the loss's foward pass for a single binary data point and class
then adds the running loss to the total loss
then set all gradients to zero before computing the model's backward pass for a single binary data point and the loss function's gradient
then updates the parameter weights through the optimizer.
'''
def train(model, train_input, train_target, loss_criterion, optimizer, eta, epochs, verbose):
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
        losses.append(sum_loss)
        if((e%5 == 0 or e == 0 or e == epochs) and verbose):
            print('epoch', e,'loss', sum_loss)
    weights = model.param()
    return losses, weights



################################################################################
# def test_model_no_batch(model, test_input, test_target, loss_criterion):
#     n_errors = 0
#     n_correct = 0
#     cx, cy, ix, iy = [], [], [], []
#     l = []
#     for s in range(len(test_input)):
#         output = model.forward(test_input.narrow(0, s, 1))
#         if(test_target.narrow(0, s, 1).item() != np.argmax(output)):
#             n_errors += 1
#             ix.append(test_input.narrow(0, s, 1)[0][0].item())
#             iy.append(test_input.narrow(0, s, 1)[0][1].item())
#             l.append(0)
#             # print('err', 'l=0', test_target.narrow(0,s,1).item())
#         else:
#             n_correct += 1
#             cx.append(test_input.narrow(0, s, 1)[0][0].item())
#             cy.append(test_input.narrow(0, s, 1)[0][1].item())
#             l.append(1)
#             # print('cor', 'l=1', test_target.narrow(0,s,1).item())
#     # print('errors', n_errors)
#     # print('correct', n_correct)
#     return n_errors, ix, iy, cx, cy, l




################################################################################
'''
input: model to be trained; binary data and corresponding class; loss criterion for testing, and verbose parameter
output: number of misclassified points, coordinates for the vector of misclassified as well as for classified points, and a label vector of the same size as the input data
for each sample within the test set: computes the model forward to precict label as a probability of being one class or the other
the most likely prediction (index corresponding to the max value of the last layer's output) is compared to the actual label 
errors, points, and labels that were misclassifed are logged accordingly
the l parameter returned makes it easier to keep track of which labels were misclassified
'''
def eval(model, test_input, test_target, loss_criterion, verbose):
    n_errors = 0
    cx, cy, ix, iy = [], [], [], []
    l = []
    for s in range(len(test_input)):
        output = model.forward(test_input.narrow(0, s, 1))
        t = test_target.narrow(0, s, 1).item()
        i = test_input.narrow(0, s, 1)
        p = np.argmax(output)
        if(t == p):
            n_correct += 1
            cx.append(i[0][0].item())
            cy.append(i[0][1].item())
            l.append(1)
        else:
            n_errors += 1
            ix.append(i[0][0].item())
            iy.append(i[0][1].item())
            l.append(0)
    return n_errors, ix, iy, cx, cy, l



################################################################################
'''
input: incorrectly labeled and correctl labeled point coordinates (for binary data)
output: none
plots the data points, with two classes -- not according to the true label (which can be determined with the circle boundary)),
but according to the classification truth
'''
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
'''
input: text to be outputted, file idenifier for easy saving
output: none
allows saving model results to csv format file
'''
def write_to_csv(text, file_id):

    with open('Output/test_{}.csv'.format(file_id), mode = 'w') as to_csv:
        for i in range(len(text)):
            to_csv.write(text[i])
            to_csv.write('\n')



################################################################################
################################################################################
################################################################################
