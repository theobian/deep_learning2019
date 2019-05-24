import sys
import os
import torch
from torch.autograd import Variable
from torch.nn import functional as F

import helperfuncs as hlf 


def trainAndTestMLP(model, x_test, y_test, x_train, y_train, y_train_aux, activfuncs, lossfuncs, opttype, learningrate, n_epochs, n_minibatchsize, benchmarktype, weightsharing=False, auxiliaryloss=False):

     # set loss-function of outputlayer
    curr_lossfunc = defineLossFunc(lossfuncs.lossfunc_outlayer)

    # set optimizer type
    optimizer = defineOptimizer(opttype, parameters=model.parameters(), learningrate=learningrate)

    
    train_loss = []
    train_error = []
    test_error = []

    # one initial forward pass
    y_train_pred_aux, y_train_pred = model(activfuncs, x=x_train, weightsharing=weightsharing, auxiliaryloss=auxiliaryloss)
    #print(y_train_pred_aux)
    #exit(0)
    curr_loss = curr_lossfunc(y_train_pred, y_train)

    if auxiliaryloss:
        n_auxlosses = len(y_train_pred_aux)
        curr_auxlossfuncs = n_auxlosses*[None]
        for i in range(n_auxlosses):
            curr_auxlossfuncs[i] = defineLossFunc(lossfuncs.lossfunc_aux)
            curr_loss = (1-lossfuncs.lossaux_weight)*curr_loss + lossfuncs.lossaux_weight/n_auxlosses*curr_auxlossfuncs[i](y_train_pred_aux[i], y_train_aux[i])

    # propagate for the number of epochs
    for i in range(n_epochs):
        ## Backpropagation
        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()
            

        ## Forward pass
        y_train_pred_aux, y_train_pred = model(activfuncs, x=x_train, weightsharing=weightsharing, auxiliaryloss=auxiliaryloss)
            

        ## Training loss and training/test error
        curr_loss = curr_lossfunc(y_train_pred, y_train)
        if auxiliaryloss:
            for i in range(n_auxlosses):
                curr_loss = (1-lossfuncs.lossaux_weight)*curr_loss + lossfuncs.lossaux_weight/n_auxlosses*curr_auxlossfuncs[i](y_train_pred_aux[i], y_train_aux[i])
        
        if benchmarktype == "SCAN_NEPOCHS":
            ### store training loss and training/test error for every epoch
            train_loss.append(curr_loss.data[0])
            train_error.append(hlf.calculateError(y_train_pred, y_train))
            y_test_pred_aux, y_test_pred = model(activfuncs, x=x_test, weightsharing=weightsharing, auxiliaryloss=auxiliaryloss)
            test_error.append(hlf.calculateError(y_test_pred, y_test))                    


    if benchmarktype == "SCAN_NONE":
        ## store training loss and training/test error of last epoch
        train_loss.append(curr_loss.data[0])
        train_error.append(hlf.calculateError(y_train_pred, y_train))
        y_test_pred_aux, y_test_pred = model(activfuncs, x=x_test, weightsharing=weightsharing, auxiliaryloss=auxiliaryloss)
        test_error.append(hlf.calculateError(y_test_pred, y_test))


    return torch.FloatTensor(test_error), torch.FloatTensor(train_loss), torch.FloatTensor(train_error)


def defineLossFunc(lossfunc):
    if lossfunc == "MSE":
        lossfunc = torch.nn.MSELoss(reduction='mean')

    elif lossfunc == "MAE":
        lossfunc = torch.nn.L1Loss(reduction='mean')

    elif lossfunc == "BinaryCrossEntropy":
        lossfunc = torch.nn.BCELoss()

    elif lossfunc == "MultiClassCrossEntropy":
        #lossfunc = torch.nn.CrossEntropyLoss(reduction='mean')
        lossfunc = torch.nn.CrossEntropyLoss()

    else:
        print("ERROR: The specified loss function " + lossfunc + " is not implemented.")
        sys.exit(2)

    return lossfunc

def defineOptimizer(opt, parameters, learningrate):
    if opt == "Adam":
        curr_opt = torch.optim.Adam(params=parameters, lr=learningrate)

    elif opt == "Adamax":
        curr_opt = torch.optim.Adamax(params=parameters, lr=learningrate)

    elif opt == "Adagrad":
        curr_opt = torch.optim.Adagrad(params=parameters, lr=learningrate)

    elif opt == "Adadelta":
        curr_opt = torch.optim.Adadelta(params=parameters, lr=learningrate)

    else:
        print("ERROR: The specified optimizer " + opt + " is not implemented.")
        sys.exit(2)

    return curr_opt

def reshapeData(curr_layertype, prev_layertype, xdata):
    if curr_layertype == "Regular1d" and prev_layertype == "Conv2d":
        return xdata.view(xdata.size(0),-1)

    return xdata
        