import torch
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(torch.nn.Module):

    def __init__(self, inlay_npercs, hidlays_npercs, outlay_npercs):
        super(MLP, self).__init__()

        self.n_hiddenlayers = length(hidlays_npercs)
        # add input layer
        self.model = nn.ModuleList([nn.Linear(inlay_npercs, hidlays_npercs[0])])
        # add hidden layers
        self.model.extend([nn.Linear(hidlays_npercs[k], hidlays_npercs[k+1]) for k in range(self.n_hiddenlayers-1)])
        # add output layer.
        self.model.append(nn.Linear(hidlays_npercs[self.n_hiddenlayers-1], outlay_npercs))


    def forward(self, all_activfuncs, x):
        # propagate through hidden layers
        for k in range(self.n_hiddenlayers+2):
            x = self.model[k](x)
            x = applyActivationFunc(x, all_activfuncs[k])
        return x


    def trainMLP(self, x_train, y_train, all_activfuncs, lossfunc, learningrate, n_epochs):
        # TODO: error checking of arguments: especially hidlays_activfuncs

        # define loss-function
        lossfunc = createLossFunc(lossfunc)

        for t in range(n_epochs):
            # forward pass
            y_pred = forward(self, all_activfuncs, x_train)
            loss = lossfunc(y_pred, y_train)

            # backward pass
            ###self.model.zero_grad()
            loss.backward()

            for p in self.model.parameters():
                p.data -= learningrate*p.grad.data

        # return the final output
        return forward(self, all_activfuncs, x_train)


    def applyActivationFunc(x, activfunc):
        if activfunc == "Tanh":
            return nn.Tanh(x)
        elif activfunc == "ReLU":
            return F.relu(x)
        else:
            print("ERROR: The specified activation function " + activ_funcs[k] + " is not implemented.")
            sys.exit(2)

    def createLossFunc(lossfunc):
        if lossfunc == "MSE":
            lossfunc = torch.nn.MSELoss(reduction='mean')

        elif lossfunc == "MAE":
            lossfunc = torch.nn.L1Loss(reduction='mean')

        elif lossfunc == "CrossEntropy":
            lossfunc = torch.nn.CrossEntropyLoss(reduction='mean')

        else:
            print("ERROR: The specified loss function " + lossfunc + " is not implemented.")
            sys.exit(2)

        return lossfunc






