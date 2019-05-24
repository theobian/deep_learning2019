import torch
from torch.autograd import Variable
from torch.nn import functional as F

import globalvariables as gv
import neuralnetworkfuncs as nnf 
import math






class MLP(torch.nn.Module):
    
    def createLayer(self, curr_subnn, curr_layer):
        #print(self.nnarch.laytyps.laytyp_hls)
        #exit()
        # get layertype
        ## input layer
        if curr_layer == 0:
            prevlay_type = None
            curr_layertype = self.nnarch.laytyps.laytyp_il
            n_inperc=self.nnarch.npercs.nperceptr_il[curr_subnn]
            kernsiz=self.nnarch.convkernsiz.kernsiz_il
            strid=self.nnarch.convstrids.strid_il
            pad=self.nnarch.convpads.pad_il
            prev_dim_xdata = self.dim_xdata_in[curr_subnn]

            if self.n_hidlayers > 0:
                n_outperc = self.nnarch.npercs.nperceptr_hls[curr_subnn,0]
            else:
                n_outperc = self.nnarch.npercs.nperceptr_ol

        ## hidden layers
        elif curr_layer > 0 and curr_layer <= self.n_hidlayers:
            
            curr_hidlay = curr_layer - 1

            #print(str(curr_subnn))
            #print(str(curr_layer))

            n_inperc = self.nnarch.npercs.nperceptr_hls[curr_subnn,curr_hidlay]
            curr_layertype = self.nnarch.laytyps.laytyp_hls[curr_hidlay]
            kernsiz = self.nnarch.convkernsiz.kernsiz_hls[curr_hidlay]
            strid = self.nnarch.convstrids.strid_hls[curr_hidlay]
            pad = self.nnarch.convpads.pad_hls[curr_hidlay]
            prev_dim_xdata = self.dim_xdata[curr_subnn,curr_hidlay]            

            # if first hidden layer, the previous layer is the input layer
            if curr_layer == 1:
                prevlay_kernsiz = self.nnarch.convkernsiz.kernsiz_il
                prevlay_strid = self.nnarch.convstrids.strid_il
                prevlay_type = self.nnarch.laytyps.laytyp_il
                prevlay_pad = self.nnarch.convpads.pad_il
 #               prev_dim_xdata = self.dim_xdata_in

            else:
                prevlay_type = self.nnarch.laytyps.laytyp_hls[curr_hidlay-1]
                prevlay_kernsiz = self.nnarch.convkernsiz.kernsiz_hls[curr_hidlay-1]
                prevlay_strid = self.nnarch.convstrids.strid_hls[curr_hidlay-1]
                prevlay_pad = self.nnarch.convpads.pad_hls[curr_hidlay-1]
     
            
            # if last hidden layer, the next layer is the output layer and the subnetworks need to be merged before that
            if curr_layer == self.n_hidlayers:
                # compute the number of input perceptrons (summing output of subnetworks)
                n_inperc = 0
                for i in range(self.n_subnns):
                    n_inperc += self.nnarch.npercs.nperceptr_hls[i,self.n_hidlayers-1]*self.n_dapackperchan[i]


                n_outperc = self.nnarch.npercs.nperceptr_ol
                self.n_perc_outlayer_all += n_outperc

            else:
                n_outperc = self.nnarch.npercs.nperceptr_hls[curr_subnn,curr_layer]

#        ## output layers
#        elif curr_layer == -1:
#            curr_layertype = self.nnarch.laytyps.laytyp_ol

        else:
            print("ERROR: The requested layer # (" + str(curr_layer) + ") is not valid! Only values between 0 and " + str(self.n_hidlayers) + " (inclusive) are allowed.")
            exit(1)


        # TODO: generalize to allow dilation
        dil=1
    
        # calculate new dimension of x-data due to the effect of padding, dilation, stride and kernel size from the current layer
        self.dim_xdata[curr_subnn,curr_layer] = int(math.floor(1+(prev_dim_xdata + 2*pad - dil*(kernsiz-1)-1)/strid))

#        print("Old input number: " + str(n_inperc))
        if curr_layertype == "Regular1d":
            # if the previous layer is a Convolutional 2D one or it was the input layer, compute the new (1D) length
            if prevlay_type == "Conv2d":
                #print(self.dim_xdata)
                #print(prev_dim_xdata)
                #exit()
                n_newinperc = prev_dim_xdata*prev_dim_xdata*n_inperc
            else:
                n_newinperc = n_inperc
            
            #print(type(n_newinperc))
            #exit()

            return torch.nn.Linear(n_newinperc, n_outperc)
    
        elif curr_layertype == "Conv2d":
    #        print(n_outperc)
    #        exit()
#            print("New input number: " + str(n_inperc))
            return torch.nn.Conv2d(n_inperc, n_outperc, kernel_size=kernsiz, stride=strid, padding=pad)
    
        else:
            print("ERROR: The requested layertype (" + layer_type + ") is not implemented!")
            exit(1)



    # input: 
    #   int[n_subnns] inlay_npercs                      ... # of perceptrons in the input layer for each neural network subunit
    #   int[n_subnns, n_hidlayers] hidlays_npercs       ... # of perceptrons in each hidden layer for each neural network subunit
    #   int outlay_npercs                               ... # of perceptrons in the output layer
    #   int[n_subnns] n_dapackperchan        ... # of datapackages (which are combined prior to the outputlayer) sent through each channel 
    def __init__(self, nnarchitecture, n_datapackages_per_channel, size_data):
        super(MLP, self).__init__()
        self.dim_xdata_in = size_data
        self.nnarch = nnarchitecture
        self.n_dapackperchan = n_datapackages_per_channel
        self.n_subnns, self.n_hidlayers = self.nnarch.npercs.nperceptr_hls.size()
        self.dim_xdata = torch.LongTensor([[0 for x in range(self.n_hidlayers+1)] for y in range(self.n_subnns)])

        self.n_perc_outlayer_all = 0
        #print(self.nnarch.npercs.nperceptr_il)
        #exit()
        # construct neural network structure
        
        ## add input layer
        curr_inlay = []
        for i in range(self.n_subnns):
#                self.dim_xdata[i,0] = size_data
            curr_inlay.append(torch.nn.ModuleList([self.createLayer(curr_subnn = i, curr_layer = 0)]))
            
        self.nns = torch.nn.ModuleList(curr_inlay)


        ## append all except the last hidden layers
        for i in range(self.n_subnns):
            curr_list = []
            for k in range(self.n_hidlayers):
                curr_hidlay = self.createLayer(curr_subnn = i, curr_layer = k+1)
                curr_list.append(curr_hidlay)
                
            self.nns[i].extend(curr_list)


        ## append last hidden layer
        self.nns.append(self.createLayer(0,self.n_hidlayers))




        # construct pooling for each layer (if applicable)
        self.pools = []

        ## input layer
        self.pools.append(self.createPooling(pooling_type=self.nnarch.pooltyp.pooltyp_il, kernsiz=self.nnarch.poolkernsiz.kernsiz_il, strid=self.nnarch.poolstrids.strid_il, pad=self.nnarch.poolpads.pad_il))

        ## hidden layers
        #print(self.nnarch.pooltyp.pooltyp_hls)
        for k in range(self.n_hidlayers):
            self.pools.append(self.createPooling(pooling_type=self.nnarch.pooltyp.pooltyp_hls[k], kernsiz=self.nnarch.poolkernsiz.kernsiz_hls[k], strid=self.nnarch.poolstrids.strid_hls[k], pad=self.nnarch.poolpads.pad_hls[k]))

#        ## output layer
#        self.pools.append(createPooling(pooling_type=self.nnarch.pooltyp.pooltyp_ol, kernsiz=self.nnarch.poolkernsiz.kernsiz_ol, strid=self.nnarch.poolstrids.strid_ol, pad=self.nnarch.poolpads.pad_ol))




    # input: 
    #   Function[n_hidlayers+2] activfuncs                      ... a list of n_hidlayers + 2 of activation functions to be applied to input layer, to each hidden layer and the outputlayer 
    #   FloatTensor[n_subnns, n_datapts, size_datapt] x         ... a list of n_subnns datasets, each passed forward separately through each subunit
    # TODO

    def forward(self, activfuncs, x, weightsharing, auxiliaryloss):

        # check if shape of input is correct
        #TODO: Generalize to a more general method that checks not only the size of the x_data, but also other consistencies such as: 
        #       weightsharing = True <=> n_subnns  = 1
        #       
        if self.isWrongShape(x_data=x, weightsharing=weightsharing, auxiliaryloss=auxiliaryloss):
            print("ERROR: The dimensions of the dataset are incorrect! The dataset has dimensions: " + str(len(x)))
            exit(1) 
        #STATUS: The dataset x has the correct dimensions to be passed to the neural network
        

        n_xdata_packages = len(x)
        x_ = n_xdata_packages*[None]
        if auxiliaryloss:
            y_auxout = n_xdata_packages*[None]
        else:
            y_auxout = [None]


        

        # propagate through layers for each channel       
        for j in range(n_xdata_packages):
            
            ## if weightsharing: propagate all x_data packages through one and the same network, else send each package through separate channel
            if weightsharing:
                i = 0
            else: 
                i = j

            #print(x[j].size())
            #exit()
            # apply input layer
            x_[j] = self.nns[i][0](x[j])
            x_[j] = self.applyActivationFunc(activfuncs.actfunc_inlayer, x_[j])
            
            if self.pools[0] != None:
                x_[j] = self.pools[0](x_[j])
            # apply hidden layers
            prev_ltype = self.nnarch.laytyps.laytyp_il
            
            for k in range(self.n_hidlayers-1):
                # reshape data (with respect to dimensions) if necessary
                #print(x_[j].size())
                x_[j] = nnf.reshapeData(curr_layertype=self.nnarch.laytyps.laytyp_hls[k], prev_layertype=prev_ltype, xdata = x_[j])

                x_[j] = self.nns[i][k+1](x_[j])
                x_[j] = self.applyActivationFunc(activfuncs.actfunc_hidlayers[k], x_[j])
                if self.pools[k+1] != None:
                    x_[j] = self.pools[k+1](x_[j])
                prev_ltype = self.nnarch.laytyps.laytyp_hls[k]

            # generate auxiliary output if auxiliary loss
            if auxiliaryloss:
                y_auxout[j] = self.applyActivationFunc(activfuncs.actfunc_auxoutlayers, x_[j])


        # concatenate into a single 2D tensor if multiple xdata_packages have been used
        # TODO: Put this into a separate method when implementing convolutional networks in order to reduce messiness of code
        #print(x_[0].size())
        if n_xdata_packages > 1:
            x_ = torch.cat(x_,dim=1)
        elif n_xdata_packages == 1:
            x_ = x_[0]
        else:
            print("ERROR: The number of x-data packages must be larger than 0 (" + str(self.n_subnns) + ").")
            exit(1)


        # apply last hidden layer
        x_ = nnf.reshapeData(curr_layertype=self.nnarch.laytyps.laytyp_hls[self.n_hidlayers-1], prev_layertype=prev_ltype, xdata = x_)
        x_ = self.nns[self.n_subnns](x_)
        #print(x_.size())
        x_ = self.applyActivationFunc(activfuncs.actfunc_hidlayers[self.n_hidlayers-1], x_)
        if self.pools[self.n_hidlayers] != None:
            x_ = self.pools[self.n_hidlayers](x_)

        x_ = self.applyActivationFunc(activfuncs.actfunc_outlayer, x_)

        return y_auxout, x_



    # input: 
    #   int[n_subnns] inlay_npercs                      ... 
    def applyActivationFunc(self, activfunc, x):
        if activfunc == "Tanh":
            return nn.Tanh(x)
        elif activfunc == "ReLU":
            return F.relu(x)
        elif activfunc == "SoftMax":
            return F.softmax(x, dim=1)
        elif activfunc == "None":
            return x

        #TODO: implement mor activation functions
        else:
            print("ERROR: The specified activation function (" + activfunc + ") is not implemented.")
            exit(2)


    def isWrongShape(self, x_data, weightsharing, auxiliaryloss):
        #TODO

        return False



    
    
    def createPooling(self, pooling_type, kernsiz, strid, pad):
        # TODO: implement further pooling types
        if pooling_type == "MaxPool2d":
            return torch.nn.MaxPool2d(kernel_size=kernsiz, stride=strid, padding=pad)
        
        elif pooling_type == None:
            return None
        
        else:
            print("ERROR: The requested poolingtype (" + pooling_type + ") is not implemented!")
            exit(1)