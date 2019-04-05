import torch
from torch.autograd import Variable
import os
import sys


import globalvariables as gv
import dlc_practical_prologue as prol 
import MLP as mlp 



###################################### functions ###################################

## Requires 2 Arguments:
#       Argument 1: number
#       Argument 2: N ... number of digits the final number should contain
def generateNDigitNumber(n, N, spacer):
    orignumb2=n
    orignumb=n
    newnumb=""

    # if length is negative or 0, then the original number is returned
    if N <= 0:
        newnumb = N;
    else:
        # determine the number of digits the original number has
        n_digs=0
        while orignumb2 != 0:
            orignumb2 /= 10
            n_digs = n_digs + 1
            
        n_zeros=N-n_digs
    
        for i in range(n_zeros):
            newnumb=newnumb+spacer
        
        newnumb=newnumb+orignumb

    return newnumb;



def printToFile(filename, data, mode):
    curr_file = open(filename, mode)
    curr_file.write(data)
    curr_file.close()





### cast tensor to the cuda datatype
##dtype_cuda = torch.cuda.FloatTensor








# prompt for Computation ID to perform
curr_id = input("Enter the computation ID in the format YYYYMMDD###: ")

compdir = gv.COMPDIR + gv.AUTHORPRAEFIX + "_" + gv.COMPPRAEFIX + curr_id + "/"

if not os.path.exists(compdir):
    print("ERROR: The directory " + compdir + " does not exist. Please enter a valid computation ID.")
    sys.exit(2)


# set computation specific directories & files
curr_ifilesdir = compdir + gv.IFILESDIR


genparfile = curr_ifilesdir + gv.GENPARS_FILENAME
nnparfile = curr_ifilesdir + gv.NNPARS_FILENAME
hlparfile = curr_ifilesdir + gv.HIDDENLAYERPARS_FILENAME
iolparfile = curr_ifilesdir + gv.IOLAYERPARAMETERS_FILENAME
ldparfile = curr_ifilesdir + gv.LOADDATAPARS_FILENAME
ofilenamesfile = curr_ifilesdir + gv.OFILES_FILENAME
cofilenamesfile = curr_ifilesdir + gv.COMMONOFILES_FILENAME



# Read input parameters 
## General Paramters 
genpars = open(genparfile).read()
genpars = genpars.splitlines()
genpars = genpars[1].split()
n_jobsteps = int(genpars[gv.INDEX_NJOBSTEP])
n_parsperlayer = int(genpars[gv.INDEX_NPARSPERLAYER])

##TODO: Set whether running on CPU or GPU thorugh a device variable: 
#device = torch.device('cpu')
# device = torch.device('cuda') # Uncomment this to run on GPU

## Load Input data parameters 
ldpars = open(ldparfile).read()
ldpars = ldpars.splitlines()
ldpars = ldpars[1].split()
sinorpairofdigs = ldpars[gv.INDEX_SINGLEORPAIRSOFDIGITS]
n_datapts = int(ldpars[gv.INDEX_NDATAPOINTS])


## Common O-filenames
cofilenames = open(cofilenamesfile).read()
cofilenames = cofilenames.splitlines()
cofilenames = cofilenames[1].split()
trainandtesterrorfilename = compdir + cofilenames[gv.INDEX_TRAIN_TEST_ERROR]


## Neural Network architecture parameters
nnpars = open(nnparfile).read()
nnpars = nnpars.splitlines()
nnpars = nnpars[1:]
nrows_nnpars = len(nnpars)
### check if correct number of lines

#print(str(nrows_nnpars != n_jobsteps))
if (nrows_nnpars != n_jobsteps):
    print("ERROR: The number of rows (" + str(nrows_nnpars) + ") in file " + genparfile + " is not equal the number of jobsteps (" + str(n_jobsteps) + ").")
    sys.exit(2)

## Hidden layer parameters
hlpars = open(hlparfile).read()
hlpars = hlpars.splitlines()
hlpars = hlpars[1:]
nrows_hlpars = len(hlpars)
### check if correct number of lines 
if nrows_hlpars != n_jobsteps:
    print("ERROR: The number of rows (" + str(nrows_hlpars) + ") in file " + hlparfile + " is not equal the number of jobsteps (" + str(n_jobsteps) + ").")
    sys.exit(2)

## IO layer parameters
iolpars = open(iolparfile).read()
iolpars = iolpars.splitlines()
iolpars = iolpars[1:]
nrows_iolpars = len(iolpars)
### check if correct number of lines 
if nrows_iolpars != n_jobsteps:
    print("ERROR: The number of rows (" + str(nrows_iolpars) + ") in file " + iolparfile + " is not equal the number of jobsteps (" + str(n_jobsteps) + ").")
    sys.exit(2)


## O-Filenames
ofilenames = open(ofilenamesfile).read()
ofilenames = ofilenames.splitlines()
ofilenames = ofilenames[1:]
nrows_ofilenames = len(ofilenames)
### check if correct number of lines 
if nrows_ofilenames != n_jobsteps:
    print("ERROR: The number of rows (" + str(nrows_ofilenames) + ") in file " + ofilenamesfile + " is not equal the number of jobsteps (" + str(n_jobsteps) + ").")
    sys.exit(2)







# Load the dataset 
x_train, y_train, clas_train, x_test, y_test, clas_test = prol.generate_pair_sets(n_datapts)

#TODO: Implementation of loading single digit images (along with all its parameters: flatten, one_hot_labels, ...)



# Perform computation for each jobstep (using the same dataset)
for i in range(n_jobsteps):

    curr_jsid = generateNDigitNumber(i, gv.NDIGITS_JOBSTEPS, gv.JOBSTEPFOLDERSPACER)
    curr_jsdir = compdir + gv.JOBSTEPPRAEFIX + curr_jsid + "/"

    ## create corresponding jobstep-directory if it does not yet exist
    try:
        os.makedirs(curr_jsdir)
    except FileExistsError:
        # directory already exists
        pass
        

    ## read jobstep-specific row of loaded parameter lists (lists from above)
    ### Neural Network architecture parameters
    curr_nnpars = nnpars[i].split()

    #### TODO: Error Checking if correct numbre of parameters

    #### set individual neural network parameters
    curr_nnmodel = nnpars[INDEX_NNMODEL]
    curr_nhidlay = int(nnpars[INDEX_HIDDEN_LAYERS])
    curr_loss = nnpars[INDEX_LOSS]
    curr_learnrate = float(nnpars[INDEX_LEARNINGRATE])
    curr_nepochs = int(nnpars[INDEX_NEPOCHS])

    #### Print for debugging purposes
    # TODO: Print to Computation.out File for future reference 
    print("############################################################")
    print("The current neural network has the following properties:")
    print("Model type: "  + curr_nnmodel)
    print("Number of hidden layers: " + str(curr_nhidlay))
    print("Loss function: " + curr_loss)
    print("Learning rate: " + str(curr_learnrate))
    print("Number of epochs: " + str(curr_nepochs))


    ### Hidden layer parameters
    curr_hlpars = hlpars[i].split()
    #### TODO: Error Checking if correct number of parameters

    #### set individual Hidden layer parameters (one value for each layer)
    curr_nperceptrons = [0]*curr_nhidlay
    curr_activfunc = [""]*curr_nhidlay
    curr_dropout = [0.0]*curr_nhidlay

    for j in range(curr_nhidlay):
        curr_nperceptrons[j] = int(hlpars[j*n_parsperlayer + INDEX_NPERCEPTRONS])
        curr_activfunc[j] = hlpars[j*n_parsperlayer + INDEX_ACTIVFUNC]
        curr_dropout[j] = float(hlpars[j*n_parsperlayer + INDEX_DROPOUT])


    ### IO layer parameters 
    curr_iolpars = iolpars[i].split()
    curr_ninpercept = int(curr_iolpars[gv.INDEX_NINPERCEPTR])
    curr_inactivfunc = curr_iolpars[gv.INDEX_INACTIVFUNC]
    curr_noutpercept = int(curr_iolpars[gv.INDEX_NOUTPERCEPTR])
    curr_outactivfunc = curr_iolpars[gv.INDEX_OUTACTIVFUNC]
    print("")
    print("The current neural network layers have the following properties:")
    print("Input layer: ")
    print("    #perceptrons: "  + str(curr_ninpercept))
    print("    activation function: "  + curr_inactivfunc)
    print("Output layer: ")
    print("    #perceptrons: "  + str(curr_noutpercept))
    print("    activation function: "  + curr_outactivfunc)
    print("Hidden layers: ")
    print("    #perceptrons: "  + str(curr_nperceptrons))
    print("    activation function: "  + curr_activfunc)
    print("    drop out: "  + str(curr_dropout))

    ### Set O-filenames
    curr_ofilenames = ofilenames[i].split()
    curr_ofile_modelparsfilename = curr_jsdir + curr_ofilenames[gv.INDEX_MODELPARS]
    print("")
    print("The output is printed to the following files:")
    print("Model Parameters: " + curr_ofile_modelparsfilename)
    print("Train and Test error: " + trainandtesterrorfilename)

    ## Construct the network
    #TODO: Generalize to allow not only MLP but also Convolutional NN and mixtures of those 
    ###################MLP(curr_nn, curr_ninpercept, curr_nperceptrons, curr_noutpercept)

    ###################all_activfuncs = [curr_inactivfunc] + curr_activfunc + [curr_outactivfunc]

    ## Train the network
    ###################y_train_pred = curr_nn.trainMLP(curr_nn, x_train, y_train, all_activfuncs, curr_loss, curr_learnrate, curr_nepochs)
    ###################train_error = calculateError(y_train_pred, y_train)

    ## Test the network
    ### TODO: Extension to have multiple repititions (using randomly selected inputdata and initial weights)
    ###################y_pred = curr_nn.forward(curr_nn, all_activfuncs, x_test)
    ###################test_error = calculateError(y_test_pred, y_test)

    ## print the train/test errors incl. STDEV, weights and anything else we want to the output directory
    ### model parameters 
    ###################data = curr_nn.model.parameters()
    ###################printToFile(curr_ofile_modelparsfilename, data, 'w')

    ### train and test errors
    #### TODO: Standard deviation print
    ###################printToFile(trainandtesterrorfilename, [train_error, test_error] , 'a')

    print("")
    print("Jobstep " + str(i) + " done!")
    print("############################################################")

