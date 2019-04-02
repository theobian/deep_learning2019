import torch
from torch.autograd import Variable
import os
import sys

import globalvariables as gv 



###################################### functions ###################################

## Requires 2 Arguments:
# 		Argument 1: number
#   	Argument 2: N ... number of digits the final number should contain
def generateNDigitNumber(n, N, spacer):
	orignumb2=n
	orignumb=n
	newnumb=""

	# if length is negative or 0, then the original number is returned
	if N <= 0:
		newnumb = N;
	else
		# determine the number of digits the original number has
		n_digs=0
		while orignumb2 != 0:
			orignumb2 /= 10;
			n_digs++;
			
		n_zeros=N-n_digs
	
		for i in range(n_zeros):
			newnumb=newnumb+spacer
		
		newnumb=newnumb+orignumb

	return newnumb;








### cast tensor to the cuda datatype
dtype_cuda = torch.cuda.FloatTensor








# prompt for Computation ID to perform
curr_id = input("Enter the computation ID in the format YYYYMMDD###: ")

compdir = gv.COMPDIR + gv.AUTHORPRAEFIX + "_" + gv.COMPPRAEFIX + curr_id

if not os.path.exists(compdir):
	print("ERROR: The directory " + compdir + " does not exist. Please enter a valid computation ID.")
	sys.exit(2)


# set computation specific directories & files
curr_ifilesdir = compdir + gv.IFILESDIR
#curr_ofilesdir = compdir + gv.OFILESDIR

genparfile = curr_ifilesdir + gv.GENPARS_FILENAME
nnparfile = curr_ifilesdir + gv.NNPARS_FILENAME
hlparfile = curr_ifilesdir + gv.HIDDENLAYERPARS_FILENAME




# Read input
## General Paramters 
genpars = open(genparfile).read.splitlines()
genpars = genpars[1].split()
n_jobsteps = genpars[gv.INDEX_NJOBSTEP]
n_parsperlayer = genpars[gv.INDEX_NPARSPERLAYER]

## Neural Network architecture parameters
nnpars = open(nnparfile).read.splitlines()
nnpars = nnpars[1:]
nrows_nnpars = length(nnpars)
### check if correct number of lines 
if nrows_nnpars != n_jobsteps:
	print("ERROR: The number of rows (" + nrows_nnpars + ") in file " + genparfile + " is not equal the number of jobsteps (" + n_jobsteps + ").")
	sys.exit(2)

## Hidden layer parameters
hlpars = open(hlparfile).read.splitlines()
hlpars = hlpars[1:]
nrows_hlpars = length(hlpars)
### check if correct number of lines 
if nrows_hlpars != n_jobsteps:
	print("ERROR: The number of rows (" + nrows_hlpars + ") in file " + hlparfile + " is not equal the number of jobsteps (" + n_jobsteps + ").")
	sys.exit(2)



# Perform computation for each jobstep
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
	curr_nhidlay = nnpars[INDEX_HIDDEN_LAYERS]
	curr_loss = nnpars[INDEX_LOSS]
	curr_learnrate = nnpars[INDEX_LEARNINGRATE]
	curr_nepochs = nnpars[INDEX_NEPOCHS]

	#### Print for debugging purposes
	print("The current neural network has the following properties:")
	print("Model type: "  + curr_nnmodel)
	print("Number of hidden layers: " + curr_nhidlay)
	print("Loss function: " + curr_loss)
	print("Learning rate: " + curr_learnrate)
	print("Number of epochs: " + curr_nepochs)


	### Hidden layer parameters
    curr_hlpars = hlpars[i].split()


	#### TODO: Error Checking if correct numbre of parameters

	#### set individual Hidden layer parameters (one value for each layer)
	curr_nperceptrons = [0]*curr_nhidlay
	curr_activfunc = [""]*curr_nhidlay
	curr_dropout = [0.0]*curr_nhidlay

	for j in range(curr_nhidlay):
		curr_nperceptrons[j] = hlpars[j*n_parsperlayer + INDEX_NPERCEPTRONS]
		curr_activfunc[j] = hlpars[j*n_parsperlayer + INDEX_ACTIVFUNC]
		curr_dropout[j] = hlpars[j*n_parsperlayer + INDEX_DROPOUT]


	## TODO: Construct the network


	## TODO: Load the data (Decide if we want to load it from 1 directory which is the same for all computations, or individually for each computation)

	## TODO: Train the network

	## TODO: Test the network 

	## TODO: print the train/test errors, weights and anything else we want to the output directory


	print("Jobstep " + i + " done!")
