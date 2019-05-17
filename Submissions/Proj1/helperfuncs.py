import sys
import os
import torch
from torch.autograd import Variable
from torch.nn import functional as F


def read2DData(filename, separator, datatype = None):
    data = open(filename).read()
    data = data.splitlines()
    n_lines = len(data)
    for i in range(n_lines):
        temp_1 = data[i].split(separator)
		
        n_datapts = len(temp_1)
		# if the last entry is empty, delete it
        if temp_1[n_datapts-1] == '':
            temp_1 = temp_1[:n_datapts-1]

        data[i] = temp_1
#		data[i] = torch.tensor(temp_1, dtype=datatype)
	
	# if the last line is empty, delete it 
    if len(data[n_lines-1]) == 0:
        data = data[:n_lines-1]
    print(type(data))
    exit()
	#return np.stack(data, axis=0)
    return torch.tensor(data)


## Requires 2 Arguments:
#       Argument 1: number
#       Argument 2: N ... number of digits the final number should contain
def generateNDigitNumber(n, N, spacer):
    orignumb2=n
    orignumb=n
    newnumb=""

    # if length is negative or 0, then the original number is returned
    if N <= 0:
        newnumb = n;
        #print("Entered If")
    else:
        # determine the number of digits the original number has
        n_digs=0
        while orignumb2 != 0:
            orignumb2 /= 10
            orignumb2 = int(orignumb2)
            n_digs = n_digs + 1
        n_zeros=N-n_digs

    
        for i in range(n_zeros):
            newnumb=newnumb+spacer
        
        newnumb=newnumb+str(orignumb)

    return newnumb;


# assumes to take a 1D list as input
def print1DDataToFile(filename, dataset, mode, spacer, alignment="horizontal"):
    curr_file = open(filename, mode)
    if alignment == "horizontal":
        for elem in dataset:
            curr_file.write(str(elem))
            curr_file.write(spacer)
        curr_file.write('\n')

    elif alignment == "vertical":
        for elem in dataset:
            curr_file.write(str(elem))
            curr_file.write('\n')

    else:
        print("ERROR: The requested alignment (" + alignment + ") is not implemented.")

    
    curr_file.close()


# assumes to take a pytorch tensor as input
def print2DDataToFile(filename, dataset, mode, spacer):
    curr_file = open(filename, mode)
    for i in range(len(dataset[:,0])):

        for j in range(len(dataset[i,:])):
            curr_file.write(str(dataset[i,j]))
            curr_file.write(spacer)
        curr_file.write('\n')
    curr_file.write('\n')
    curr_file.close()


def calculateError(y_pred, y_target):
    n_row, n_col = y_pred.data.size()
    dummy, ind_pred = torch.max(y_pred,1)
    dummy, ind_tar = torch.max(y_target,1)

    res = (ind_pred != ind_tar).float()
    res = res.data[:]
    #print(type(res))
    #print(res)
    #exit(0)
    return res.sum()/n_row


#def formatTargetData(y_target):
 #   # produce a Nx2 output with ones in the first column if x <= y and ones in the second column if y > x
 #   temp = (y_target[:] < gv.NUMPREC)
 #   result = torch.stack([y_target[:], temp.long()],1)
 #   return result

def prepare_data_sets(x_train, y_train, clas_train, x_test, y_test, clas_test, normalize=True):
    
    if normalize:
        mu, std = x_train.mean(), x_train.std()
        x_train.sub_(mu).div_(std)
        x_test.sub_(mu).div_(std)
    return x_train, y_train, clas_train, x_test, y_test, clas_test


