import torch
import os
import sys

from torch.autograd import Variable


import globalvariables as gv 



### cast tensor to the cuda datatype
dtype_cuda = torch.cuda.FloatTensor








# prompt for Computation ID to perform
curr_id = input("Enter the computation ID in the format YYYYMMDD###": )

compdir = gv.COMPDIR + "/" + gv.AUTHORPRAEFIX + "_" + gv.COMPPRAEFIX + curr_id

if not os.path.exists(compdir):
	print("ERROR: The directory " + compdir + " does not exist. Please enter a valid computation ID.")
	sys.exit(2)






print("STATUS: The current computation directory is: " + compdir)


