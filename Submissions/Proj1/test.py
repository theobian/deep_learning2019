import torch
from torch.autograd import Variable
import os
import sys
import copy


import globalvariables as gv
import dlc_practical_prologue as prol
import MLP as mlp
import helperfuncs as hlf 
import neuralnetworkfuncs as nnf
import NeuralNetworkArchitecture as nna 
import Layertypes as lts 

import ActivFuncs as actfunc
import LossFuncs as lossfunc
import NPerceptrons as npercs 
import ConvKernelSizes as cokesi
import ConvPaddings as copa 
import ConvStrides as cost 
import Pooltypes as poty 
import PoolKernelSizes as pokesi 
import PoolPaddings as popa 
import PoolStrides as post 


# the list of computation IDs of computations that are going to be run: 

all_compids = ["20190424001", "20190514001", "20190514002", "20190516003", "20190516004", "20190516005"]
# set directories relative to current path 


# prompt for Computation ID to perform
#curr_id = input("Enter the computation ID in the format YYYYMMDD###: ")
#curr_id = "20190514002"

for cid in all_compids:
    print("STATUS: Computation with ID#: " + cid + " will start now!")

    compdir = gv.COMPDIR + gv.AUTHORPRAEFIX + "_" + gv.COMPPRAEFIX + cid + "/"
    #compdir = gv.COMPDIR + gv.AUTHORPRAEFIX + "_" + gv.COMPPRAEFIX + curr_id + "/"
    
    #if not os.path.exists(compdir):
    #    print("ERROR: The directory " + compdir + " does not exist. Please enter a valid computation ID.")
    #    sys.exit(2)
    
    
    # set computation specific directories & files
    curr_ifilesdir = compdir + gv.IFILESDIR
    
    
    genparfile = curr_ifilesdir + gv.GENPARS_FILENAME
    nnparfile = curr_ifilesdir + gv.NNPARS_FILENAME
    hlparfile = curr_ifilesdir + gv.HIDDENLAYERPARS_FILENAME
    iolparfile = curr_ifilesdir + gv.IOLAYERPARAMETERS_FILENAME
    auxolparfile = curr_ifilesdir + gv.AUXLOSSPARAMETERS_FILENAME
    ldparfile = curr_ifilesdir + gv.LOADDATAPARS_FILENAME
    ofilenamesfile = curr_ifilesdir + gv.OFILES_FILENAME
    cofilenamesfile = curr_ifilesdir + gv.COMMONOFILES_FILENAME
    requoptforachitfile = curr_ifilesdir + gv.REQUOPTFORARCHIT_FILENAME
    
    
    # Read input parameters 
    ## General Paramters 
    genpars = open(genparfile).read()
    genpars = genpars.splitlines()
    genpars = genpars[1].split()
    n_jobsteps = int(genpars[gv.COL_NJOBSTEP])
    n_repititions = int(genpars[gv.COL_NREPITTIONS])
    n_parsperreg1dlayer = int(genpars[gv.COL_NPARSPERREG1DLAYER])
    n_parsperconv2dlayer = int(genpars[gv.COL_NPARSPERCONV2DLAYER])
    benchmarktype = genpars[gv.COL_BENCHMARKINGTYPE]
    
    
    ## Requested different options for the NN architectures
    requops = open(requoptforachitfile).read()
    requops = requops.splitlines()
    requops = requops[0].split(gv.STRSEP_REQUOPTFORARCHIT)
    
    
    ## Input data parameters 
    ldpars = open(ldparfile).read()
    ldpars = ldpars.splitlines()
    ldpars = ldpars[1].split()
    sinorpairofdigs = ldpars[gv.COL_SINGLEORPAIRSOFDIGITS]
    n_datapts = int(ldpars[gv.COL_NDATAPOINTS])
    
    
    ## Common O-filenames
    cofilenames = open(cofilenamesfile).read()
    cofilenames = cofilenames.splitlines()
    cofilenames = cofilenames[1].split(gv.STRSEP_COOFILENAMESFILE)
    ofilename_alljs_trainerr_1chan = compdir + cofilenames[gv.COL_ALLJS_TRAINERR_1CHAN]
    ofilename_alljs_testerr_1chan = compdir + cofilenames[gv.COL_ALLJS_TESTERR_1CHAN]
    ofilename_alljs_trainloss_1chan = compdir + cofilenames[gv.COL_ALLJS_TRAINLOSS_1CHAN]
    ofilename_alljs_avgtrainerr_1chan = compdir + cofilenames[gv.COL_ALLJS_AVGTRAINERR_1CHAN]
    ofilename_alljs_avgtesterr_1chan = compdir + cofilenames[gv.COL_ALLJS_AVGTESTERR_1CHAN]
    ofilename_alljs_trainerr_2chan = compdir + cofilenames[gv.COL_ALLJS_TRAINERR_2CHAN]
    ofilename_alljs_testerr_2chan = compdir + cofilenames[gv.COL_ALLJS_TESTERR_2CHAN]
    ofilename_alljs_trainloss_2chan = compdir + cofilenames[gv.COL_ALLJS_TRAINLOSS_2CHAN]
    ofilename_alljs_avgtrainerr_2chan = compdir + cofilenames[gv.COL_ALLJS_AVGTRAINERR_2CHAN]
    ofilename_alljs_avgtesterr_2chan = compdir + cofilenames[gv.COL_ALLJS_AVGTESTERR_2CHAN]
    ofilename_alljs_trainerr_2chan_wshar = compdir + cofilenames[gv.COL_ALLJS_TRAINERR_2CHAN_WSHAR]
    ofilename_alljs_testerr_2chan_wshar = compdir + cofilenames[gv.COL_ALLJS_TESTERR_2CHAN_WSHAR]
    ofilename_alljs_trainloss_2chan_wshar = compdir + cofilenames[gv.COL_ALLJS_TRAINLOSS_2CHAN_WSHAR]
    ofilename_alljs_avgtrainerr_2chan_wshar = compdir + cofilenames[gv.COL_ALLJS_AVGTRAINERR_2CHAN_WSHAR]
    ofilename_alljs_avgtesterr_2chan_wshar = compdir + cofilenames[gv.COL_ALLJS_AVGTESTERR_2CHAN_WSHAR]
    ofilename_alljs_trainerr_2chan_auxloss = compdir + cofilenames[gv.COL_ALLJS_TRAINERR_2CHAN_AUXLOSS]
    ofilename_alljs_testerr_2chan_auxloss = compdir + cofilenames[gv.COL_ALLJS_TESTERR_2CHAN_AUXLOSS]
    ofilename_alljs_trainloss_2chan_auxloss = compdir + cofilenames[gv.COL_ALLJS_TRAINLOSS_2CHAN_AUXLOSS]
    ofilename_alljs_avgtrainerr_2chan_auxloss = compdir + cofilenames[gv.COL_ALLJS_AVGTRAINERR_2CHAN_AUXLOSS]
    ofilename_alljs_avgtesterr_2chan_auxloss = compdir + cofilenames[gv.COL_ALLJS_AVGTESTERR_2CHAN_AUXLOSS]
    ofilename_alljs_trainerr_2chan_wshar_auxloss = compdir + cofilenames[gv.COL_ALLJS_TRAINERR_2CHAN_WSHAR_AUXLOSS]
    ofilename_alljs_testerr_2chan_wshar_auxloss = compdir + cofilenames[gv.COL_ALLJS_TESTERR_2CHAN_WSHAR_AUXLOSS]
    ofilename_alljs_trainloss_2chan_wshar_auxloss = compdir + cofilenames[gv.COL_ALLJS_TRAINLOSS_2CHAN_WSHAR_AUXLOSS]
    ofilename_alljs_avgtrainerr_2chan_wshar_auxloss = compdir + cofilenames[gv.COL_ALLJS_AVGTRAINERR_2CHAN_WSHAR_AUXLOSS]
    ofilename_alljs_avgtesterr_2chan_wshar_auxloss = compdir + cofilenames[gv.COL_ALLJS_AVGTESTERR_2CHAN_WSHAR_AUXLOSS]
    
    
    ## Neural Network architecture parameters
    nnpars = open(nnparfile).read()
    nnpars = nnpars.splitlines()
    nnpars = nnpars[1:]
    nrows_nnpars = len(nnpars)
    
    ### check if correct number of lines
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
    
    
    ## Auxiliary Outputlayerparameters
    auxolpars = open(auxolparfile).read()
    auxolpars = auxolpars.splitlines()
    auxolpars = auxolpars[1:]
    nrows_auxolpars = len(auxolpars)
    
    ### check if correct number of lines 
    if nrows_auxolpars != n_jobsteps:
        print("ERROR: The number of rows (" + str(nrows_auxolpars) + ") in file " + auxolparfile + " is not equal the number of jobsteps (" + str(n_jobsteps) + ").")
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
    
    
    
    # Perform computation for each jobstep (using the same dataset)
    for i in range(n_jobsteps):
    
        curr_js = i+1
        curr_jsid = hlf.generateNDigitNumber(curr_js, gv.NDIGITS_JOBSTEPS, gv.JOBSTEPFOLDERSPACER)
        curr_jsdir = compdir + gv.JOBSTEPPRAEFIX + curr_jsid + "/"
    
        ## create corresponding jobstep-directory if it does not yet exist
        try:
            os.makedirs(curr_jsdir)
        except FileExistsError:
            # directory already exists
            pass
            
    
        ## read jobstep-specific row of loaded parameter lists (lists from above)
        ### Neural Network architecture parameters
        curr_nnpars = nnpars[i].split(gv.STRSEP_NNPARSFILE)
        #### TODO: Error Checking if correct numbre of parameters
        #### set individual neural network parameters
        curr_nhidlay = int(curr_nnpars[gv.COL_HIDDEN_LAYERS])
        curr_loss = curr_nnpars[gv.COL_LOSS]
        curr_optimizer = curr_nnpars[gv.COL_OPTIMZER]
        curr_learnrate = float(curr_nnpars[gv.COL_LEARNINGRATE])
        curr_nepochs = int(curr_nnpars[gv.COL_NEPOCHS])
        curr_nminibatchsize = int(curr_nnpars[gv.COL_NMINIBATCHSIZE])
    
    
    
        #### Print for debugging purposes
        # TODO: Print to Computation.out File for future reference 
        print("############################################################")
        print("The current neural network has the following properties:")
    #    print("Model type: "  + curr_nnmodel)
        print("Number of hidden layers: " + str(curr_nhidlay))
        print("Loss function: " + curr_loss)
        print("Current optimizer: " + curr_optimizer)
        print("Learning rate: " + str(curr_learnrate))
        print("Number of epochs: " + str(curr_nepochs))
        print("Minibatchsize: " + str(curr_nminibatchsize))
    
    
        ### Hidden layer parameters
        curr_hlpars = hlpars[i].split(gv.STRSEP_HIDDENLAYERPARSFILE)
        #### TODO: Error Checking if correct number of parameters
    
        #### set individual Hidden layer parameters (one value for each layer)     
        curr_hl_types = []
        curr_hl_nperceptrons = []
        curr_hl_activfuncs = []
        curr_hl_dropout = []
        curr_hl_convstride = []
        curr_hl_convkernelsize = []
        curr_hl_convpadding = []
        curr_hl_pooltype = []
        curr_hl_poolstride = []
        curr_hl_poolkernelsize = []
        curr_hl_poolpadding = []
        counter = 0
        for j in range(curr_nhidlay):
            curr_hl_types.append(curr_hlpars[counter + gv.COL_HL_TYPES])
            
            #### properties common to all layer types
            
            curr_hl_nperceptrons.append(int(curr_hlpars[counter + gv.COL_HL_NPERCEPTRONS]))
            curr_hl_activfuncs.append(curr_hlpars[counter + gv.COL_HL_ACTIVFUNCS])
    ##        curr_hl_dropout.append(float(curr_hlpars[counter + gv.COL_HL_DROPOUTS]))
    
            #### properties specific to individual layer types
            #print(str(counter))
            if curr_hl_types[j] == "Regular1d":
                curr_hl_convstride.append(1)
                curr_hl_convkernelsize.append(1)
                curr_hl_convpadding.append(0)
                curr_hl_pooltype.append(None)
                curr_hl_poolstride.append(None)
                curr_hl_poolkernelsize.append(None)
                curr_hl_poolpadding.append(None)
    
    #            curr_hl_convstride.append(None)
    #            curr_hl_convkernelsize.append(None)
    #            curr_hl_convpadding.append(None)
    #            curr_hl_pooltype.append(None)
    #            curr_hl_poolstride.append(None)
    #            curr_hl_poolkernelsize.append(None)
    #            curr_hl_poolpadding.append(None)
                counter = counter + n_parsperreg1dlayer
            
            elif curr_hl_types[j] == "Conv2d":
                curr_hl_convstride.append(int(curr_hlpars[counter + gv.COL_HL_CONVSTRIDES]))
                curr_hl_convkernelsize.append(int(curr_hlpars[counter + gv.COL_HL_CONVKERNELSIZES]))
                curr_hl_convpadding.append(int(curr_hlpars[counter + gv.COL_HL_CONVPADDINGS]))
                curr_hl_pooltype.append(curr_hlpars[counter + gv.COL_HL_POOLTYPES])
                curr_hl_poolstride.append(int(curr_hlpars[counter + gv.COL_HL_POOLSTRIDES]))
                curr_hl_poolkernelsize.append(int(curr_hlpars[counter + gv.COL_HL_POOLKERNELSIZES]))
                curr_hl_poolpadding.append(int(curr_hlpars[counter + gv.COL_HL_POOLPADDINGS]))
                counter = counter + n_parsperconv2dlayer
    
            else:
                print("ERROR: The selected layertype (" + curr_hl_types[j] + ") is not implemented! ")
                exit(1)
    
    
        curr_hl_nperceptrons = torch.LongTensor(curr_hl_nperceptrons)
    
        ### IO layer parameters (No dropout for IO layers (eventually generalize that))
        curr_iolpars = iolpars[i].split(gv.STRSEP_IOLAYERPARSFILE)
        
        #### input layer
        ##### properties common to all layer types
        curr_il_type = curr_iolpars[gv.COL_IL_TYPE]
        curr_il_nperceptrons = int(curr_iolpars[gv.COL_IL_NPERCEPTR])
        curr_il_activfunc = curr_iolpars[gv.COL_IL_ACTIVFUNC]
        
        ##### properties specific to individual layer types
        if curr_il_type == "Regular1d":
            curr_il_convstride = 1
            curr_il_convkernelsize = 1
            curr_il_convpadding = 0
            curr_il_pooltype = None 
            curr_il_poolstride = None
            curr_il_poolkernelsize = None
            curr_il_poolpadding = None
    
    
    #        curr_il_convstride = None
    #        curr_il_convkernelsize = None
    #        curr_il_convpadding = None
    #        curr_il_pooltype = None 
    #        curr_il_poolstride = None
    #        curr_il_poolkernelsize = None
    #        curr_il_poolpadding = None
            
        elif curr_il_type == "Conv2d":
            curr_il_convstride = int(curr_hlpars[gv.COL_IL_CONVSTRIDE])
            curr_il_convkernelsize = int(curr_hlpars[gv.COL_IL_CONVKERNELSIZE])
            curr_il_convpadding = int(curr_hlpars[gv.COL_IL_CONVPADDING])
            curr_il_pooltype = curr_hlpars[gv.COL_IL_POOLTYPE]
            curr_il_poolstride = int(curr_hlpars[gv.COL_IL_POOLSTRIDE])
            curr_il_poolkernelsize = int(curr_hlpars[gv.COL_IL_POOLKERNELSIZE])
            curr_il_poolpadding = int(curr_hlpars[gv.COL_IL_POOLPADDING])
    
        else:
            print("ERROR: The selected layertype (" + curr_il_type + ") is not implemented! ")
            exit(1)
    
        #### output layer
        ##### properties common to all layer types
        curr_ol_type = curr_iolpars[gv.COL_OL_TYPE]
        curr_ol_nperceptrons = int(curr_iolpars[gv.COL_OL_NPERCEPTR])
        curr_ol_activfunc = curr_iolpars[gv.COL_OL_ACTIVFUNC]
        
        ##### properties specific to individual layer types
        if curr_ol_type == "Regular1d":
            curr_ol_convstride = 1
            curr_ol_convkernelsize = 1
            curr_ol_convpadding = 0
            curr_ol_pooltype = None 
            curr_ol_poolstride = None
            curr_ol_poolkernelsize = None
            curr_ol_poolpadding = None
    
    #        curr_ol_convstride = None
    #        curr_ol_convkernelsize = None
    #        curr_ol_convpadding = None
    #        curr_ol_pooltype = None 
    #        curr_ol_poolstride = None
    #        curr_ol_poolkernelsize = None
    #        curr_ol_poolpadding = None
            
        elif curr_ol_type == "Conv2d":
            curr_ol_convstride = int(curr_hlpars[gv.COL_OL_CONVSTRIDE])
            curr_ol_convkernelsize = int(curr_hlpars[gv.COL_OL_CONVKERNELSIZE])
            curr_ol_convpadding = int(curr_hlpars[gv.COL_OL_CONVPADDING])
            curr_ol_pooltype = int(curr_hlpars[gv.COL_OL_POOLTYPE])
            curr_ol_poolstride = int(curr_hlpars[gv.COL_OL_POOLSTRIDE])
            curr_ol_poolkernelsize = int(curr_hlpars[gv.COL_OL_POOLKERNELSIZE])
            curr_ol_poolpadding = int(curr_hlpars[gv.COL_OL_POOLPADDING])
    
        else:
            print("ERROR: The selected layertype (" + curr_ol_types + ") is not implemented! ")
            exit(1)
    
    
        print("")
        print("The current neural network layers have the following properties:")
        print("Input layer: ")
        print("    Layer type: " + curr_il_type)
        print("    #perceptrons: "  + str(curr_il_nperceptrons))
        print("    activation function: "  + curr_il_activfunc)
        print("Output layer: ")
        print("    Layer type: " + curr_ol_type)
        print("    #perceptrons: "  + str(curr_ol_nperceptrons))
        print("    activation function: "  + curr_ol_activfunc)
        print("Hidden layers: ")
        print("    #perceptrons: "  + str(curr_hl_nperceptrons))
        print("    activation function: "  + str(curr_hl_activfuncs))
    #    print("    drop out: "  + str(curr_hl_dropout))
    
    
        ### Auxiliary Output layer parameters
        curr_auxloss_pars = auxolpars[i].split(gv.STRSEP_AUXOLAYERPARSFILE)
        curr_auxloss_activfunc = curr_auxloss_pars[gv.COL_AUXLOSS_ACTIVFUNC]
        curr_auxloss_loss = curr_auxloss_pars[gv.COL_AUXLOSS_LOSS]
        curr_auxloss_weight = float(curr_auxloss_pars[gv.COL_AUXLOSS_WEIGHT])
    
    
        
        
    
        
        ### store information of architecture of NN as well as activation functions and loss functions each in separate objects
        #### activation & loss functions
        curr_activfuncs = actfunc.ActivFuncs(activfunc_inlay=curr_il_activfunc, activfunc_hidlays=curr_hl_activfuncs, activfunc_outlay=curr_ol_activfunc, activfunc_auxoutlays=curr_auxloss_activfunc)
        curr_lossfuncs = lossfunc.LossFuncs(loss_outlayer=curr_loss, loss_aux=curr_auxloss_loss, weight_loss_aux=curr_auxloss_weight)
        
        #### layers info
        curr_layertypes = lts.Layertypes(layertype_inlay=curr_il_type, layertype_hidlays=curr_hl_types, layertype_outlay=curr_ol_type)
        curr_convkernelsizes = cokesi.ConvKernelSizes(kernelsize_inlay=curr_il_convkernelsize, kernelsize_hidlays=curr_hl_convkernelsize, kernelsize_outlay=curr_ol_convkernelsize) 
        curr_convpaddings = copa.ConvPaddings(padding_inlay=curr_il_convpadding, padding_hidlays=curr_hl_convpadding, padding_outlay=curr_ol_convpadding)
        curr_convstrides = cost.ConvStrides(stride_inlay=curr_il_convstride, stride_hidlays=curr_hl_convstride, stride_outlay=curr_ol_convstride)
        
        #### Pooling info
        curr_pooltypes = poty.Pooltypes(pooltype_inlay=curr_il_pooltype, pooltype_hidlays=curr_hl_pooltype, pooltype_outlay=curr_ol_pooltype)
        curr_poolkernelsizes = cokesi.ConvKernelSizes(kernelsize_inlay=curr_il_poolkernelsize, kernelsize_hidlays=curr_hl_poolkernelsize, kernelsize_outlay=curr_ol_poolkernelsize) 
        curr_poolpaddings = copa.ConvPaddings(padding_inlay=curr_il_poolpadding, padding_hidlays=curr_hl_poolpadding, padding_outlay=curr_ol_poolpadding)
        curr_poolstrides = cost.ConvStrides(stride_inlay=curr_il_poolstride, stride_hidlays=curr_hl_poolstride, stride_outlay=curr_ol_poolstride)
    
    
    
        ## Set O-filenames
        curr_ofilenames = ofilenames[i].split(gv.STRSEP_OFILENAMESFILE)
        curr_ofile_modelparsfilename = curr_jsdir + curr_ofilenames[gv.COL_MODELPARS]
    
    
        if benchmarktype == "SCAN_NEPOCHS":
            n_rows = curr_nepochs
            
            ### additional o-filenames
            curr_ofile_trainerrvseps_1chan = curr_jsdir + curr_ofilenames[gv.COL_TRAINERR_EPOCHS_1CHAN]
            curr_ofile_testerrvseps_1chan = curr_jsdir + curr_ofilenames[gv.COL_TESTERR_EPOCHS_1CHAN]
            curr_ofile_trainlossvseps_1chan = curr_jsdir + curr_ofilenames[gv.COL_TRAINLOSS_EPOCHS_1CHAN]
            curr_ofile_trainerrvseps_2chan = curr_jsdir + curr_ofilenames[gv.COL_TRAINERR_EPOCHS_2CHAN]
            curr_ofile_testerrvseps_2chan = curr_jsdir + curr_ofilenames[gv.COL_TESTERR_EPOCHS_2CHAN]
            curr_ofile_trainlossvseps_2chan = curr_jsdir + curr_ofilenames[gv.COL_TRAINLOSS_EPOCHS_2CHAN]
            curr_ofile_trainerrvseps_2chan_wshar = curr_jsdir + curr_ofilenames[gv.COL_TRAINERR_EPOCHS_2CHAN_WSHAR]
            curr_ofile_testerrvseps_2chan_wshar = curr_jsdir + curr_ofilenames[gv.COL_TESTERR_EPOCHS_2CHAN_WSHAR]
            curr_ofile_trainlossvseps_2chan_wshar = curr_jsdir + curr_ofilenames[gv.COL_TRAINLOSS_EPOCHS_2CHAN_WSHAR]
            curr_ofile_trainerrvseps_2chan_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TRAINERR_EPOCHS_2CHAN_AUXLOSS]
            curr_ofile_testerrvseps_2chan_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TESTERR_EPOCHS_2CHAN_AUXLOSS]
            curr_ofile_trainlossvseps_2chan_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TRAINLOSS_EPOCHS_2CHAN_AUXLOSS]
            curr_ofile_trainerrvseps_2chan_wshar_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TRAINERR_EPOCHS_2CHAN_WSHAR_AUXLOSS]
            curr_ofile_testerrvseps_2chan_wshar_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TESTERR_EPOCHS_2CHAN_WSHAR_AUXLOSS]
            curr_ofile_trainlossvseps_2chan_wshar_auxloss = curr_jsdir + curr_ofilenames[gv.COL_TRAINLOSS_EPOCHS_2CHAN_WSHAR_AUXLOSS]
    
        elif benchmarktype == "SCAN_OTHER":
            n_rows = 1
    
        else:
            print("ERROR: The requested benchmarktype (" + benchmarktype + ") is not implemented!")
            exit(2)
    
    
        train_loss_1chan = torch.zeros([n_rows, n_repititions])
        train_error_1chan = torch.zeros([n_rows, n_repititions])
        test_error_1chan = torch.zeros([n_rows, n_repititions])
        train_loss_2chan = torch.zeros([n_rows, n_repititions])
        train_error_2chan = torch.zeros([n_rows, n_repititions])
        test_error_2chan = torch.zeros([n_rows, n_repititions])
        train_loss_2chan_wshar = torch.zeros([n_rows, n_repititions])
        train_error_2chan_wshar = torch.zeros([n_rows, n_repititions])
        test_error_2chan_wshar = torch.zeros([n_rows, n_repititions])
        train_loss_2chan_auxloss = torch.zeros([n_rows, n_repititions])
        train_error_2chan_auxloss = torch.zeros([n_rows, n_repititions])
        test_error_2chan_auxloss = torch.zeros([n_rows, n_repititions])
        train_loss_2chan_wshar_auxloss = torch.zeros([n_rows, n_repititions])
        train_error_2chan_wshar_auxloss = torch.zeros([n_rows, n_repititions])
        test_error_2chan_wshar_auxloss = torch.zeros([n_rows, n_repititions])
    
        # prepare variables to store the neural network architecture for the cases where the two digit-images are treated separately: once with and once without weightsharing
        curr_il_nperceptrons_1dig = torch.LongTensor([int(curr_il_nperceptrons/2)])
        curr_il_nperceptrons_2digs = torch.LongTensor([curr_il_nperceptrons_1dig[0], curr_il_nperceptrons_1dig[0]])
        #curr_il_nperceptrons_2digs = torch.stack(curr_il_nperceptrons_2digs)
    
        if curr_nhidlay > 0:
            curr_hl_nperceptrons_1dig = curr_hl_nperceptrons/2
            curr_hl_nperceptrons_2digs = [curr_hl_nperceptrons_1dig, curr_hl_nperceptrons_1dig]
            curr_hl_nperceptrons_2digs = torch.stack(curr_hl_nperceptrons_2digs)
        else:
            #TODO: convert to tensors
            curr_hl_nperceptrons_1dig = []
            curr_hl_nperceptrons_2digs = [[],[]]
        
        #print(curr_hl_nperceptrons_2digs)
        #exit(0)
        curr_hl_nperceptrons = curr_hl_nperceptrons.unsqueeze(0)
        curr_hl_nperceptrons_1dig = [curr_hl_nperceptrons_1dig]
        curr_hl_nperceptrons_1dig = torch.stack(curr_hl_nperceptrons_1dig)
        #curr_hl_nperceptrons_2digs.unsqueeze(0).repeat(0,2)
        #print(requops)
        #exit(0)
    
        # repeat several times in order to obtain an estimate of the training/test error
        for j in range(n_repititions):
    
    
            ## load the dataset
            x_train, y_train, clas_train, x_test, y_test, clas_test = prol.generate_pair_sets(n_datapts)
    
            # prepare the dataset (normalize, ....)
            x_train, y_train, clas_train, x_test, y_test, clas_test = hlf.prepare_data_sets(x_train, y_train, clas_train, x_test, y_test, clas_test)
    
            curr_size_xdata = x_train.size(3)
    #        print(curr_size_xdata)
    #        exit()
            # modify structure and dimensions to be applicable for NN
            ## auxiliary y-training data
            clas_train = [clas_train[:,0], clas_train[:,1]]
    
            ## x-training data for two-channel structured NNs depending on whether input layer is a 2D convolutional one or not 
            if curr_layertypes.laytyp_il == "Conv2d":
                x_train_2chan = [x_train[:,0,:,:].unsqueeze(1), x_train[:,1,:,:].unsqueeze(1)]
                x_test_2chan = [x_test[:,0,:,:].unsqueeze(1), x_test[:,1,:,:].unsqueeze(1)]
            elif curr_layertypes.laytyp_il == "Regular1d":
                x_train_2chan = [x_train[:,0,:,:].contiguous().view(n_datapts,-1), x_train[:,1,:,:].contiguous().view(n_datapts,-1)]
                x_test_2chan = [x_test[:,0,:,:].contiguous().view(n_datapts,-1), x_test[:,1,:,:].contiguous().view(n_datapts,-1)]
            else:
                print("ERROR: The layertype requested for the inputlayer (" + curr_layertypes.laytyp_il + ") is not implemented!")
                exit(1)
    
            ## Construct the network      
            ### one network for the two digit-images <=> no weightsharing, no auxiliary loss
            if "Default" in requops: 
                curr_single_nn_npercs = npercs.NPerceptrons(torch.LongTensor([curr_il_nperceptrons]), curr_hl_nperceptrons, curr_ol_nperceptrons)
                curr_single_nn_nnarch = nna.NeuralNetworkArchitecture(layertypes=curr_layertypes, nperceptrs=curr_single_nn_npercs, convkernelsizes=curr_convkernelsizes, convpaddings=curr_convpaddings, convstrides=curr_convstrides, \
                    pooltype=curr_pooltypes, poolkernelsizes=curr_poolkernelsizes, poolpaddings=curr_poolpaddings, poolstrides=curr_poolstrides)    
                curr_single_nn = mlp.MLP(nnarchitecture=curr_single_nn_nnarch, n_datapackages_per_channel=[1], size_data=[curr_size_xdata])
    
            ### two (independent) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output: no weightsharing, no auxiliary loss 
            if "Default" in requops or "Default2ChanOnly" in requops:
                curr_two_indep_nn_npercs = npercs.NPerceptrons(curr_il_nperceptrons_2digs, curr_hl_nperceptrons_2digs, curr_ol_nperceptrons)
                curr_two_indep_nn_nnarch = nna.NeuralNetworkArchitecture(layertypes=curr_layertypes, nperceptrs=curr_two_indep_nn_npercs, convkernelsizes=curr_convkernelsizes, convpaddings=curr_convpaddings, convstrides=curr_convstrides, \
                    pooltype=curr_pooltypes, poolkernelsizes=curr_poolkernelsizes, poolpaddings=curr_poolpaddings, poolstrides=curr_poolstrides)
                curr_two_indep_nn = mlp.MLP(nnarchitecture=curr_two_indep_nn_nnarch, n_datapackages_per_channel=[1, 1], size_data=[curr_size_xdata, curr_size_xdata])
            
            ### two (weight-sharing) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output: weightsharing
            if "WeightSharing" in requops:
                curr_two_wshare_nn_npercs = npercs.NPerceptrons(curr_il_nperceptrons_1dig, curr_hl_nperceptrons_1dig, curr_ol_nperceptrons)
                curr_two_wshare_nn_nnarch = nna.NeuralNetworkArchitecture(layertypes=curr_layertypes, nperceptrs=curr_two_wshare_nn_npercs, convkernelsizes=curr_convkernelsizes, convpaddings=curr_convpaddings, convstrides=curr_convstrides, \
                    pooltype=curr_pooltypes, poolkernelsizes=curr_poolkernelsizes, poolpaddings=curr_poolpaddings, poolstrides=curr_poolstrides)
                curr_two_wshare_nn = mlp.MLP(nnarchitecture=curr_two_wshare_nn_nnarch, n_datapackages_per_channel=[2], size_data=[curr_size_xdata])
    
            ### two (independent) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output: no weightsharing, auxiliary loss
            if "AuxiliaryLoss" in requops:
                curr_two_indep_nn_npercs_aux = npercs.NPerceptrons(curr_il_nperceptrons_2digs, curr_hl_nperceptrons_2digs, curr_ol_nperceptrons)
                curr_two_indep_nn_nnarch_aux = nna.NeuralNetworkArchitecture(layertypes=curr_layertypes, nperceptrs=curr_two_indep_nn_npercs_aux, convkernelsizes=curr_convkernelsizes, convpaddings=curr_convpaddings, convstrides=curr_convstrides, \
                    pooltype=curr_pooltypes, poolkernelsizes=curr_poolkernelsizes, poolpaddings=curr_poolpaddings, poolstrides=curr_poolstrides)
                curr_two_indep_nn_aux = mlp.MLP(nnarchitecture=curr_two_indep_nn_nnarch_aux, n_datapackages_per_channel=[1, 1], size_data=[curr_size_xdata, curr_size_xdata])
    
            ### two (weight-sharing) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output: weightsharing & auxiliary loss
            if "WeightSharingAndAuxiliaryLoss" in requops:
                curr_two_wshare_nn_npercs_aux = npercs.NPerceptrons(curr_il_nperceptrons_1dig, curr_hl_nperceptrons_1dig, curr_ol_nperceptrons)
                curr_two_wshare_nn_nnarch_aux = nna.NeuralNetworkArchitecture(layertypes=curr_layertypes, nperceptrs=curr_two_wshare_nn_npercs_aux, convkernelsizes=curr_convkernelsizes, convpaddings=curr_convpaddings, convstrides=curr_convstrides, \
                    pooltype=curr_pooltypes, poolkernelsizes=curr_poolkernelsizes, poolpaddings=curr_poolpaddings, poolstrides=curr_poolstrides)
                curr_two_wshare_nn_aux = mlp.MLP(nnarchitecture=curr_two_wshare_nn_nnarch_aux, n_datapackages_per_channel=[2], size_data=[curr_size_xdata])
    
    
            ## Train and test the networks
            ### one network for the two digit-images <=> no weightsharing, no auxiliary loss
            if "Default" in requops:
                test_error_1chan[:,j], train_loss_1chan[:,j], train_error_1chan[:,j] = nnf.trainAndTestMLP(curr_single_nn, [x_test.view(n_datapts,-1)], y_test, [x_train.view(n_datapts,-1)], y_train, clas_train, curr_activfuncs, curr_lossfuncs, curr_optimizer, curr_learnrate, curr_nepochs, curr_nminibatchsize, benchmarktype)
                print("STATUS: Training and Testing of 1-channel NN done!")
    
            ### two (independent) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output <=> no weightsharing, no auxiliary loss
            if "Default" in requops or "Default2ChanOnly" in requops: 
                test_error_2chan[:,j], train_loss_2chan[:,j], train_error_2chan[:,j] = nnf.trainAndTestMLP(curr_two_indep_nn, x_test_2chan, y_test, x_train_2chan, y_train, clas_train, curr_activfuncs, curr_lossfuncs, curr_optimizer, curr_learnrate, curr_nepochs, curr_nminibatchsize, benchmarktype)
                print("STATUS: Training and Testing of 2-channel NN done!")
    
            ### two (weight-sharing) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output <=> weightsharing however no auxiliary loss
            if "WeightSharing" in requops:
                test_error_2chan_wshar[:,j], train_loss_2chan_wshar[:,j], train_error_2chan_wshar[:,j] = nnf.trainAndTestMLP(curr_two_wshare_nn, x_test_2chan, y_test, x_train_2chan, y_train, clas_train, curr_activfuncs, curr_lossfuncs, curr_optimizer, curr_learnrate, curr_nepochs, curr_nminibatchsize, benchmarktype, weightsharing=True)
                print("STATUS: Training and Testing of 2-channel NN with weightsharing done!")
    
            ### two (independent) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output using auxiliary loss <=> no weightsharing however auxiliary loss
            if "AuxiliaryLoss" in requops:
                test_error_2chan_auxloss[:,j], train_loss_2chan_auxloss[:,j], train_error_2chan_auxloss[:,j] = nnf.trainAndTestMLP(curr_two_indep_nn_aux, x_test_2chan, y_test, x_train_2chan, y_train, clas_train, curr_activfuncs, curr_lossfuncs, curr_optimizer, curr_learnrate, curr_nepochs, curr_nminibatchsize, benchmarktype, auxiliaryloss=True)
                print("STATUS: Training and Testing of 2-channel NN with auxiliary loss done!")
    
            ### two (weight-sharing) networks of the same architecture - one for each digit-image, combined at the outputlayer to give 1 output using auxiliary loss <=> weightsharing & auxiliary loss
            if "WeightSharingAndAuxiliaryLoss" in requops:
                test_error_2chan_wshar_auxloss[:,j], train_loss_2chan_wshar_auxloss[:,j], train_error_2chan_wshar_auxloss[:,j] = nnf.trainAndTestMLP(curr_two_wshare_nn_aux, x_test_2chan, y_test, x_train_2chan, y_train, clas_train, curr_activfuncs, curr_lossfuncs, curr_optimizer, curr_learnrate, curr_nepochs, curr_nminibatchsize, benchmarktype, weightsharing=True, auxiliaryloss=True)
                print("STATUS: Training and Testing of 2-channel NN with weightsharing and auxiliary loss done!")
            
    
    
        if "Default" in requops:
            ## print values for each repetition separately to file
            ### train errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainerr_1chan, train_error_1chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
            hlf.print1DDataToFile(ofilename_alljs_trainerr_2chan, train_error_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_testerr_1chan, test_error_1chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
            hlf.print1DDataToFile(ofilename_alljs_testerr_2chan, test_error_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### train losses of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainloss_1chan, train_loss_1chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
            hlf.print1DDataToFile(ofilename_alljs_trainloss_2chan, train_loss_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
    
            ## print averaged values (over the number of repitions) to file
            ### train error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_1chan, [train_error_1chan.mean(1)[n_rows-1], train_error_1chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_2chan, [train_error_2chan.mean(1)[n_rows-1], train_error_2chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_1chan, [test_error_1chan.mean(1)[n_rows-1], test_error_1chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_2chan, [test_error_2chan.mean(1)[n_rows-1], test_error_2chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
        if "Default2ChanOnly" in requops:
            ## print values for each repetition separately to file
            ### train errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainerr_2chan, train_error_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_testerr_2chan, test_error_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### train losses of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainloss_2chan, train_loss_2chan[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
    
            ## print averaged values (over the number of repitions) to file
            ### train error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_2chan, [train_error_2chan.mean(1)[n_rows-1], train_error_2chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_2chan, [test_error_2chan.mean(1)[n_rows-1], test_error_2chan.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
    
        if "WeightSharing" in requops:
            ### train errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainerr_2chan_wshar, train_error_2chan_wshar[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
            
            ### test errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_testerr_2chan_wshar, test_error_2chan_wshar[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
            
            ### train losses of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainloss_2chan_wshar, train_loss_2chan_wshar[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ## print averaged values (over the number of repitions) to file
            ### train error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_2chan_wshar, [train_error_2chan_wshar.mean(1)[n_rows-1], train_error_2chan_wshar.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_2chan_wshar, [test_error_2chan_wshar.mean(1)[n_rows-1], test_error_2chan_wshar.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
        if "AuxiliaryLoss" in requops:
            ### train errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainerr_2chan_auxloss, train_error_2chan_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
        
            ### test errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_testerr_2chan_auxloss, test_error_2chan_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
        
            ### train losses of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainloss_2chan_auxloss, train_loss_2chan_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
        
            ## print averaged values (over the number of repitions) to file
            ### train error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_2chan_auxloss, [train_error_2chan_auxloss.mean(1)[n_rows-1], train_error_2chan_auxloss.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
        
            ### test error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_2chan_auxloss, [test_error_2chan_auxloss.mean(1)[n_rows-1], test_error_2chan_auxloss.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
        if "WeightSharingAndAuxiliaryLoss" in requops:
            ### train errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainerr_2chan_wshar_auxloss, train_error_2chan_wshar_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test errors of last epoch
            hlf.print1DDataToFile(ofilename_alljs_testerr_2chan_wshar_auxloss, test_error_2chan_wshar_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ### train losses of last epoch
            hlf.print1DDataToFile(ofilename_alljs_trainloss_2chan_wshar_auxloss, train_loss_2chan_wshar_auxloss[n_rows-1,:] , 'a', gv.SPACER, alignment="horizontal")
    
            ## print averaged values (over the number of repitions) to file
            ### train error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtrainerr_2chan_wshar_auxloss, [train_error_2chan_wshar_auxloss.mean(1)[n_rows-1], train_error_2chan_wshar_auxloss.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
            ### test error of last epoch
            hlf.print1DDataToFile(ofilename_alljs_avgtesterr_2chan_wshar_auxloss, [test_error_2chan_wshar_auxloss.mean(1)[n_rows-1], test_error_2chan_wshar_auxloss.std(1)[n_rows-1]] , 'a', gv.SPACER, alignment="horizontal")
    
    
    
    
        
        ## depending on benchmark type print also additional data 
        if benchmarktype == "SCAN_NEPOCHS": 
            ### print data for each epoch step
            if "Default" in requops:
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_1chan, train_error_1chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_1chan, test_error_1chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_1chan, train_loss_1chan, 'w', gv.SPACER)
    
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_2chan, train_error_2chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_2chan, test_error_2chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_2chan, train_loss_2chan, 'w', gv.SPACER)
            if "Default2ChanOnly" in requops:
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_2chan, train_error_2chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_2chan, test_error_2chan, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_2chan, train_loss_2chan, 'w', gv.SPACER)
    
            if "WeightSharing" in requops:
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_2chan_wshar, train_error_2chan_wshar, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_2chan_wshar, test_error_2chan_wshar, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_2chan_wshar, train_loss_2chan_wshar, 'w', gv.SPACER)
    
            if "AuxiliaryLoss" in requops:
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_2chan_auxloss, train_error_2chan_auxloss, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_2chan_auxloss, test_error_2chan_auxloss, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_2chan_auxloss, train_loss_2chan_auxloss, 'w', gv.SPACER)
    
            if "WeightSharingAndAuxiliaryLoss" in requops:
                hlf.print2DDataToFile(curr_ofile_trainerrvseps_2chan_wshar_auxloss, train_error_2chan_wshar_auxloss, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_testerrvseps_2chan_wshar_auxloss, test_error_2chan_wshar_auxloss, 'w', gv.SPACER)
                hlf.print2DDataToFile(curr_ofile_trainlossvseps_2chan_wshar_auxloss, train_loss_2chan_wshar_auxloss, 'w', gv.SPACER)
    
        elif benchmarktype == "SCAN_OTHER":
            pass
    
        else:
            print("ERROR: The requested benchmarktype (" + benchmarktype + ") is not implemented!")
            exit(2)
    
        
    
    
        print("")
        print("Jobstep " + str(curr_js) + " done!")
        #exit(0)
        print("############################################################")

        print("STATUS: Computation with ID#: " + cid + " finished!")

