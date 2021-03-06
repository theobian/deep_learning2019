import os
# general parameters 
## directories 
#PROJDIR = "/home/dankl/Desktop/MyLocalData/ConferencesAndCourses/SS2019_DeepLearning/Proj1/"
#PROJDIR = "/home/chemophysiker/VirtualEnvironmentsPython/VE002_Python3_4/ML_course/projects/project2/"
PROJDIR = os.getcwd() + "/"

SCRIPTDIR = PROJDIR + "Scripts/"
COMPDIR = PROJDIR + "Computations/"
PROGRAMDIR = PROJDIR + "Programs/"
EVALDIR = PROJDIR + "Evaluations/"
RESULTDIR = PROJDIR + "Results/"

## praefixes
COMPPRAEFIX = "COMP"
AUTHORPRAEFIX = "MD"
JOBSTEPPRAEFIX = "JS"

## Diverse
NDIGITS_JOBSTEPS = 7
JOBSTEPFOLDERSPACER = "0"


## filenames
COMPDESCR_FILENAME = COMPDIR + AUTHORPRAEFIX + "_ComputationDescriptions.txt"






# computation specific settings
## subdirectories
IFILESDIR = "Input/"
OFILESDIR = "Output/"

## filenames
GENPARS_FILENAME = "GeneralParameters.txt"
NNPARS_FILENAME = "NeuralNetworkParameters.csv"
HIDDENLAYERPARS_FILENAME = "HiddenLayerParameters.csv"
IOLAYERPARAMETERS_FILENAME = "IOLayerParameters.csv"
AUXLOSSPARAMETERS_FILENAME = "AuxiliaryLossParameters.csv"
LOADDATAPARS_FILENAME = "LoadDataParameters.txt"
OFILES_FILENAME = "OFilenames.csv"
COMMONOFILES_FILENAME = "CommonOFilenamesForAllJSs.txt"
REQUOPTFORARCHIT_FILENAME = "RequestedOptionsForArchitecture.csv" 	#Allowed Values:  Default	WeightSharing	AuxiliaryLoss	WeightSharingAndAuxiliaryLoss


## column-indices in specific files
### General Parameters
COL_NJOBSTEP = 0
COL_NREPITTIONS = 1
COL_NPARSPERREG1DLAYER = 2
COL_NPARSPERCONV2DLAYER = 3
COL_BENCHMARKINGTYPE = 4

### Neural network parameters 
COL_HIDDEN_LAYERS = 0
COL_LOSS = 1
COL_OPTIMZER = 2
COL_LEARNINGRATE = 3
COL_NEPOCHS = 4
COL_NMINIBATCHSIZE = 5


### Hidden layer parameters
COL_HL_TYPES = 0
COL_HL_NPERCEPTRONS = 1
COL_HL_ACTIVFUNCS = 2
##COL_HL_DROPOUTS = 3
COL_HL_CONVSTRIDES = 3
COL_HL_CONVKERNELSIZES = 4
COL_HL_CONVPADDINGS = 5
COL_HL_POOLTYPES = 6
COL_HL_POOLSTRIDES = 7
COL_HL_POOLKERNELSIZES = 8
COL_HL_POOLPADDINGS = 9



### IO layer parameters
COL_IL_TYPE = 0
COL_IL_NPERCEPTR = 1
COL_IL_ACTIVFUNC = 2
COL_IL_CONVSTRIDE = 3
COL_IL_CONVKERNELSIZE = 4
COL_IL_CONVPADDING = 5
COL_IL_POOLTYPE = 6
COL_IL_POOLSTRIDE = 7
COL_IL_POOLKERNELSIZE = 8
COL_IL_POOLPADDING = 9
COL_OL_TYPE = 10
COL_OL_NPERCEPTR = 11
COL_OL_ACTIVFUNC = 12
COL_OL_CONVSTRIDE = 13
COL_OL_CONVKERNELSIZE = 14
COL_OL_CONVPADDING = 15
COL_OL_POOLTYPE = 16
COL_OL_POOLSTRIDE = 17
COL_OL_POOLKERNELSIZE = 18
COL_OL_POOLPADDING = 19

### Auxiliary loss parameters
COL_AUXLOSS_ACTIVFUNC = 0
COL_AUXLOSS_LOSS = 1
COL_AUXLOSS_WEIGHT = 2

### Load Data parameters
COL_SINGLEORPAIRSOFDIGITS = 0
COL_NDATAPOINTS = 1

### O-filenames common to all jobsteps
COL_ALLJS_TRAINERR_1CHAN = 0
COL_ALLJS_TESTERR_1CHAN = 1
COL_ALLJS_TRAINLOSS_1CHAN = 2
COL_ALLJS_AVGTRAINERR_1CHAN = 3
COL_ALLJS_AVGTESTERR_1CHAN = 4
COL_ALLJS_TRAINERR_2CHAN = 5
COL_ALLJS_TESTERR_2CHAN = 6
COL_ALLJS_TRAINLOSS_2CHAN = 7
COL_ALLJS_AVGTRAINERR_2CHAN = 8
COL_ALLJS_AVGTESTERR_2CHAN = 9
COL_ALLJS_TRAINERR_2CHAN_WSHAR = 10
COL_ALLJS_TESTERR_2CHAN_WSHAR = 11
COL_ALLJS_TRAINLOSS_2CHAN_WSHAR = 12
COL_ALLJS_AVGTRAINERR_2CHAN_WSHAR = 13
COL_ALLJS_AVGTESTERR_2CHAN_WSHAR = 14
COL_ALLJS_TRAINERR_2CHAN_AUXLOSS = 15
COL_ALLJS_TESTERR_2CHAN_AUXLOSS = 16
COL_ALLJS_TRAINLOSS_2CHAN_AUXLOSS = 17
COL_ALLJS_AVGTRAINERR_2CHAN_AUXLOSS = 18
COL_ALLJS_AVGTESTERR_2CHAN_AUXLOSS = 19
COL_ALLJS_TRAINERR_2CHAN_WSHAR_AUXLOSS = 20
COL_ALLJS_TESTERR_2CHAN_WSHAR_AUXLOSS = 21
COL_ALLJS_TRAINLOSS_2CHAN_WSHAR_AUXLOSS = 22
COL_ALLJS_AVGTRAINERR_2CHAN_WSHAR_AUXLOSS = 23
COL_ALLJS_AVGTESTERR_2CHAN_WSHAR_AUXLOSS = 24


### O-filenames individual to jobsteps
COL_MODELPARS = 0
COL_TRAINERR_EPOCHS_1CHAN = 1
COL_TESTERR_EPOCHS_1CHAN = 2
COL_TRAINLOSS_EPOCHS_1CHAN = 3
COL_TRAINERR_EPOCHS_2CHAN = 4
COL_TESTERR_EPOCHS_2CHAN = 5
COL_TRAINLOSS_EPOCHS_2CHAN = 6
COL_TRAINERR_EPOCHS_2CHAN_WSHAR = 7
COL_TESTERR_EPOCHS_2CHAN_WSHAR = 8
COL_TRAINLOSS_EPOCHS_2CHAN_WSHAR = 9
COL_TRAINERR_EPOCHS_2CHAN_AUXLOSS = 10
COL_TESTERR_EPOCHS_2CHAN_AUXLOSS = 11
COL_TRAINLOSS_EPOCHS_2CHAN_AUXLOSS = 12
COL_TRAINERR_EPOCHS_2CHAN_WSHAR_AUXLOSS = 13
COL_TESTERR_EPOCHS_2CHAN_WSHAR_AUXLOSS = 14
COL_TRAINLOSS_EPOCHS_2CHAN_WSHAR_AUXLOSS = 15

### Output Data
COL_OUTPUT_TRAINERROR = 0
COL_OUTPUT_TRAINERROR_STD = 1

COL_OUTPUT_TESTERROR = 0
COL_OUTPUT_TESTERROR_STD = 1




## string-separators in specific files
STRSEP_IOLAYERPARSFILE = ","
STRSEP_HIDDENLAYERPARSFILE = ","
STRSEP_NNPARSFILE = ","
STRSEP_OFILENAMESFILE = ","
STRSEP_COOFILENAMESFILE = ","
STRSEP_AUXOLAYERPARSFILE = ","
STRSEP_REQUOPTFORARCHIT = ","





# default plotting format options 
PLOT_LINEWIDTH = 1.5


# datatypes


# numerical 
NUMPREC = 1e-16


# writing of numerical data to files
SPACER = ','