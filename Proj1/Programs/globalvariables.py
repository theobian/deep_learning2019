# general parameters 
## directories 
PROJDIR = "/home/dankl/Desktop/MyLocalData/ConferencesAndCourses/SS2019_DeepLearning/Proj1/"
#PROJDIR = "/home/chemophysiker/VirtualEnvironmentsPython/VE002_Python3_4/ML_course/projects/project2/"
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
NNPARS_FILENAME = "NeuralNetworkParameters.txt"
HIDDENLAYERPARS_FILENAME = "HiddenLayerParameters.txt"
IOLAYERPARAMETERS_FILENAME = "IOLayerParameters.txt"
LOADDATAPARS_FILENAME = "LoadDataParameters.txt"
OFILES_FILENAME = "Ofilenames.txt"
COMMONOFILES_FILENAME = "CommonOFilenamesForAllJSs.txt"


## column-indices in specific files
### General Parameters
INDEX_NJOBSTEP = 0
INDEX_NPARSPERLAYER = 1

### Neural network parameters 
INDEX_NNMODEL = 0
INDEX_HIDDEN_LAYERS = 1
INDEX_LOSS = 2
INDEX_LEARNINGRATE = 3
INDEX_NEPOCHS = 4


### Hidden layer parameters
INDEX_NPERCEPTRONS = 0
INDEX_ACTIVFUNC = 1
INDEX_DROPOUT = 2


### IO layer parameters
INDEX_NINPERCEPTR = 0
INDEX_INACTIVFUNC = 1
INDEX_NOUTPERCEPTR = 2
INDEX_OUTACTIVFUNC = 3


### Load Data parameters
INDEX_SINGLEORPAIRSOFDIGITS = 0
INDEX_NDATAPOINTS = 1

### Common O-filenames
INDEX_TRAIN_TEST_ERROR = 0

### O-filenames
INDEX_MODELPARS = 0




## datatypes
### cast tensor to the cuda datatype
#dtype_cuda = torch.cuda.FloatTensor




