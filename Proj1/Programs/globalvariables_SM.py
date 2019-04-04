# general parameters 
## directories 
PROJDIR = "/home/dankl/Desktop/MyLocalData/ConferencesAndCourses/SS2019_DeepLearning/Miniproject01/"
#PROJDIR = "/home/chemophysiker/VirtualEnvironmentsPython/VE002_Python3_4/ML_course/projects/project2/"
SCRIPTDIR = PROJDIR + "Scripts/"
COMPDIR = PROJDIR + "Computations/"
PROGRAMDIR = PROJDIR + "Programs/"
EVALDIR = PROJDIR + "Evaluations/"
RESULTDIR = PROJDIR + "Results/"

## praefixes
COMPPRAEFIX = "COMP"
AUTHORPRAEFIX = "SM"
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



## datatypes
### cast tensor to the cuda datatype
#dtype_cuda = torch.cuda.FloatTensor




