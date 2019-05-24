
class NeuralNetworkArchitecture():
    
    # members: 
    #   TODO

    # methods:
    #   TODO


    def __init__(self, layertypes, nperceptrs, convkernelsizes, convpaddings, convstrides, pooltype, poolkernelsizes, poolpaddings, poolstrides):
        super(NeuralNetworkArchitecture, self).__init__()
        self.laytyps = layertypes
        self.npercs = nperceptrs
        self.convkernsiz = convkernelsizes
        self.convpads = convpaddings
        self.convstrids = convstrides
        self.pooltyp = pooltype
        self.poolkernsiz = poolkernelsizes
        self.poolpads = poolpaddings
        self.poolstrids = poolstrides

