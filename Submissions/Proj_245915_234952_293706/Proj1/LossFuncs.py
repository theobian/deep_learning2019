
class LossFuncs():
    
    # members: 
    #   TODO

    # methods:
    #   TODO


    def __init__(self, loss_outlayer, loss_aux, weight_loss_aux):
        super(LossFuncs, self).__init__()

        self.lossfunc_outlayer = loss_outlayer
        self.lossfunc_aux = loss_aux
        self.lossaux_weight = weight_loss_aux