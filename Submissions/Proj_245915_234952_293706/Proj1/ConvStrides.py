
class ConvStrides():
    
    # members: 
    #   TODO

    # methods:
    #   TODO


    def __init__(self, stride_inlay, stride_hidlays, stride_outlay):
        super(ConvStrides, self).__init__()

        self.strid_il = stride_inlay
        self.strid_hls = stride_hidlays
        self.strid_ol = stride_outlay