
class ConvPaddings():
    
    # members: 
    #   TODO

    # methods:
    #   TODO


    def __init__(self, padding_inlay, padding_hidlays, padding_outlay):
        super(ConvPaddings, self).__init__()

        self.pad_il = padding_inlay
        self.pad_hls = padding_hidlays
        self.pad_ol = padding_outlay