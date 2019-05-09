

class Optimizer():
	def __init__(self, *args):
		pass

	def update_parameters(self, input_parameters):
		pass

# child class instance should have the model parameters/weights+bias as attributes, and uptdate them when required. Learning Rate and Momentum should be in there!
class SGD(Optimizer):
	def __init__(self, parameters, learning_rate, *args):
		pass

	def update_parameters(self):
		pass


class SGD_momentum(Optimizer):
	def __init__(self, parameters, learning_rate, momentum, *args):
		pass

	def update_parameters(self):
		pass
