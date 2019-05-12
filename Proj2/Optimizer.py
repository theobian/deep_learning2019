#==============================================================================#
#==============================================================================#
#==============================================================================#
class Optimizer():

	def __init__(self, *args):
		raise NotImplementedError


	def update(self, *args):
		raise NotImplementedError


#==============================================================================#
#==============================================================================#
#==============================================================================#
class SGD(Optimizer):

	def __init__(self, parameters, learning_rate):
		self.parameters = parameters
		self.learning_rate = learning_rate


	def update(self):
		for w in self.parameters:
			w[0] -= self.learning_rate * w[1]



class SGDMomentum(Optimizer):

	def __init__(self, parameters, learning_rate, gamma=0.9):
		self.parameters = input_parameters
		self.learning_rate = learning_rate
		self.gamma = gamma


	def update(self, prev_update):
		for w in self.parameters:
			w[0] -= self.learning_rate * w[1]
			self.learning_rate += (self.gamma * prev_update)


#==============================================================================#
#==============================================================================#
#==============================================================================#
