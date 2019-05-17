from Sequential import *
from Module import *
from Optimizer import *
from Utils import *

################################################################################
################################################################################
################################################################################
n_train = 1000
n_test = 1000
# n_train_long = 10000

train_input, train_target = dummy_data_gen(n_train)
test_input, test_target = dummy_data_gen(n_test)
dummy_data_plot(train_input, train_target)

################################################################################
std = 0.1
# std2 = 1.0
model1=Sequential(Linear(2,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,2, std))


model2=Sequential(Linear(2,25, std),
                Tanh(),
                Linear(25,25, std),
                Tanh(),
                Linear(25,25, std),
                Tanh(),
                Linear(25,25, std),
                Tanh(),
                Linear(25,2, std))

################################################################################
model = model2

eta = 0.001
optimizer = SGD(model.param(), eta)

################################################################################
loss_criterion = MSELoss()

################################################################################
mini_batch_size = 5
epochs = 50


print('\n Training')
losses = train_model_no_batch(model, train_input, train_target, loss_criterion, optimizer, epochs)

print('\n Testing')
n_errors, n_correct, ix, iy, cx, cy, x, y, l = test_model_no_batch(model, test_input, test_target, loss_criterion)

print('\n Plotting')
dummy_plot_results(ix, iy, cx, cy)

# print(x+y)
# print(l)
# print(test_input)
# print(test_target)




################################################################################
################################################################################
################################################################################
