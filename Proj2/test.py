from Sequential import *
from Module import *
from Optimizer import *
from Utils import *

################################################################################
################################################################################
################################################################################
n_train = 10000
n_test = 1000

train_input, train_target = data_gen(n_train)
test_input, test_target = data_gen(n_test)

################################################################################
std = 0.1

model1 = Sequential(Linear(2,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,25, std),
                Relu(),
                Linear(25,2, std))


model2 = Sequential(Linear(2,25, std),
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
epochs = 100


print('\n Training')
# train_model(model, train_input, train_target, loss_criterion, optimizer, epochs, mini_batch_size)
losses, weights = train_model_no_batch(model, train_input, train_target, loss_criterion, optimizer, eta, epochs)

print('\n Testing')
# test_model_batch(model, test_input, test_target, loss_criterion, mini_batch_size)
n_errors_train, n_correct_train, ix_train, iy_train, cx_train, cy_train,l_train = test_model_wip(model, train_input, train_target, loss_criterion)
n_errors_test, n_correct_test, ix_test, iy_test, cx_test, cy_test,l_test = test_model_wip(model, test_input, test_target, loss_criterion)
print('train error {} %'.format(n_errors_train/len(train_input)))
print('test error {} %'.format(n_errors_test/len(test_input)))
print('\n Plotting')
# data_plot(test_input, test_target)
# plot_results(ix, iy, cx, cy)


################################################################################
################################################################################
################################################################################
