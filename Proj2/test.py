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

train_input, train_target = data_gen(n_train)
test_input, test_target = data_gen(n_test)

# t_train_input, _ = data_transform(train_input)
# t_test_input, _ = data_transform(test_input)
# t_train_input = data_reshape(train_input)
# t_test_input = data_reshape(test_input)
################################################################################
std = 0.1
# std2 = 1.0
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
epochs = 10


print('\n Training')
# train_model(model, train_input, train_target, loss_criterion, optimizer, epochs, mini_batch_size)
losses, weights = train_model_no_batch(model, train_input, train_target, loss_criterion, optimizer, eta, epochs)

print('\n Testing')
# test_model_batch(model, test_input, test_target, loss_criterion, mini_batch_size)
# n_errors, n_correct, ix, iy, cx, cy,l = test_model_no_batch(model, test_input, test_target, loss_criterion)
n_errors, n_correct, ix, iy, cx, cy,l = test_model_wip(model, test_input, test_target, loss_criterion)

print('\n Plotting')
plot_results(ix, iy, cx, cy)
# print(len(test_input))
# print(len(l))
# for w in range(len(test_input)):
#     print('input', test_input[w], '\n')
#     print('target', test_target[w], '\n')
#     print('prediction',l[w], '\n')
    # if(l[w]):print('cx', cx[w], 'cy', cy[w], '\n \n \n')
    # if(not l[w]):print('ix', ix[w], 'iy', iy[w], '\n \n \n')
    # print('ix', cx[w], 'iy', cy[w], '\n')
    # print('x', x[w], '\n')
    # print('y', y[w], '\n')
    # if(n_errors!=0):print('i', ix[w],iy[w], '\n')
    # if(n_correct!=0):print('c', cx[w],cy[w], '\n', '\n')


data_plot(test_input, test_target)
################################################################################
################################################################################
################################################################################
