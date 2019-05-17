'''
test
'''
from Sequential import *
from Module import *
from Optimizer import *
from Utils import *
import time



################################################################################
################################################################################
################################################################################
'''
main
creates a dataset, a model, an optimizer, a loss criterion, with learning rate, level of layer standard deviation initialization for weights and biases, batch size
prints losses and training/testing errors
'''
def main():


    torch.set_grad_enabled(False)


    n_train, n_test = 1000, 1000
    train_input, train_target = data_gen(n_train)
    test_input, test_target = data_gen(n_test)
    std = 1e-1
    batch_size = 20
    epochs = 50
    eta = 1e-1


    model = Sequential(Linear(2,25, std),
                        Tanh(),
                        Linear(25,25, std),
                        Tanh(),
                        Linear(25,25, std),
                        Tanh(),
                        Linear(25,25, std),
                        Tanh(),
                        Linear(25,2, std))


    optimizer = SGDMomentum(model.param(), eta)


    loss_criterion = MSELoss()


    print('\n Training...')
    losses, weights = train(model, train_input, train_target, loss_criterion, optimizer, eta, epochs, verbose)



    print('\n Testing...')
    n_error_train, ix_train, iy_train, cx_train, cy_train, l_train = eval(model, train_input, train_target, loss_criterion, verbose)
    n_error_test,  ix_test, iy_test, cx_test, cy_test,l_test = eval(model, test_input, test_target, loss_criterion, verbose)
    print('train error {} %'.format(n_error_train/len(train_input)))
    print('test error {} %'.format(n_error_test/len(test_input)))

    

################################################################################
if __name__ == '__main__':
    main()