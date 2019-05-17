from Sequential import *
from Module import *
from Optimizer import *
from Utils import *

################################################################################
################################################################################
################################################################################
def test(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save):
        

     
        train_input, train_target = data_gen(n_train)
        test_input, test_target = data_gen(n_test)

        ################################################################################


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

        model3 = Sequential(Linear(2,50, std),
                        Tanh(),
                        Linear(50,50, std),
                        Tanh(),
                        Linear(50,50, std),
                        Tanh(),
                        Linear(50,50, std),
                        Tanh(),
                        Linear(50,2, std))

        model4 = Sequential(Linear(2,10, std),
                        Tanh(),
                        Linear(10,10, std),
                        Tanh(),
                        Linear(10,10, std),
                        Tanh(),
                        Linear(10,10, std),
                        Tanh(),
                        Linear(10,2, std))
        ################################################################################
        if(model_no ==1): model = model1
        if(model_no ==1): model = model2
        if(model_no ==1): model = model3
        if(model_no ==1): model = model4
        

        optimizer = SGD(model.param(), eta)

        ################################################################################
        loss_criterion = MSELoss()

        ################################################################################


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

         if(verbose):
                print('eta {}, std {}, epochs{}, batch_size {}, n_train {}, n_test {}, error train % {}, error test %{}, tictoc {}'.format(
                    eta, std, epochs, batch_size, n_train, n_test, 100.*n_error_train/n_train, 100.*n_error_test/n_test, tictoc))

         if(plot):
                data_plot(test_input, test_target)
                plot_results(ix, iy, cx, cy)

         if(save):
                text = []
                text.append('eta {}'.format(eta))
                text.append('epochs {}'.format(epochs))
                text.append('batch size {}'.format(batch_size))
                text.append('n_train {}'.format(n_train))
                text.append('n_test {}'.format(n_test))
                text.append('error train % {}'.format(100*n_error_train/n_train))
                text.append('error test % {}'.format(100*n_error_test/n_test))
                text.append('tictoc {}'.format(tictoc))
                write_to_csv(text, id)

################################################################################
################################################################################
################################################################################


def main():
    torch.set_grad_enabled(False)
    model_no = 2
    eat = 1e-4
    n_train, n_test = 1000, 1000
    std = 0.1
    batch_size = 20
    epochs = 100
    verbose, plot, save = False, False, True
    test(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save)


if __name__ == '__main__':
    main()