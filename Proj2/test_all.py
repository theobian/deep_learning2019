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
creates a dataset, a model, an optimizer, a loss criterion, with learning rate, level of layer standard deviation initialization for weights and biases, batch size
prints losses and training/testing errors
'''
def test(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save, file_id):


################################################################################
    train_input, train_target = data_gen(n_train)
    test_input, test_target = data_gen(n_test)


################################################################################
    model1 = Sequential(Linear(2,25, std),
                        ReLU(),
                        Linear(25,25, std),
                        ReLU(),
                        Linear(25,25, std),
                        ReLU(),
                        Linear(25,25, std),
                        ReLU(),
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

    model5 = Sequential(Linear(2,20, std),
                        Tanh(),
                        Linear(20,20, std),
                        Tanh(),
                        Linear(20,2, std))    


################################################################################
    if(model_no == 1): model = model1
    if(model_no == 2): model = model2
    if(model_no == 3): model = model3
    if(model_no == 4): model = model4
    if(model_no == 5): model = model5


################################################################################
    optimizer = SGD(model.param(), eta)


################################################################################
    loss_criterion = MSELoss()


################################################################################
    tic = time.time()
    print('\n Training...')
    # train_model(model, train_input, train_target, loss_criterion, optimizer, epochs, mini_batch_size)
    losses, weights = train(model, train_input, train_target, loss_criterion, optimizer, eta, epochs, verbose)


################################################################################
    print('\n Testing...')
    # test_model_batch(model, test_input, test_target, loss_criterion, mini_batch_size)
    n_error_train, ix_train, iy_train, cx_train, cy_train, l_train = eval(model, train_input, train_target, loss_criterion, verbose)
    n_error_test,  ix_test, iy_test, cx_test, cy_test,l_test = eval(model, test_input, test_target, loss_criterion, verbose)
    toc = time.time()
    tictoc = toc -  tic
    print('train error {} %'.format(n_error_train/len(train_input)))
    print('test error {} %'.format(n_error_test/len(test_input)))

    
################################################################################
    if(verbose): 
        print('eta {}, std {}, epochs{}, batch_size {}, n_train {}, n_test {}, error train % {}, error test %{}, tictoc {}'.format(
        eta, std, epochs, batch_size, n_train, n_test, 100.*n_error_train/n_train, 100.*n_error_test/n_test, tictoc))


################################################################################
    if(plot):
        print('\n Plotting...')
        data_plot(test_input, test_target)
        plot_results(ix, iy, cx, cy)


################################################################################
    if(save):
        print('\n Saving...')
        text = []
        text.append('model no {}'.format(model_no))
        text.append('eta {}'.format(eta))
        text.append('epochs {}'.format(epochs))
        text.append('batch size {}'.format(batch_size))
        text.append('n_train {}'.format(n_train))
        text.append('n_test {}'.format(n_test))
        text.append('error train % {}'.format(100*n_error_train/n_train))
        text.append('error test % {}'.format(100*n_error_test/n_test))
        text.append('tictoc {}'.format(tictoc))
        write_to_csv(text, file_id)


################################################################################
################################################################################
################################################################################
'''
runs the framework with specified parameters
makes sure to turn the torch grad off
runs different iterations for various parameters
'''
def main():


################################################################################    
	torch.set_grad_enabled(False)

	verbose, plot, save = False, False, True


	# ################################################################################  
	# model_no = 2    
	# n_train, n_test = 1000, 1000
	# std = 0.1
	# batch_size = 20
	# epochs = 100
	# file_id = ['model2_eta_1e-4', 'model2_eta_1e-3', 'model2_eta_1e-2', 'model2_eta_1e-1']
	# eta = [1e-4, 1e-3, 1e-2, 1e-1]
	# for i in range(len(file_id)):
	# 	test(model_no, n_train, n_test, std, eta[i], batch_size, epochs, verbose, plot, save, file_id[i])


	# ################################################################################
	# model_no = [1, 2, 3, 4, 5]
	# n_train, n_test = 1000, 1000
	# std = 0.1
	# batch_size = 20
	# epochs = 100
	# file_id = ['model1', 'model2', 'model3', 'model4', 'model5']
	# eta = 1e-1
	# for i in range(len(file_id)):
	# 	test(model_no[i], n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save, file_id[i])


	# ################################################################################
	# model_no = 2
	# n_train, n_test = 1000, 1000
	# std = 0.1
	# batch_size = 20
	# epochs = [10, 50, 100, 300]
	# file_id = ['model2_e_10', 'model2_e_50', 'model2_e_100', 'model2_e_300']
	# eta = 1e-1
	# for i in range(len(file_id)):
	# 	test(model_no, n_train, n_test, std, eta, batch_size, epochs[i], verbose, plot, save, file_id[i])


	# ################################################################################
	# model_no = 2
	# n_train, n_test = 1000, 1000
	# std = [1e-2, 1e-1, 1]
	# batch_size = 20
	# epochs = 100
	# file_id = ['model2_std_1e-2', 'model2_std_1e-1', 'model2_std_1']
	# eta = 1e-1
	# for i in range(len(file_id)):
	# 	test(model_no, n_train, n_test, std[i], eta, batch_size, epochs, verbose, plot, save, file_id[i])


	# ################################################################################
	# model_no = 5
	# n_train, n_test = 1000, 1000
	# std = 1e-1
	# batch_size = 20
	# epochs = 100
	# file_id = 'model5'
	# eta = 1e-1
	# fest(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save, file_id)


	# ################################################################################
	# model_no = 2
	# n_train, n_test = 10000, 1000
	# std = 1e-1
	# batch_size = 20
	# epochs = 50
	# file_id = 'model2_optimal'
	# eta = 1e-1
	# test(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save, file_id)


	# ###############################################################################
	# model_no = 2
	# n_train, n_test = 1000, 1000
	# std = 1e-1
	# batch_size = 20
	# epochs = [10, 50, 100, 300, 500]
	# file_id = ['model2_sub_e_10', 'model2_sub_e_50', 'model2_sub_e_100', 'model2_sub_e_1000', 'model2_sub_e_10000']
	# eta = 1e-1
	# for i in range(len(file_id)):
	# 	test(model_no, n_train, n_test, std, eta, batch_size, epochs[i], verbose, plot, save, file_id[i])       


	# ##############################################################################
	# model_no = 2
	# n_train, n_test = [10, 100, 1000, 10000], [10, 100, 1000, 10000]
	# std = 1e-1
	# batch_size = 20
	# epochs = 50
	# file_id = ['model2_sub_n_10', 'model2_sub_n_100', 'model2_sub_n_1000', 'model2_sub_n_10000']
	# eta = 1e-1
	# for i in range(len(file_id)):
	# 	test(model_no, n_train[i], n_test[i], std, eta, batch_size, epochs, verbose, plot, save, file_id[i])       


	# ###############################################################################
	# model_no = 2
	# n_train, n_test = 1000, 1000
	# std = 1e-1
	# batch_size = 20
	# epochs = 50
	# file_id = ['model2_submission_1', 'model2_submission_2', 'model2_submission_3', 'model2_submission_4', 'model2_submission_5']
	# eta = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	# for i in range(len(file_id)):
	# 	test(model_no, n_train, n_test, std, eta[i], batch_size, epochs, verbose, plot, save, file_id[i])


	################################################################################
	model_no = 2
	n_train, n_test = 1000, 1000
	std = 1e-1
	batch_size = 20
	epochs = 20
	file_id = ['model2_optsub_1', 'model2_optsub_2', 'model2_optsub_3', 'model2_optsub_4', 'model2_optsub_5',
			'model2_optsub_6', 'model2_optsub_7', 'model2_optsub_8', 'model2_optsub_9', 'model2_optsub_10']
	eta = 1e-1
	for i in range(len(file_id)):
		test(model_no, n_train, n_test, std, eta, batch_size, epochs, verbose, plot, save, file_id[i])

################################################################################
################################################################################
################################################################################
'''
main
'''
if __name__ == '__main__':
    main()