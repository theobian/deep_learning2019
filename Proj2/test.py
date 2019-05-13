from Sequential import *
from Module import *
from Optimizer import *
from Utils import *


def main():

    data_nb = 1000
    train_input, train_target = generate_disc_set(data_nb)
    test_input, test_target = generate_disc_set(data_nb)


    std= 0.1
    model=Linear(FullyConnectedLayer(2, 25, std),
        ReLU(), FullyConnectedLayer(25,25, std),
        ReLU(), FullyConnectedLayer(25,25, std),
        ReLU(), FullyConnectedLayer(25,25, std),
        ReLU(), FullyConnectedLayer(25,2, std))

    eta=0.0001
    optimizer = SGD(model.get_parameters(), eta)

    criterion=MSELoss()

    mini_batch_size=50
    epochs=100
    min_loss, best_parameters, losses = train(model, train_input, train_target, epochs, mini_batch_size, criterion, optimizer, True)

    n_error, mask = evaluate(model, test_input, test_target, True)



if __name__ == '__main__':
    main()
