from parameters import *
from termcolor import colored

class Agent:
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def __str__(self):
        return f"num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}"

    def action(self, choice):
        print(f"action {choice} will be taken")
        if choice == 0:
            self.change(ep=num_epochs_stepsize) # higher
        elif choice == 1:
            self.change(ep=-num_epochs_stepsize) # lower
        elif choice == 2:
            self.change(bs=batch_size_stepsize) # higher
        elif choice == 3:
            self.change(bs=-batch_size_stepsize) # lower
        elif choice == 4:
            self.change(lr=learning_rate_stepsize) # higher
        elif choice == 5:
            self.change(lr=(learning_rate_stepsize*-1)) # lower


    def change(self, ep=0, bs=0, lr=0):
        print(f"ep:{ep}, bs:{bs}, lr:{lr}")

        self.num_epochs = self.num_epochs + ep
        self.batch_size = self.batch_size + bs
        self.learning_rate = self.learning_rate + lr

        self.learning_rate = float("{0:.3f}".format(self.learning_rate))

        print(f"num_epochs:{self.num_epochs}, batch_size:{self.batch_size}, learning_rate:{self.learning_rate}")

        barrier = False

        # fix limits of parameters
        if self.num_epochs < num_epochs_min:
            self.num_epochs = num_epochs_min
            barrier = True
        elif self.num_epochs > num_epochs_max:
            self.num_epochs = num_epochs_max
            barrier = True

        if self.batch_size < batch_size_min:
            self.batch_size = batch_size_min
            barrier = True
        elif self.batch_size > batch_size_max:
            self.batch_size = batch_size_max
            barrier = True

        if self.learning_rate < learning_rate_min:
            self.learning_rate = learning_rate_min
            barrier = True
        elif self.learning_rate > learning_rate_max:
            self.learning_rate = learning_rate_max
            barrier = True

        if barrier == True:
            raise Exception("barrier reached")