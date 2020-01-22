from parameters import *
from termcolor import colored

class Agent:
    def __init__(self, num_epochs, batch_size, learning_rate, momentum):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
    
    def __str__(self):
        return f"num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}"

    def action(self, choice):
        print(f"action {choice} will be taken")
        if choice == 0:
            self.change(num_epochs=num_epochs_stepsize) # higher
        elif choice == 1:
            self.change(num_epochs=-num_epochs_stepsize) # lower
        
        elif choice == 2:
            self.change(batch_size=batch_size_stepsize) # higher
        elif choice == 3:
            self.change(batch_size=-batch_size_stepsize) # lower
        
        elif choice == 4:
            self.change(learning_rate=learning_rate_stepsize) # higher
        elif choice == 5:
            self.change(learning_rate=-learning_rate_stepsize) # lower

        elif choice == 6:
            self.change(momentum=momentum_stepsize) # higher
        elif choice == 7:
            self.change(momentum=-momentum_stepsize) # lower


    def change(self, num_epochs=False, batch_size=False, learning_rate=False, momentum=False):
        self.num_epochs += num_epochs
        self.batch_size += batch_size
        self.learning_rate += learning_rate
        self.momentum += momentum

        # 1 <= num_epochs <= 10
        # 1 <= batch_size <= 10
        # 0 <= learning_rate <= 1

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
        
        if self.momentum < momentum_min:
            self.momentum = momentum_min
            barrier = True
        elif self.momentum > momentum_max:
            self.momentum = momentum_max
            barrier = True

        if barrier == True:
            raise Exception("barrier reached")