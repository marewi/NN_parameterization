from parameters import num_epochs_stepsize, batch_size_stepsize, learning_rate_stepsize
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


    def change(self, num_epochs=False, batch_size=False, learning_rate=False):
        self.num_epochs += num_epochs
        self.batch_size += batch_size
        self.learning_rate += learning_rate

        # 1 <= num_epochs <= 10
        # 1 <= batch_size <= 10
        # 0 <= learning_rate <= 1

        barrier = False

        # fix limits of parameters
        if self.num_epochs < 1:
            self.num_epochs = 1
            barrier = True
        elif self.num_epochs > 10:
            self.num_epochs = 10
            barrier = True

        if self.batch_size < 1:
            self.batch_size = 1
            barrier = True
        elif self.batch_size > 10:
            self.batch_size = 10
            barrier = True

        if self.learning_rate < 0:
            self.learning_rate = 0
            barrier = True
        elif self.learning_rate > 1:
            self.learning_rate = 1
            barrier = True

        if barrier == True:
            raise Exception("barrier reached")