class Agent:
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def __str__(self):
        return f"num_epochs: {self.num_epochs}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}"

    def action(self, choice):
        if choice == 0:
            self.change(num_epochs=1) # higher
        elif choice == 1:
            self.change(num_epochs=-1) # lower
        
        if choice == 2:
            self.change(batch_size=1) # higher
        elif choice == 3:
            self.change(batch_size=-1) # lower
        
        if choice == 4:
            self.change(learning_rate=0.00001) # higher
        elif choice == 5:
            self.change(learning_rate=-0.00001) # lower


    def change(self, num_epochs=False, batch_size=False, learning_rate=False):
        self.num_epochs += num_epochs
        self.batch_size += batch_size
        self.learning_rate += learning_rate

        # 0 <= num_epochs <= 100
        # 0 <= batch_size <= 100
        # 0 <= learning_rate <= 1

        # fix limits of parameters
        if self.num_epochs < 0:
            self.num_epochs = 0
        elif self.num_epochs > 100:
            self.num_epochs = 100

        if self.batch_size < 0:
            self.batch_size = 0
        elif self.batch_size > 100:
            self.batch_size = 100

        if self.learning_rate < 0:
            self.learning_rate = 0
        elif self.learning_rate > 1:
            self.learning_rate = 1
        