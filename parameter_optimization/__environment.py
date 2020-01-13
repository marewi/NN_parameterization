class Agent:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __str__(self):
        return f"param1: {self.param1}, param2: {self.param2}"

    def action(self, choice):
        if choice == 0:
            self.change(param1=0.1) # higher
        elif choice == 1:
            self.change(param1=-0.1) # lower
        
        if choice == 2:
            self.change(param2=0.1) # higher
        elif choice == 3:
            self.change(param2=-0.1) # lower


    def change(self, param1=False, param2=False):
        self.param1 += param1
        self.param2 += param2

        # fix limits of parameters
        if self.param1 < 0:
            self.param1 = 0
        elif self.param1 > 1:
            self.param1 = 1
        elif self.param2 < 0:
            self.param2 = 0
        elif self.param2 > 1:
            self.param2 = 1
        