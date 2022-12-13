import numpy as np
import matplotlib.pyplot as plt

class Linear_Regression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.p = 0
        self.error = 0
        self.size = len(x)
        

    def train(self):
        self.p, self.error, _, _, _ = np.polyfit(self.x, self.y, 1, full=True)
        error = self.error
        p = self.p
        for i in range(1, self.size - 2):
            i += 1
            self.p, self.error, _, _, _ = np.polyfit(self.x, self.y, i, full=True)
            if self.error < error:
                error = self.error
                p = self.p
            else:
                break
        self.p = p
        self.error = error
        print(self.p)
    
    def estimate(self, x):
        power = len(self.p) - 1
        result = 0
        for i in range(power):
            result += self.p[i]*(x**(power - i))
            print(result)
        print(result)


                

        
        
    
    
