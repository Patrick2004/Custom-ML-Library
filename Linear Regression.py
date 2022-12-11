import numpy as np
import matplotlib.pyplot as plt
import sys

class Linear_Regression:
    def __init__(self):
        self.x = []
        self.y = []
        self.size = 0

        self.weight = []
        self.error = sys.maxsize
        self.cost = 0
        
    def dataset(self, x, y):
        self.x = x
        self.y = y
        self.size = len(x)
        
    def train(self):
        error = 0
        weight = []
        for i in range(1, self.size-1):
            weight, error, _, _, _ = np.polyfit(self.x, self.y, i, full=True)
            if (error > self.error):
                break
            
            self.error = error
            self.weight = weight        

    def predict(self, X):
        result = []
        for x in X:
            result.append(sum([self.weight[-(i+1)]*(x**i) for i in range(len(self.weight))]))
        return result

lr = Linear_Regression()
lr.dataset([1, 2, 3, 4, 5, 6, 7], [2, 4, 6, 8, 10, 12, 14])
lr.train()
print("\nLR: ", lr.predict([2, 6, 8, 20]))
