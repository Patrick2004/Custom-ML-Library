import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

class Polynomial_Linear_Regression:
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


class Multiple_Linear_Regression:
    def __init__(self):
        
        self.weight = [0]
        self.cost = []
        self.learning_rate = 0
        
    def dataset(self, x, y):
        self.x = x
        self.y = y

        self.parameter_count = len(self.x[0])
        for i in range(self.parameter_count):
            self.weight.append(1)
        self.size = len(y)
        
    def train(self, epoch, batch_size, learning_rate):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        data_batch = []
        
        remainder = self.size%batch_size
        max_index = self.size-remainder

        batch_x = [self.x[i:i+batch_size] for i in range(0, max_index, batch_size)]

        batch_y = [self.y[i:i+batch_size] for i in range(0, max_index, batch_size)]

        if (remainder != 0):
            batch_x.append(self.x[max_index:])
            batch_y.append(self.y[max_index:])

        data_batch = [batch_x, batch_y]
##        print(data_batch[0], " || ", data_batch[1])
        
        for e in range(epoch):
            for i in range(len(data_batch[0])):
                self.gradient_descent(data_batch[0][i], data_batch[1][i])
                
            cost = self.cost_function(self.x, self.y)
            print("epoch: {} (cost: {:.20f})".format(e+1, cost))

        arr_y = [self.predict(x) for x in self.x]
        arr_x = self.x

##        plt.scatter(self.x, self.y)
##        plt.plot(arr_x, arr_y)
##        plt.show()

        plt.plot([i for i in range(1, epoch+1)], self.cost)
        plt.show()

    def predict(self, x):
##        print([self.weight[i+1]*x[i] for i in range(len(x))])
        return sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]

    def predictList(self, X):
        result = []
        for x in X:
            result.append(self.predict(x))
        return result

    def cost_function(self, X, Y):
        cost = (1/len(X))*sum([(self.predict(X[i]) - Y[i])**2 for i in range(len(Y))])
        self.cost += [cost]
##        print(self.learning_rate/(2*len(X)))
##        print(self.predict(X[0]))
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (2*self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (2* self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])

y_arr = []
x_arr = []

with open("data.csv") as f:
    file = csv.reader(f)
    for i, row in enumerate(file):
##        if i > 101:
##            break

        if i == 0:
            continue
        y_arr.append(int(float((row[2]))))
        x_arr.append(list(map(int, list(map(float, row[3:])))))
        
##    print(y_arr)
##    print()
##    print(x_arr)

MLR = Multiple_Linear_Regression()
MLR.dataset(x_arr, y_arr)
MLR.train(19, 10, 0.00000000001)
print("\nMultiple LR: ", MLR.predictList([x_arr[2], x_arr[6], x_arr[8], x_arr[20]]))

##MLR = Multiple_Linear_Regression()
##MLR.dataset([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10], [6, 12], [7, 14], [8, 16], [9, 18], [10, 20], [11, 22], [12, 24], [13, 26], [14, 28], [15, 30]],
##            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
##MLR.train(100, 15, 0.001)
##print("\nMultiple LR: ", MLR.predictList([[2, 5], [6, 12], [8, 16], [20, 40]]))

##PLR = Polynomial_Linear_Regression()
####PLR.dataset([1, 2, 3, 4, 5, 6, 7], [2, 4, 6, 8, 10, 12, 14])
##PLR.train()
##print("\nPolynomial LR: ", PLR.predict([2, 6, 8, 20]))



class Multiple_Logistic_Regression:
    def __init__(self):
        
        self.weight = [0]
        self.cost = []
        self.learning_rate = 0
        
    def dataset(self, x, y):
        self.x = x
        self.y = y

        self.parameter_count = len(self.x[0])
        for i in range(self.parameter_count):
            self.weight.append(1)
        self.size = len(y)
        
    def train(self, epoch, batch_size, learning_rate):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        data_batch = []
        
        remainder = self.size%batch_size
        max_index = self.size-remainder

        batch_x = [self.x[i:i+batch_size] for i in range(0, max_index, batch_size)]

        batch_y = [self.y[i:i+batch_size] for i in range(0, max_index, batch_size)]

        if (remainder != 0):
            batch_x.append(self.x[max_index:])
            batch_y.append(self.y[max_index:])

        data_batch = [batch_x, batch_y]
        print(data_batch[0], " || ", data_batch[1])
        
        for e in range(epoch):
            for i in range(len(data_batch[0])):
                self.gradient_descent(data_batch[0][i], data_batch[1][i])
                
            cost = self.cost_function(data_batch[0][i], data_batch[1][i])
            print("epoch: {} (cost: {:.20f})".format(e+1, cost))

        arr_y = [self.predict(x) for x in self.x]
        arr_x = self.x

        plt.scatter(self.x, self.y)
        plt.plot(arr_x, arr_y)
        plt.show()

        plt.plot([i for i in range(1, epoch+1)], self.cost)
        plt.show()

    def predict(self, x):
        return 1/(1 + e**(-(sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0])))

    def predictList(self, X):
        result = []
        for x in X:
            result.append(self.predict(x))
        return result

    def cost_function(self, X, Y):
        cost = (self.learning_rate/(2*len(X)))*sum([(self.predict(X[i]) - Y[i])**2 for i in range(len(Y))])
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])
