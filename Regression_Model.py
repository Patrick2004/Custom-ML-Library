import numpy as np
import matplotlib.pyplot as plt
import sys, csv, math

class Regression_Model:
    def __init__(self):
        self.weight = [0]
        self.cost = []
        self.learning_rate = 0

    def print_weights(self):
        print(self.weight)

    def save_weights(self):
        f = open("saved_weights.txt", 'a')
        f.write(str(self.weight))
        f.close()
        
        
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
        
        for e in range(epoch):
            for i in range(len(data_batch[0])):
                self.gradient_descent(data_batch[0][i], data_batch[1][i])
                
            cost = self.cost_function(self.x, self.y)
        print("epoch: {} (cost: {:.20f})".format(e+1, cost))

        self.printModelPerformance(epoch)

    def predictList(self, X):
        result = []
        for x in X:
            result.append(self.predict(x))
        return result

    def setModelMetrics(self):
        pass

    def printModelPerformance(self, epoch):
        arr_y = [self.predict(x) for x in self.x]
        arr_x = self.x

        plt.plot([i for i in range(1, epoch+1)], self.cost)
        plt.show()


class Linear_Regression(Regression_Model):
    
    def predict(self, x):
        return sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]

    def cost_function(self, X, Y):
        cost = math.sqrt((1/(2*len(X)))*sum([(self.predict(X[i]) - Y[i])**2 for i in range(len(Y))]))
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])


class Logistic_Regression(Regression_Model):
    
    def predict(self, x):
        result = 1/(1 + math.e**(-(sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]))) 
        
        return result

    def categorise(self, result):
        cat_result = []
        for i in result:
            if i < 0.5:
                cat_result.append('0')
            else:
                cat_result.append('1')
        return cat_result

    def cost_function(self, X, Y):
        cost = -(1/len(X)) * sum( [Y[i]*math.log(self.predict(X[i]), 10) + (1 - Y[i])*math.log(1-self.predict(X[i]), 10) for i in range(len(Y))])
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])
        
y_arr = []
x_arr = []
category = []

with open("candy-data.csv") as f:
    file = csv.reader(f)
    for i, row in enumerate(file):
        if i == 0:
            category = row[1:10]
            print(category)
            continue
        l = list(map(float, row[10:-1]))
        l.append(float(row[-1])/100)
        y_arr.append(int(row[9]))
        x_arr.append(l)

MLR = Logistic_Regression()
##print(x_arr)
##print(y_arr)
MLR.dataset(x_arr, y_arr)
MLR.train(3000, 32, 0.01)
print("\nMultiple LR: ", MLR.categorise(MLR.predictList([x_arr[1], x_arr[2], x_arr[3], x_arr[4]])))
MLR.print_weights()
MLR.save_weights()

class Polynomial_Regression:
    def __init__(self):
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







                

        
        
    
    
