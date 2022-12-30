from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
import sys, csv, math

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

class Regression_Model:
    def __init__(self):
        self.weight = [0]
        self.cost = []
        self.learning_rate = 0
        self.default_file_name = "model_weights.txt"

    def print_weights(self):
        print("weights: ", self.weight)

    def save_weights(self, fileDir = None):
        if (fileDir == None):
            fileDir = self.default_file_name

        with open(fileDir, 'a') as f:
            f.write(str(self.weight))
            f.close()

    def load_weights(self, fileDir = None):
        if (fileDir == None):
            fileDir = self.default_file_name

        with open(fileDir, 'r') as f:
            raw_line = f.readline()
            self.weight = [float(x) for x in raw_line.strip("[] ").split(",")]
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

    def printModelPerformance(self, epoch = None, idx = None):
        if epoch !=  None:
            plt.plot([i for i in range(1, epoch+1)], self.cost)
            plt.show()
        
        if idx != None:
            arr_y = [self.predict(x) for x in self.x]
            arr_x = [x[idx] for x in self.x]

            plt.plot(arr_x, arr_y)
            plt.show()
            # print("\nx: ", arr_x, "  \ny: ", arr_y)

    def printDatasetOutput(self):
        return [self.predictList(self.x)]


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

    def classify(self, X):
        result_list = []
        if type(X) == int:
            X = [X]

        for x in X:
            if x < 0.5:
                result_list.append('0')
            else:
                result_list.append('1')
        return result_list

    def cost_function(self, X, Y):
        cost = -(1/len(X)) * sum( [Y[i]*math.log(self.predict(X[i]), 10) + (1 - Y[i])*math.log(1-self.predict(X[i]), 10) for i in range(len(Y))])
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])

class SVM(Regression_Model):
    def train(self, epoch, batch_size, learning_rate, reg_param):
        self.reg = reg_param
        return super().train(epoch, batch_size, learning_rate)

    def predict(self, x):
        result = sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]
        return result
    
    def classify(self, X):
        result_list = []
        if type(X) == int:
            X = [X]

        for x in X:
            if self.predict(x) < 0:
                result_list.append(-1)
            else:
                result_list.append(1)
        return result_list

    def hinge_loss(self, x, y):
        return max(0, 1 - y*self.predict(x))

    def cost_function(self, X, Y):
        cost = 1/len(X) * sum([(abs(sum([self.weight[i+1]**2 for i in range(self.parameter_count)])))/2 + self.reg*self.hinge_loss(X[n], Y[n]) for n in range(len(X))])
        self.cost += [cost]
        return cost

    def gradient_descent(self, X, Y):
        # delta_weight = 1/len(X) * sum([self.weight if self.hinge_loss(X[n], Y[n]) == 0 else [self.weight[i+1]-X[n][i]*(self.reg*Y[n]) for i in range(self.parameter_count)] for n in range(len(X))])
        delta_weight =  [0 for i in range(self.parameter_count+1)]
        for n in range(len(X)):
            if self.hinge_loss(X[n], Y[n]) == 0:
                for i in range(self.parameter_count + 1):
                    delta_weight[i] += self.weight[i]
            else:
                for i in range(self.parameter_count):
                    delta_weight[i+1] += self.weight[i+1]-X[n][i]*(self.reg*Y[n])
                delta_weight[0] += self.weight[0]-1*(self.reg*Y[n])
        
        for i in range(len(delta_weight)):
            delta_weight[i] *= 1/len(X) 

        for j in range(self.parameter_count+1):
            self.weight[j] -= self.learning_rate*delta_weight[j]


svm = SVM()

x = []
y = []

for i in range(100):
    y.append(-1)
    y.append(1)
    x.append([i, i+1])
    x.append([i, i+5])

svm.dataset(x, y)
svm.train(1000, 32, 0.001, 0.1)
svm.print_weights()
svm.printModelPerformance()
print(svm.classify([[1, 0], [1, 2], [-1, 0], [100, 99], [-1, -2]]))

# y_arr = []
# x_arr = []
# category = []

# test_x = []
# test_y = []
# test_idx = 0

# with open("candy-data.csv") as f:
#     file = csv.reader(f)

#     for i, row in enumerate(file):
#         if i == 0:
#             category = row[1:10]
#             print("category: ", category)
#             continue

#         sub_list = list(map(float, row[10:-1])) #10:-1
#         sub_list.append(float(row[-1])/100)

#         test_x.append(sub_list[test_idx])

#         x_arr.append(sub_list)
#         y_arr.append(int(row[9]))

#     test_y = y_arr

# # plt.scatter(test_x, test_y, color = "orange")

# MLR = Logistic_Regression()
# MLR.dataset(x_arr, y_arr)
# MLR.train(3000, 32, 0.01)
# MLR.save_weights()
# # MLR.load_weights()
# # MLR.print_weights()
# MLR.printModelPerformance()

# print("\nTest sample: ", x_arr[1])
# # print("\nMultiple LR: ", MLR.categorise(MLR.predictList([x_arr[1], x_arr[2], x_arr[3], x_arr[4]])))
# print("\n", [MLR.categorise(MLR.predictList(MLR.x))])
