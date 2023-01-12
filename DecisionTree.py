import numpy as np
import matplotlib.pyplot as plt
import sys, csv, math
import random

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
        self.cost = []
        self.default_file_name = "model_weights.txt"
        
    def dataset(self, X, Y):
        if (len(X) != len(Y)):
            raise Exception("Error: datasets are not of same size. Check XY input.")

        self.x = X #[np.array(x) for x in X]
        self.y = Y #[np.array(y) for y in Y]

        self.weight = [0]
        self.parameter_count = len(self.x[0])
        
        for _ in range(self.parameter_count):
            self.weight.append(1)
            
        self.size = len(Y)
        
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


class Binomial_Logistic_Regression(Regression_Model):
    def predict(self, x):
        result = 1/(1 + math.exp(-(sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]))) 
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

    def cost_function(self, X, Y): #BINARY CROSS ENTROPY
        cost = -(1/len(X)) * sum( [Y[i]*math.log(self.predict(X[i]), 10) + (1 - Y[i])*math.log(1-self.predict(X[i]), 10) for i in range(len(Y))])
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j+1] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])
        self.weight[0] -= (self.learning_rate/len(X))*sum([(self.predict(X[i]) - Y[i]) for i in range(len(Y))])


class Multinominal_Logistic_Regression(Regression_Model):
    def dataset(self, X, Y):
        if (len(X) != len(Y)):
            raise Exception("ERROR: datasets are not of same size. Check XY input.")

        self.size = len(Y)

        self.x = X
        self.y = Y

        self.parameter_count = len(self.x[0])
        self.class_count = len(self.y[0])
        
        self.weight = np.ones((self.parameter_count, self.class_count))
    
    def train(self, epoch, batch_size, learning_rate, reg_param):
        self.reg = reg_param
        return super().train(epoch, batch_size, learning_rate)

    def predict(self, x):
        p_vector = self.softmax_function(np.dot(x, self.weight))
        return p_vector

    def classify(self, X): #argmax function
        result_list = []
        if type(X) == int:
            X = [X]

        for x in X:
            label = self.argmax(self.predict(x))
            result_list.append(label)

        return result_list

    def softmax_function(self, p):
        if type(p) != np.ndarray:
            p = np.array(p)

        # #IMPT: max normalisation
     
        softmax_array = np.exp(p) / np.sum(np.exp(p))
        return softmax_array

    def argmax(self, s):
        return np.argmax(s)

    def cross_entropy(self, arr, labels): # -> (softmax probabilty array, labels)
        idx = labels.index(1) #index of true label
        return -1*math.log(arr[idx])

    def cost_function(self, X, Y): #CATAGORICAL CROSS ENTROPY
        p = [self.predict(x) for x in X] #softmax probabilities
        cost = 0

        for i in range(len(X)):
            cost += self.cross_entropy(p[i], Y[i]) #truth table (labels)

        cost /= len(X)
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        p = [self.predict(x) for x in X]

        delta_P = np.subtract(Y, p)
        trans_X = np.transpose(X)

        delta_weight = (1/len(X)) * np.dot(trans_X, delta_P) + 2*self.reg*self.weight
        self.weight += self.learning_rate*delta_weight

# MLR = Multinominal_Logistic_Regression()
# x = []
# y = []


# for j in range(100):
#     x.append([random.randint(-30, -1)])
#     y.append([1, 0])

#     x.append([random.randint(1, 30)])
#     y.append([0, 1])

# MLR.dataset(x, y)
# MLR.train(1000, 32, 0.001, 0.0001)

# MLR.print_weights()
# MLR.save_weights()
# MLR.printModelPerformance()
# print(MLR.classify([-1, 40, -20, 80, -15, 70]))

class SVM(Regression_Model): #HARD MARGIN VS SOFT MARGIN
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

# svm = SVM()
# x = []
# y = []

# for i in range(100):
#     y.append(-1)
#     y.append(1)
#     x.append([i, i+1, i])
#     x.append([i, i+5, i])

# svm.dataset(x, y)
# svm.train(1000, 32, 0.0001, 1000)
# svm.print_weights()
# svm.printModelPerformance()
# print(svm.classify([[1, -2, 0], [1, -20, 0], [-1, 15, 10], [100, 102, 10], [1, 6, 10]]))

class DecisionTree():

    def __init__(self):
        self.CLASSIFICATION = 1
        self.REGRESSION = 1

    def dataset(self, X, Y):
        if (len(X) != len(Y)):
            raise Exception("ERROR: datasets are not of same size. Check XY input.")

        self.size = len(Y)

        self.x = np.array(X) #[np.array(x) for x in X]
        self.y = np.array(Y) #[np.array(y) for y in Y]

        self.parameter_count = len(self.x[0])
        self.class_count = np.max(self.y) + 1 #len(self.y[0])
    
    def train(self, branchtype):
        pass

    def regressionSplit(self):
        pass
    
    def classificationSplit(self):
        xt = np.transpose(self.x)
        print(xt, end='\n\n')

        relative_gini_score = []
        for p in range(self.parameter_count):
            sub_class_count = np.max(xt[p]) + 1
            # print(sub_class_count, self.class_count)
            count = np.zeros((sub_class_count, self.class_count))
            # print(count)

            for i in range(self.size):
                iy = self.y[i]
                ix = xt[p][i]

                count[ix][iy] += 1
            
            print(count)


            # _, counts = np.unique(xt[i], return_counts=True)
            # pk = counts/np.sum(counts)

            # gini_score = np.sum(pk*(1-pk))

            # relative_gini_score.append(abs(gini_score - 0.5)) #Higher relative score indicates better split/purity

        # print(relative_gini_score, end='\n\n')
        # return np.argsort(relative_gini_score)[::-1]
    
dt = DecisionTree()
dt.dataset([[0, 0, 0], [0, 1, 1], [1, 2, 0]], [0, 0, 1])
print(dt.classificationSplit())
