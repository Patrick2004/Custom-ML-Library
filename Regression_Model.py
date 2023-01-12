import numpy as np
import matplotlib.pyplot as plt
import sys, csv, math

class Regression_Model:
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
        
        for e in range(epoch):
            for i in range(len(data_batch[0])):
                self.gradient_descent(data_batch[0][i], data_batch[1][i])
                
            cost = self.cost_function(self.x, self.y)
            print("epoch: {} (cost: {:.20f})".format(e+1, cost))

##        self.printModelPerformance(epoch)

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
        return 1/(1 + math.e**(-(sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0])))

    def cost_function(self, X, Y):
        cost = -(1/len(X)) * sum( [Y[i]*math.log(self.predict(X[i]), 10) + (1 - Y[i])*math.log(1-self.predict(X[i]), 10) for i in range(len(Y))])
        self.cost += [cost]
        return cost
    
    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count):
            self.weight[j] -= (self.learning_rate/len(X))*sum( [(self.predict(X[i]) - Y[i] )*X[i][j] for i in range(len(Y))])


class multi_Logistic_Regression(Regression_Model):

    def __init__(self):
        self.weight = []
        self.cost = []
        self.learning_rate = 0

    def dataset(self, x, y):
        self.x = x
        self.y = y

        self.parameter_count = len(self.x[0])
        self.classcount = len(self.y[0])

        for i in range(self.classcount):
            col = []
            for i in range(self.parameter_count):
                col.append(1)
            self.weight.append(col)

        print(self.weight)
        self.size = len(y)
    
    def train(self, epoch, batch_size, learning_rate, classes):
        self.classes = classes
        return super().train(epoch, batch_size, learning_rate)
    
    def predict(self, x):
        y_output = []
        for i in range(len(self.weight)):
            y_clas = 0
            for j in range(len(self.weight[i])):
                y_clas += self.weight[i][j]*x[i]
                print(self.weight[i][j])
            y_output.append(y_clas)
        return y_output

    def cost_function(self, X, Y):
        k = Y.index(1)
        for i in X:
            cost = self.predict(i)[k]
            self.cost += [cost]
            return cost

    def gradient_descent(self, X, Y):
        pass
##        for j in range(self.classes):
##            
##            self.weight[j] -= (self.learning_rate/len(X))*sum([X[k]*(Y[k] - self.predict(X[k])) for k in range(len(Y))])
        
                
            

class SVM_Algorithm(Regression_Model):
    def train(self, epoch, batch_size, learning_rate, reg_param):
        self.reg = reg_param
        return super().train(epoch, batch_size, learning_rate)
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
    
    def predict(self, x):
        return sum([self.weight[i+1]*x[i] for i in range(len(x))]) + self.weight[0]
        
    def cost_function(self, X, Y):
        cost = self.reg
        for i in range(len(Y)):
            if self.predict(X[i])*Y[i] >= 1:
                pass
            else:
                cost += 1 - self.predict(X[i])*Y[i]
        return cost

    def gradient_descent(self, X, Y):
        for j in range(self.parameter_count + 1):
            for i in range(len(Y)):
                if self.predict(X[i])*Y[i] >= 1:
                    self.weight[j] -= (self.learning_rate)*self.reg
                else:
                    self.weight[j] -= (self.learning_rate)*(self.predict(X[i])*Y[i] - self.reg)

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
            gini_index = 0
            total_number = self.size
            sub_class_count = np.max(xt[p]) + 1
            # print(sub_class_count, self.class_count)
            count = np.zeros((sub_class_count, self.class_count))
            # print(count)

            for i in range(self.size):
                iy = self.y[i]
                ix = xt[p][i]

                count[ix][iy] += 1
            
            print(count)
            
            for c in count:
                gini_index += (c[0]+c[1])/total_number * (1 - ((c[0]/(c[0]+c[1]))**2 + (c[1]/(c[0]+c[1]))**2))
            print(gini_index)        
            
        


            # _, counts = np.unique(xt[i], return_counts=True)
            # pk = counts/np.sum(counts)

            # gini_score = np.sum(pk*(1-pk))

            # relative_gini_score.append(abs(gini_score - 0.5)) #Higher relative score indicates better split/purity

        # print(relative_gini_score, end='\n\n')
        # return np.argsort(relative_gini_score)[::-1]
    
dt = DecisionTree()
dt.dataset([[0, 0, 0], [0, 1, 1], [1, 2, 0]], [0, 0, 1])
print(dt.classificationSplit())

##
##for i in range(100):
##    y.append(-1)
##    y.append(1)
##    x.append([i, i+1, 0])
##    x.append([i, i+5, 10])

##with open("candy-data.csv") as f:
##    file = csv.reader(f)
##    for i, row in enumerate(file):
##        if i == 0:
##            continue
##        
##        y.append((row[1:2]))
##        x.append(list(map(int, list(map(float, row[10:])))))

##SVM = SVM_Algorithm()
##SVM.dataset(x, y)
##SVM.train(5000, 5, 0.00000001, 10)
##print(SVM.weight)
###print("\nMultiple LR: ", SVM.predictList([x_arr[2], x_arr[6], x_arr[8]]))
##print(SVM.classify([[-10, 9, 10], [-5, 15, 10], [10, 20, 10], [30, 50, 10], [-10, -20, 0]]))

##mlr = multi_Logistic_Regression()
##mlr.dataset(x, y)
##mlr.train(1000, 10, 0.0001, 3)
##print(mlr.weight)
##print("\nMultiple LR: ", mlr.predictList([x_arr[2], x_arr[6], x_arr[8]]))


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







                

        
        
    
    
