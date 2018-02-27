import numpy as np
import matplotlib.pyplot as plt
import math
import sys

class LogisticRegression:
    
    def __init__(self, input_file, output_file, threshold=0.0001, max_iters = 100000):
        self.x = self.ReadLinFile(input_file)
        self.y = self.ReadLinFile(output_file)
        self.num_examples = self.x.shape[0]
        self.theta = np.zeros(3)
        self.threshold = threshold
        self.max_iterations = max_iters
    
    #function to read file with multiple features
    def ReadLinFile(self, file_name):
        fin = open(file_name, 'r')
        data = []
        for inp in fin:
            data.append(list(map(float,inp[:-1].split(","))))
        return np.array(data)
    
    #function to normalize data
    def NormalizeData(self):
        self.x = self.x.T
        mu1 = np.mean(self.x[0])
        mu2 = np.mean(self.x[1])
        sigma1 = np.std(self.x[0])
        sigma2 = np.std(self.x[1])
        self.x[0] = (self.x[0]-mu1)/sigma1
        self.x[1] = (self.x[1]-mu2)/sigma2
        self.x = self.x.T
    
    #function for minimizing log likelihood using newton's method
    def NewtonMethod(self):
        converged = False
        iter = 0
        while((not converged) and iter < self.max_iterations):
            x = np.c_[np.ones((self.num_examples,1)),self.x]
            thetaT_x = np.dot(x,self.theta)
            g_thetaT_x = np.array([1/(1+math.exp(-x)) for x in thetaT_x])
            delTheta = np.dot(x.T,np.c_[g_thetaT_x.T]-self.y)                   #gradient vector
            hessian = x.T.dot(np.diag(((1-g_thetaT_x)*g_thetaT_x))).dot(x)      #hessian matrix
            #newton's method update
            update =  np.dot(np.linalg.inv(hessian),delTheta)
            self.theta = self.theta - update.T[0]
            #check convergence
            converged = np.linalg.norm(update) < self.threshold
            iter+=1
        
        return self.theta
    
    
def plotBinaryClasses(theta, lr):
    
    class0x1, class0x2 = lr.x[np.where(lr.y==0)[0]].T
    class1x1, class1x2 = lr.x[np.where(lr.y==1)[0]].T
    
    plt.scatter(class0x1, class0x2, s=10, label="0")
    plt.scatter(class1x1, class1x2, s=10, label="1")
    x=np.linspace(-3,3)
    plt.plot(x, eval(str(theta[0]/(-theta[2])) +"+"+ str(theta[1]/(-theta[2]))+"*x"))
    plt.legend()
    plt.title("LogisticRegression")
    plt.show()

if __name__=='__main__':
    
    #create a logistic regression object
    if(len(sys.argv)==3):
        lr = LogisticRegression(sys.argv[1],sys.argv[2])
    else:
        lr = LogisticRegression("logisticX.csv","logisticY.csv")
    
    #normalize data to 0 mean and 1 standard deviation
    lr.NormalizeData()
    
    #compute parameters using newton's method
    theta = lr.NewtonMethod()
    print(str(theta[0])+" + "+str(theta[1])+"x1 + "+str(theta[2])+"x2 = 0")
    
    #plot decision boundary along with input data
    plotBinaryClasses(theta, lr)
    