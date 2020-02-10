 
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


# sigmoid
def Sigmoid(z):
    return 1/(1 + np.exp(-z))

# gradients
def Gradient(theta,x,y):
    m,n = x.shape
    h_x = Sigmoid(x.dot(theta))
    h_x = h_x.reshape((m,1))
    
    gradients = (1./m)*(np.matmul((x.T),(h_x - y)))
    return gradients.flatten()

# cost function
def CostFunc(theta,x,y):
    m,n = x.shape
    h_x = Sigmoid(np.matmul(x,theta))
    h_x = h_x.reshape((m,1))
    cost = (-1./m)*np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
    return(cost)
    
# main routine
if __name__=="__main__":
    
    # load data
    data = np.loadtxt(r'ex2data1.txt', delimiter=',')
    
    # get X
    X = data[:,[0,1]]
    # add a column of 1s (bias term)
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)

    # get y
    y = data[:,2].reshape(len(X),1)
    
    # initialize theta
    m , n = X.shape
    initial_theta = np.zeros(n)
    
    # get optimal theta using Scipy's optimization module
    Result = op.minimize(fun = CostFunc, 
                        x0 = initial_theta, 
                        args = (X, y),
                        method = 'TNC',
                        jac = Gradient)
    # get the result
    optimal_theta = Result.x
    
    # print theta
    print('theta is:',optimal_theta)
    
    # draw decision boundary
    boundary = [(-optimal_theta[0]-optimal_theta[1]*x)/optimal_theta[2] for x in X[:,1]]
    
    # accepted
    X_accepted = X[np.where(y==[1.])[0]]
    X_rejected = X[np.where(y==[0.])[0]]
    
    # plot decision boundary
    fig,ax = plt.subplots()
    ax.scatter(X_accepted[:,1],X_accepted[:,2],marker='+',color='black')
    ax.scatter(X_rejected[:,1],X_rejected[:,2],marker='o',color='y')
    ax.plot(X[:,1],boundary)
    plt.show()