import numpy as np
import projectLib as lib
import matplotlib.pyplot as plt

###########################

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((training["n_ratings"], training["n_movies"] + training["n_users"]))
    temp_mov = training["movies"]
    temp_usr = training["users"]
    for i in range(training["n_ratings"]):
        A[i][temp_mov[i]] = 1
        A[i][temp_usr[i]+training["n_movies"]] = 1
    return A

# we also get c
def getc(rBar, ratings):
    c = np.zeros(0)
    for i in ratings:
        c = np.append(c,i-rBar)
    return c

# apply the functions
A = getA(trStats)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    A_T = A.transpose()
    temp_1 = np.matmul(A_T,A)
    temp_2 = np.matmul(A_T,c)
    temp_3 = np.linalg.inv(temp_1)
    return np.matmul(temp_3,temp_2)

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    A_T = A.transpose()
    temp_1 = np.matmul(A_T,A)
    temp_2 = np.matmul(A_T,c)
    I = np.identity(np.linalg.matrix_rank(temp_1)+1)
    temp_3 = np.linalg.inv(temp_1-l*I)
    return np.matmul(temp_3,temp_2)

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)

# Regularised version
x_ = np.linspace(0,20,101)
x = x_[1:]
y = []
z = []
for l in x:
    b = param_reg(A, c, l)
    #print("Linear regression, l = %f" % l)
    rmse_1 = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
    z.append(rmse_1)
    #print("RMSE for training %f" % rmse_1)
    rmse_2 = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    #print("RMSE for validation %f" % rmse_2)
    y.append(rmse_2)
    
    
plt.plot(x, y, label = 'Validation')
plt.plot(x, z, label = 'Training')
plt.legend()
plt.show()
