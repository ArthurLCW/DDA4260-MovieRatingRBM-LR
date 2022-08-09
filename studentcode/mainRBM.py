import numpy as np
import rbm
import projectLib as lib
import time

trainingOld = lib.getTrainingData() # original: no old
validationOld = lib.getValidationData()

##### remastered, shuffle data
totalData = np.concatenate((trainingOld, trainingOld), axis=0)
np.random.shuffle(totalData)
allData = np.array_split(totalData, 10)
leftOutVal = allData.copy()[0]
allData.pop(0)
KFoldData = np.concatenate(allData, axis=0)
training = KFoldData
validation = leftOutVal


# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
F = 50
epochs = 250
gradientLearningRate = 0.05

### my hyper-parameter for extensions ###
# momentum
momentum_coe = 0.5 # need to be modified
momentum_val = np.zeros([100, F, K])

# regularization
reg_coe = 0.01 # regularization coefficient

# mini-batch size
B = 1

# early_stop
early_stop_flag = True

# decrease lr
decrease_lr_flag = True



# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
b_h = rbm.getInitialBiasHidden(F)
b_v = rbm.getInitialBiasVisible(trStats["n_movies"], K)
grad = np.zeros(W.shape)
grad_bh = np.zeros(b_h.shape)
grad_bv = np.zeros(b_v.shape)
momentum_val_bh = np.zeros(b_h.shape)
momentum_val_bv = np.zeros(b_v.shape)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)

# my preparation for graph drawing
epoch_list = []
training_loss_list = []
validation_loss_list = []


for epoch in range(1, epochs + 1):
    time_start = time.time()
    # in each epoch, we'll visit all users in a random order
    visitingOrder = np.array(trStats["u_users"])
    np.random.shuffle(visitingOrder) # my: get an array with 0-299 (random order)

    counting_num = 0 # used for mini-batch

    for user in visitingOrder:
        counting_num += 1

        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training) # my: all seen movies and ratings, for user

        # build the visible input
        v = rbm.getV(ratingsForUser) # m*5 matrix

        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :] # my: W with only the info about rated movie
        bvForUser = b_v[ratingsForUser[:, 0], :] # my: bv with only seen part m*5

        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, b_h)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser, bvForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, b_h)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad[ratingsForUser[:, 0], :, :] = gradientLearningRate * (posprods[ratingsForUser[:, 0], :, :] -
                                                                   negprods[ratingsForUser[:, 0], :, :])

        # gradient descent for bias
        grad_bv[ratingsForUser[:, 0], :] = v - negData
        grad_bh = posHiddenProb - negHiddenProb

        # my: extensions
        # momentum
        if momentum_coe != 0:
            momentum_val[ratingsForUser[:, 0], :, :] = momentum_coe * momentum_val[ratingsForUser[:, 0], :, :] + \
                                                       grad[ratingsForUser[:, 0], :, :]
            grad[ratingsForUser[:, 0], :, :] = momentum_val[ratingsForUser[:, 0], :, :] # notice: =

            momentum_val_bh = momentum_coe * momentum_val_bh + grad_bh
            grad_bh = momentum_val_bh
            momentum_val_bv[ratingsForUser[:, 0], :] = momentum_coe * momentum_val_bv[ratingsForUser[:, 0], :]
            grad_bv[ratingsForUser[:, 0], :] = momentum_val_bv[ratingsForUser[:, 0], :]

        # regularization
        if reg_coe != 0: # put this after other changes to W
            grad[ratingsForUser[:, 0], :, :] += - gradientLearningRate * reg_coe * W[ratingsForUser[:, 0], :, :]


        if counting_num == B:
            W[ratingsForUser[:, 0], :, :] += grad[ratingsForUser[:, 0], :, :] # / B
            b_v[ratingsForUser[:, 0], :] += gradientLearningRate * grad_bv[ratingsForUser[:, 0], :] # / B
            b_h += gradientLearningRate * grad_bh # / B
            counting_num = 0


    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training, b_h, b_v) ###
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, validation, b_h, b_v)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    time_end = time.time()
    print("### EPOCH %d ###" % epoch)
    print("Training loss = %f" % trRMSE)
    print("Validation loss = %f" % vlRMSE)
    print("Training time = %f" % (time_end-time_start))

    training_loss_list.append(trRMSE)
    validation_loss_list.append(vlRMSE)
    epoch_list.append(epoch)

    # my: decrease lr
    if decrease_lr_flag:
        gradientLearningRate = gradientLearningRate * lib.decrease_lr(validation_loss_list, 6, 2, 0.03, 0.1)

    # my: early stop and
    if early_stop_flag:
        if lib.early_stop(validation_loss_list, 8, 2, 0.01):
            break

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example
# predictedRatings = np.array([rbm.predictForUser(user, W, training, b_h, b_v) for user in trStats["u_users"]])
# np.savetxt("predictedRatingsOldMethod.txt", predictedRatings)

# my: draw graph
lib.draw_loss(training_loss_list, validation_loss_list, epoch_list, F, gradientLearningRate, momentum_coe,
              reg_coe, B, early_stop_flag, decrease_lr_flag)