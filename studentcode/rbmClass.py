import numpy as np
import rbm
import projectLib as lib
import time

class rbmModel():
    def __init__(self, training, validation, K, F, epochs, gradientLearningRate, momentum_coe, reg_coe, early_stop_flag,
                 decrease_lr_flag):
        ''':param
            K: num of rating ranks = 5
            F: num of hidden layers
        '''
        ### parameters initialization ###
        self.training = training
        self.validation = validation
        self.K = K
        self.F = F
        self.epochs = epochs
        self.gradientLearningRate = gradientLearningRate
        self.momentum_coe = momentum_coe
        self.reg_coe = reg_coe
        self.early_stop_flag = early_stop_flag
        self.decrease_lr_flag = decrease_lr_flag


        ### calculation initialization
        self.momentum_val = np.zeros([100, self.F, self.K])
        self.trStats = lib.getUsefulStats(self.training)
        self.vlStats = lib.getUsefulStats(self.validation)
        self.W = rbm.getInitialWeights(self.trStats["n_movies"], self.F, self.K)
        self.b_h = rbm.getInitialBiasHidden(self.F)
        self.b_v = rbm.getInitialBiasVisible(self.trStats["n_movies"], self.K)
        self.grad = np.zeros(self.W.shape)
        self.grad_bh = np.zeros(self.b_h.shape)
        self.grad_bv = np.zeros(self.b_v.shape)
        self.momentum_val_bh = np.zeros(self.b_h.shape)
        self.momentum_val_bv = np.zeros(self.b_v.shape)
        self.posprods = np.zeros(self.W.shape)
        self.negprods = np.zeros(self.W.shape)

        self.epoch_list = []
        self.training_loss_list = []
        self.validation_loss_list = []
        self.training_time_list = []

        self.trainingNow = True
        self.epoch = 0


    def trainEpoch(self):
        # if not self.trainingNow:
        #     return

        self.epoch += 1
        time_start = time.time()
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(self.trStats["u_users"])
        np.random.shuffle(visitingOrder)  # my: get an array with 0-299 (random order)

        if self.trainingNow:
            for user in visitingOrder:
                # get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, self.training)  # my: all seen movies and ratings, for user

                # build the visible input
                v = rbm.getV(ratingsForUser)  # m*5 matrix

                # get the weights associated to movies the user has seen
                weightsForUser = self.W[ratingsForUser[:, 0], :, :]  # my: W with only the info about rated movie
                bvForUser = self.b_v[ratingsForUser[:, 0], :]  # my: bv with only seen part m*5

                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, self.b_h)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                self.posprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(v, posHiddenProb)

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser, bvForUser)
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, self.b_h)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                self.negprods[ratingsForUser[:, 0], :, :] = rbm.probProduct(negData, negHiddenProb)

                # we average over the number of users in the batch (if we use mini-batch)
                self.grad[ratingsForUser[:, 0], :, :] = self.gradientLearningRate * (
                            self.posprods[ratingsForUser[:, 0], :, :] - self.negprods[ratingsForUser[:, 0], :, :])

                # gradient descent for bias
                self.grad_bv[ratingsForUser[:, 0], :] = v - negData
                self.grad_bh = posHiddenProb - negHiddenProb

                # my: extensions
                # momentum
                if self.momentum_coe != 0:
                    self.momentum_val[ratingsForUser[:, 0], :, :] = self.momentum_coe * self.momentum_val[ratingsForUser[:, 0], :, :] + self.grad[ratingsForUser[:, 0], :, :]
                    self.grad[ratingsForUser[:, 0], :, :] = self.momentum_val[ratingsForUser[:, 0], :, :]  # notice: =

                    self.momentum_val_bh = self.momentum_coe * self.momentum_val_bh + self.grad_bh
                    self.grad_bh = self.momentum_val_bh
                    self.momentum_val_bv[ratingsForUser[:, 0], :] = self.momentum_coe * self.momentum_val_bv[ratingsForUser[:, 0], :]
                    self.grad_bv[ratingsForUser[:, 0], :] = self.momentum_val_bv[ratingsForUser[:, 0], :]

                # regularization
                if self.reg_coe != 0:  # put this after other changes to W
                    self.grad[ratingsForUser[:, 0], :, :] += - self.gradientLearningRate * self.reg_coe * self.W[ratingsForUser[:, 0], :, :]

                self.W[ratingsForUser[:, 0], :, :] += self.grad[ratingsForUser[:, 0], :, :]
                self.b_v[ratingsForUser[:, 0], :] += self.gradientLearningRate * self.grad_bv[ratingsForUser[:, 0], :]
                self.b_h += self.gradientLearningRate * self.grad_bh

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set
        tr_r_hat = rbm.predict(self.trStats["movies"], self.trStats["users"], self.W, self.training, self.b_h, self.b_v)  ###
        trRMSE = lib.rmse(self.trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(self.vlStats["movies"], self.vlStats["users"], self.W, self.validation, self.b_h, self.b_v)
        vlRMSE = lib.rmse(self.vlStats["ratings"], vl_r_hat)

        time_end = time.time()
        # print("### EPOCH %d ###" % self.epoch)
        # print("Training loss = %f" % trRMSE)
        # print("Validation loss = %f" % vlRMSE)
        # print("Training time = %f" % (time_end - time_start))

        self.training_loss_list.append(trRMSE)
        self.validation_loss_list.append(vlRMSE)
        self.epoch_list.append(self.epoch)
        self.training_time_list.append(time_start-time_end)

        # my: decrease lr
        if self.decrease_lr_flag:
            self.gradientLearningRate = self.gradientLearningRate * lib.decrease_lr(self.validation_loss_list, 6, 2, 0.03, 0.1)

        # my: early stop and
        if self.early_stop_flag:
            if lib.early_stop(self.validation_loss_list, 8, 2, 0.01):
                self.trainingNow = False

        return trRMSE, vlRMSE



    def validate(self, validationData, calRMSE=False):
        # used for left-out validation
        vlStatLocal = lib.getUsefulStats(validationData)
        vl_r_hat = rbm.predict(vlStatLocal["movies"], vlStatLocal["users"], self.W, validationData, self.b_h,
                               self.b_v)
        vlRMSE = 0
        if calRMSE:
            vlRMSE = lib.rmse(vlStatLocal["ratings"], vl_r_hat)
        return vl_r_hat, vlRMSE



