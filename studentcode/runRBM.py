from rbmClass import rbmModel
import projectLib as lib
import numpy as np
import crossValidationLib
import rbm
# for rbm with k-folds

training = lib.getTrainingData()
validation = lib.getValidationData()

### hyper-parameters ###
K = 5
F = 50
epochs = 30
gradientLearningRate = 0.05
# momentum
momentum_coe = 0.5 # need to be modified
momentum_val = np.zeros([100, F, K])
# regularization
reg_coe = 0.01 # regularization coefficient
B = 1
# early_stop
early_stop_flag = True
# decrease lr
decrease_lr_flag = True


# data for cross validation
k_folds = 5 # 1 for left validation, others for cross valdation
totalData = np.concatenate((training, validation), axis=0)
np.random.shuffle(totalData)
listInstance = []

temp0 = crossValidationLib.genAllData(totalData, k_folds)
leftOutVal = temp0[0]
kFoldList = temp0[1]

# init rbm instances
for i in range(k_folds):
    TrainValTemp = crossValidationLib.genKthTrainVal(kFoldList, i)
    trainingTemp = TrainValTemp[0]
    validationTemp = TrainValTemp[1]
    listInstance.append(rbmModel(training=trainingTemp, validation=validationTemp, K=K, F=F, epochs=epochs,
                        gradientLearningRate=gradientLearningRate, momentum_coe=momentum_coe, reg_coe=reg_coe,
                        early_stop_flag=early_stop_flag, decrease_lr_flag=decrease_lr_flag))


# rbmInstance0 = rbmModel(training=training, validation=validation, K=K, F=F, epochs=epochs,
#                         gradientLearningRate=gradientLearningRate, momentum_coe=momentum_coe, reg_coe=reg_coe,
#                         early_stop_flag=early_stop_flag, decrease_lr_flag=decrease_lr_flag)

avgTrainLoss, avgValLoss, leftOutValLoss = [], [], []
for epoch in range(1, epochs+1):
    avgTrainLoss.append(0)
    avgValLoss.append(0)
    leftOutValLoss.append(0)
    print("epoch in total: ", epoch)
    listOfLeftOutResultsInKModels = []

    for instance in range(k_folds):
        # print("Instance: ", instance)
        tempResult = listInstance[instance].trainEpoch()
        avgTrainLoss[epoch-1] += tempResult[0]/k_folds
        avgValLoss[epoch-1] += tempResult[1]/k_folds
        # leftOutValLoss[epoch-1] += listInstance[instance].validate(leftOutVal)/k_folds
        # leftOutValResult += listInstance[instance].validate(leftOutVal, True)[0]/k_folds
        listOfLeftOutResultsInKModels.append(listInstance[instance].validate(leftOutVal, False)[0])

    # calculate left-out validation result.
    vlStatLocal = lib.getUsefulStats(leftOutVal)
    leftOutValResult = crossValidationLib.getLeftOutPrediction(listOfLeftOutResultsInKModels, k_folds)
    # print(leftOutValResult)
    # print(vlStatLocal["ratings"])
    LeftOutVLRMSE = lib.rmse(vlStatLocal["ratings"], leftOutValResult)
    leftOutValLoss[epoch-1] = LeftOutVLRMSE


    print('AVG train RMSE: ', avgTrainLoss[epoch-1])
    print('AVG val RMSE: ', avgValLoss[epoch - 1])
    print('left-out val RMSE: ', leftOutValLoss[epoch - 1])
    print(' ')



# lib.draw_loss(avgTrainLoss, avgValLoss, listInstance[0].epoch_list, leftOutValLoss)
lib.draw_loss(avgTrainLoss, avgValLoss, listInstance[0].epoch_list, F, gradientLearningRate, momentum_coe,
              reg_coe, B, early_stop_flag, decrease_lr_flag, k_folds, leftOutValLoss)

# # output text:
# predictedRatings = 0
# trStats = lib.getUsefulStats(training)
# for i in range(k_folds):
#     predictedRatings += np.array([rbm.predictForUser(user, listInstance[i].W, training, listInstance[i].b_h,
#                                                      listInstance[i].b_v) for user in trStats["u_users"]])/k_folds
#
# np.savetxt("predictedRatings.txt", predictedRatings)



