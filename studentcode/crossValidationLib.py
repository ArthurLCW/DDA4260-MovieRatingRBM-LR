import numpy as np

def genAllData(data, k):
    ### notice:
    # 1. choose 1/10 data to be left-out validation data
    # 2. the rest data do k-fold validation
    allData = np.array_split(data, 10)
    leftOutVal = allData.copy()[0]
    allData.pop(0)
    KFoldData = np.concatenate(allData, axis=0)
    return leftOutVal, np.array_split(KFoldData, k)

def genKthTrainVal(dataList:list, kth):
    # kth start from 0
    dataVal = dataList[kth]
    dataTrainList = dataList.copy()
    dataTrainList.pop(kth)
    dataTrain = np.concatenate(dataTrainList, axis=0)
    return dataTrain, dataVal

def getLeftOutPrediction(listOfList, k_folds):
    ret = []
    numOfRaings = len(listOfList[0])
    for i in range(numOfRaings):
        temp = 0
        for j in range(k_folds):
            temp += listOfList[j][i]/k_folds
        ret.append(temp)
    return ret


