import matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

def getTrainingData():
    return np.genfromtxt("training.csv", delimiter=",", dtype=np.int)

def getValidationData():
    return np.genfromtxt("validation.csv", delimiter=",", dtype=np.int)


def getUsefulStats(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()

    users = [x[1] for x in training]
    u_users = np.unique(users).tolist()

    return {
        "movies": movies, # movie IDs
        "u_movies": u_movies, # unique movie IDs
        "n_movies": len(u_movies), # number of unique movies

        "users": users, # user IDs
        "u_users": u_users, # unique user IDs
        "n_users": len(u_users), # number of unique users

        "ratings": [x[2] for x in training], # ratings
        "n_ratings": len(training) # number of ratings
    }

def getRatingsForUser(user, training):
    # user is a user ID
    # training is the training set
    # ret is a matrix, each row is [m, r] where
    #   m is the movie ID
    #   r is the rating, 1, 2, 3, 4 or 5
    return np.array([[x[0], x[2]] for x in training if x[1] == user])

# RMSE function to tune your algorithm
def rmse(r, r_hat):
    r = np.array(r)
    r_hat = np.array(r_hat)
    return np.linalg.norm(r - r_hat) / np.sqrt(len(r))


# my: draw graph
def draw_loss(loss_training: list, loss_validation: list, epochs: list, F: int, gradientLearningRate: float,
              momentum_coe: float, reg_coe: float, B: float, early_stop_flag: bool, decrease_lr_flag: bool,
              k_folds: int = 0, anotherVal: list = []):
    palette = plt.get_cmap('Set1')
    plt.plot(epochs, loss_training, color=palette(3), marker='*', label='Training RMSE')
    plt.plot(epochs, loss_validation, color=palette(1), marker='*', label='Validation RMSE')
    graph_title = "F_" + str(F) + ", lr_" + str(gradientLearningRate) + ", momCoe_" + str(momentum_coe) \
                  + ", regCoe_" + str(reg_coe) + ", B_" + str(B) + ", earlyStop_" + str(early_stop_flag) + \
                  ", decreaseLr_" + str(decrease_lr_flag)
    if len(anotherVal) != 0:
        plt.plot(epochs, anotherVal, color=palette(2), marker='*', label='left out Validation RMSE')
        plt.text(epochs[-1], anotherVal[-1], (epochs[-1], anotherVal[-1]))
        graph_title += ",kFolds_" + str(k_folds)

    # plt.title(graph_title)
    plt.text(epochs[-1], loss_training[-1], (epochs[-1], loss_training[-1]))
    plt.text(epochs[-1], loss_validation[-1], (epochs[-1], loss_validation[-1]))
    plt.xlabel('epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig("./new_figures/" + graph_title + ".jpg")
    plt.show()



def early_stop(ref_list: list, ref_size: int, test_size: int, improve_threshold: int):
    # ref size: length of all number used.
    if (len(ref_list) < ref_size):
        return False

    tested_list = ref_list[-test_size:]
    pure_ref_list = ref_list[-ref_size:]
    tested_avg = np.mean(tested_list)
    pure_ref_list_avg = np.mean(pure_ref_list)
    # print(tested_list, ' ', pure_ref_list)
    # print(tested_avg, ' ', pure_ref_list_avg)

    if pure_ref_list_avg - tested_avg <= improve_threshold:
        print('early stop')
        return True
    else:
        return False

def decrease_lr(ref_list: list, ref_size: int, test_size: int, improve_threshold: int, decrease_rate: float):
    # ref size: length of all number used.
    if (len(ref_list) < ref_size):
        return 1

    tested_list = ref_list[-test_size:]
    pure_ref_list = ref_list[-ref_size:]
    tested_avg = np.mean(tested_list)
    pure_ref_list_avg = np.mean(pure_ref_list)
    # print(tested_list, ' ', pure_ref_list)
    # print(tested_avg, ' ', pure_ref_list_avg)

    if pure_ref_list_avg - tested_avg <= improve_threshold:
        print('decrease lr')
        return decrease_rate
    else:
        return 1


# draw_loss([1,2,3],[4,5,6],[1,2,3],1,1,1,1,1,True,True,5,[8,9,10])