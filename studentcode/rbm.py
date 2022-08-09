import numpy as np
import projectLib as lib
import math

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    # lcw: indicate the num of ratings for each movie
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    # ret: m * 5 (m->the num of movies that this user has seen)
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K)) #####################
    # return np.zeros([m, F, K])

def getInitialBiasHidden(F):
    # F is the number of hidden units
    return np.random.normal(0, 0.1, (F)) #######################
    # return np.zeros(F)

def getInitialBiasVisible(m, K):
    # m is the number of visible units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, K)) ####################
    # return np.zeros([m, K])

def sig(x):
    ### TO IMPLEMENT ### finished
    # x is a real vector of size n lcw: x 1D npArray
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    ret = np.zeros(x.size)
    for i in range(x.size):
        ret[i] = 1/(1 + math.exp(-x[i]))
    return ret

def visibleToHiddenVec(v, w, b_h):
    ### TO IMPLEMENT ### finished. v is vector with all values. 5 0s if not rated.
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # b_h: vec F
    # ret should be a vector of size F
    F = w.shape[1]
    ret = np.zeros(F)

    for i in range(F):
        # ret[i] = np.multiply(w[:, i, :], v).sum()
        ret[i] = np.multiply(w[:, i, :], v).sum() + b_h[i]
    return sig(ret)

def hiddenToVisible(h, w, b_v):
    ### TO IMPLEMENT ### finished.
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # b_v is an array of size m x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    F = h.shape[0]
    m = w.shape[0]
    ret = np.zeros([m, 5])

    for i in range(m):
        for j in range(5):
            # ret[i][j] = np.multiply(w[i, :, j], h).sum()
            ret[i][j] = np.multiply(w[i, :, j], h).sum() + b_v[i][j]
    for i in range(m):
        ret[i] = softmax(ret[i])
    return ret

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i
    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq, b_h, b_vq):
    ### TO IMPLEMENT ### finished, but a little different.
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # b_h: vec F ###
    # b_vq: vec 5, index q of b_v (array m*5) ###

    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    ###### my ret: whole matrix m*5 (former)

    hidden_vec = visibleToHiddenVec(v, w, b_h)  # vector F
    sample_hidden_vec = sample(hidden_vec)  # vector F
    # negative_v = hiddenToVisible(sample_hidden_vec, w)  # m*5 matrix

    F = hidden_vec.shape[0]
    ret = np.zeros(5)
    for j in range(5):
        ret[j] = np.multiply(wq[:, j], sample_hidden_vec).sum() + b_vq[j]
    ret = softmax(ret)
    return ret

    # # learning
    # hidden_vec = visibleToHiddenVec(v, w) # vector F
    # positive_grad = probProduct(v, hidden_vec) # 3d array
    #
    # # unlearning
    # sample_hidden_vec = sample(hidden_vec) # vector F
    # negative_v = hiddenToVisible(sample_hidden_vec, w) # m*5 matrix
    #
    # negative_hidden_vec = visibleToHiddenVec(negative_v, w)
    # negative_grad = probProduct(negative_v, negative_hidden_vec)



def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ### finished
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability

    # my:
    # input: m*5 matrix; output: m vector
    # ret = np.zeros(ratingDistribution.shape[0])
    # for i in range(ratingDistribution.shape[0]):
    #     temp = ratingDistribution[i]
    #     max_val = np.max(temp)
    #     max_idx = np.where(max_val == temp)[0][0]
    #     ret[i] = max_idx + 1
    # return ret

    # my: input: vec 5, return: single num
    max_val = np.max(ratingDistribution)
    return np.where(max_val == ratingDistribution)[0][0] + 1


def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution

    # my:
    # input: m*5 matrix; output: m vector
    # ret = np.zeros(ratingDistribution.shape[0])
    # for i in range(ratingDistribution.shape[0]):
    #     temp = ratingDistribution[i]
    #     temp_val = 0
    #     for j in range(5):
    #         temp_val += temp[j]*(j+1)
    #     ret[i] = temp_val
    # return ret

    # my: input: vec 5, return: single num
    temp_val = 0
    for j in range(5):
        temp_val += ratingDistribution[j]*(j+1)
    return temp_val


def predictMovieForUser(q, user, W, training, b_h, b_v, predictType="exp"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    # b_h: vec F ###
    # b_v: (array m*5) ###

    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser) # m(num of movies user sees) * 5
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :], b_h, b_v[q, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, b_h, b_v, predictType="exp"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, b_h, b_v, predictType=predictType) for (movie, user) in list(zip(movies, users))]

def predictForUser(user, W, training, b_h, b_v, predictType="exp"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    # b_h: vec F ###
    # b_v: (array m*5) ###
    # my: ret: vector m, indicates all ratings
    m = W.shape[0]
    ret = np.zeros(m)
    for i in range(m):
        ret[i] = predictMovieForUser(i, user, W, training, b_h, b_v, predictType)
    return ret
