import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import math
import time

def create_matrix(df, user_list):
    x = np.zeros((10000, 10000))
    for i in range(10000):
        mask = df['user_id'].isin(user_list[i])
        filtered_df = df.loc[mask]
        movie_id = filtered_df['movie_id'].values.tolist()
        movie_rating = filtered_df['ratings'].values.tolist()
        for j in range(len(movie_id)):
            x[i][movie_id[j] - 1] = movie_rating[j]
    return x

def score(x):
    cos_sim = cosine_similarity(x)
    A = np.copy(x)
    A[A > 0 ] = 1
    prediction = np.dot(cos_sim, x)
    prediction = prediction / np.dot(cos_sim, A)
    return prediction, A

def calc_rmse(prediction, target):
    sum = 0
    test_size = 0
    for i in range(10000):
        for j in range(10000):
            if(target[i][j] != 0):
                sum += (target[i][j] - prediction[i][j])**2
                test_size += 1
    return math.sqrt(sum/test_size)

def target_func(x, A, U, V, lmd):
    j = 0.5 * ( np.linalg.norm(A * (x - U @ V.T ))**2 + lmd * ( np.linalg.norm(U)**2 ) + lmd * ( np.linalg.norm(V)**2 ) )
    return j

def get_NMF(x, k):
    model = NMF(n_components=k, init='random', random_state=0 , max_iter=1000000)
    U = model.fit_transform(x)
    V = model.components_
    return U, V.T

def conv_algo(x, A, U, V, lmd, alpha):
    i = 0
    j_list = []
    j_list.append(target_func(x, A, U, V, lmd))
    while(i < 500):
        print(i)
        partial_u = (A * (U @ V.T - x) ) @ V  + 2 * lmd * U
        partial_v =  np.transpose( A * ( U @ V.T - x) ) @ U + 2 * lmd * V
        U = U - alpha * partial_u
        V = V - alpha * partial_v
        j_list.append(target_func(x, A, U, V, lmd))
        i = i + 1
    return U @ V.T, j_list

def ex1():
    #####TRAIN
    user_df = pd.read_csv(r"/Users/massimo/Desktop/BDI/HW2/Project2-data/users.txt", header=None)
    user_list = user_df.values.tolist()
    netflix_df = pd.read_csv(r"/Users/massimo/Desktop/BDI/HW2/Project2-data/netflix_train.txt", sep=" ", header=None)
    netflix_df.columns = ["user_id", "movie_id", "ratings", "rating_date"]
    train_x = create_matrix(netflix_df, user_list)
    #####TEST
    netflix_df = pd.read_csv(r"/Users/massimo/Desktop/BDI/HW2/Project2-data/netflix_test.txt", sep=" ", header=None)
    netflix_df.columns = ["user_id", "movie_id", "ratings", "rating_date"]
    test_x = create_matrix(netflix_df, user_list)
    return train_x, test_x

def ex2(train_x, test_x):
    predicted, A = score(train_x)
    return calc_rmse(predicted, test_x), A  

def ex3(train_x, A, U, V, lmd, alpha):
    start_func = time.time()
    prediction, j_list= conv_algo(train_x, A, U, V, lmd, alpha)
    print(time.time() - start_func)
    return prediction, j_list


start_time = time.time()

train_x, test_x = ex1()
rmse, A = ex2(train_x, test_x)
print(rmse)
'''
np.savetxt("train.txt", train_x, fmt = '%i', delimiter= ' ')
np.savetxt("test.txt", test_x, fmt = '%i', delimiter = ' ')
np.savetxt("A.txt", A, fmt = '%i', delimiter = ' ')

df = pd.read_csv(r"train.txt", sep=" ", header=None)
train_x = df.to_numpy()
df = pd.read_csv(r"test.txt", sep=" ", header=None)
test_x = df.to_numpy()

df = pd.read_csv(r"A.txt", sep=" ", header=None)
A = df.to_numpy()
'''
###k≠50
rmse_list = []

U, V = get_NMF(train_x, 50)

prediction, j_list = ex3(train_x, A, U, V, 0.01, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
plt.plot(j_list)
plt.ylabel('Target function value')
plt.xlabel('alpha = 1e-5, k = 50, lambda = 0.01')
plt.show()


prediction, l_list= ex3(train_x, A, U, V, 0.001, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))

prediction, j_list = ex3(train_x, A, U, V, 1, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
### K = 10

U, V = get_NMF(train_x, 10)

prediction, j_list= ex3(train_x, A, U, V, 0.001, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
prediction, j_list = ex3(train_x, A, U, V, 0.01, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
prediction, j_list = ex3(train_x, A, U, V, 1, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))


###K = 100
U, V = get_NMF(train_x, 100)

prediction, j_list= ex3(train_x, A, U, V, 0.001, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
prediction, j_list = ex3(train_x, A, U, V, 0.01, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
prediction, j_list = ex3(train_x, A, U, V, 1, 0.00001)
rmse_list.append(calc_rmse(prediction, test_x))
print(rmse_list)

print(time.time() - start_time)