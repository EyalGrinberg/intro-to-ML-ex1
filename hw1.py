import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

#loading data
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
"""
#defining training set and test set of the images
import numpy.random
idx = numpy.random.RandomState(0).choice(70000, 11000) 
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def kNN_algo(train_set_images, train_labels_images, query_image, k):
    #creating an array with the distance of each image from query
    dists_from_query = np.zeros(len(train_set_images))
    for i in range(len(train_set_images)):
        dists_from_query[i] = np.linalg.norm(train_set_images[i] - query_image)
    #creating labels counting array in length 10
    labels_cnt_arr = np.zeros(10)
    for i in range(k):
        idx_of_nearest_neighbor = np.argmin(dists_from_query) #find index of nearest neighbor
        labels_cnt_arr[int(train_labels_images[idx_of_nearest_neighbor])] += 1 #add 1 to the labels counter
        dists_from_query[idx_of_nearest_neighbor] = np.Inf #don't take the same min dist anymore
    return np.argmax(labels_cnt_arr) #the most common label

def accuracy(n, k):
    #take only n samples from train set
    small_train_set = train[:n]
    small_train_labels = train_labels[:n]
    success_cnt = 0
    for i in range(len(test)):
        success_cnt += kNN_algo(small_train_set, small_train_labels, test[i], k) == int(test_labels[i]) # =1 if prediction was good, else =0
    return success_cnt / 1000

# item (c)
k_vec = np.arange(1, 101)
accuracy_vec = np.zeros(100)
for k in k_vec:
    print(k)
    accuracy_vec[k - 1] = accuracy(1000, k)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.plot(k_vec, accuracy_vec)
plt.show()

# item (d)
accuracy_vec_n = np.zeros(50)
n_vec = np.arange(100, 5001, 100)
for i in range(50):
    print(i)
    accuracy_vec_n[i] = accuracy(n_vec[i], 1)
plt.xlabel("n")
plt.ylabel("accuracy")
plt.plot(n_vec, accuracy_vec_n)
plt.show()
"""
print("\ndata[1].shape ---> ",data[1].shape)
print("\ndata.shape ---> ",data.shape)
print("\ndata.shape[0] ---> " ,data.shape[0])
print("\nlabels.shape ---> " , labels.shape)