import cv2 as cv
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat
# define the sigmoid function
def sigmoid(s) :
    return 1 / (1 + np.exp(-s))


# derivative of sigmoid  function
def d_sigmoid (s):
    return sigmoid(s) * (1-sigmoid(s))
def fc(w,a):
    z_next=np.dot(w,a)
    a_next=sigmoid(z_next)
    return a_next,z_next
def bc(w,z,delta_next):
    w_trans=np.transpose(w)
    delta=np.dot(w_trans,delta_next)*d_sigmoid(z)
    return delta
#step4: define cost function
def cost(a,y):
    J = 1/2 * np.sum((a - y)**2)
    return J
#step5: define evaluation index
def accuracy(a,y):
    mini_batch = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc
train_size=10000
test_size=2000
train_labels=[]
test_labels=[]
label=[0,0,0,0,0,0,0,0,0,0]
J = []
Acc = []
max_epoch=100
mini_batch = 100
alpha=0.05
delta=[]
L=5
layer_size=np.array([784, 256, 128,64,10])
w=[]
m = loadmat("./mnist_small_matlab.mat")
trainData, train_labels = m['trainData'], m['trainLabels']
testData, test_labels = m['testData'], m['testLabels']
train_list = trainData.reshape(-1, train_size)
test_list = testData.reshape(-1, test_size)
print(train_list.shape, test_list.shape)

# J=list(np.load("J.npy"))
# Acc=list(np.load("Acc.npy"))
#
# w.append(np.load("w0.npy"))
# w.append(np.load("w1.npy"))
# w.append(np.load("w2.npy"))
# w.append(np.load("w3.npy"))
a=[np.ones((784,mini_batch),dtype=float),np.ones((256,mini_batch),dtype=float),np.ones((128,mini_batch),dtype=float),np.ones((256,mini_batch),dtype=float),np.ones((10,mini_batch),dtype=float)]
y=[np.ones((10,mini_batch),dtype=float)]
z=[np.ones((784,mini_batch),dtype=float),np.ones((256,mini_batch),dtype=float),np.ones((128,mini_batch),dtype=float),np.ones((256,mini_batch),dtype=float),np.ones((10,mini_batch),dtype=float)]
delta=[np.ones((784,mini_batch),dtype=float),np.ones((256,mini_batch),dtype=float),np.ones((128,mini_batch),dtype=float),np.ones((64,mini_batch),dtype=float),np.ones((10,mini_batch),dtype=float)]
for iw in range(1,L):
    w.append(0.1*np.random.randn(layer_size[iw],layer_size[iw-1]))
for iter in range(0,max_epoch):
    idx = np.random.permutation(train_size)
    for i in range(0,int(train_size/mini_batch)):
        start_idx = i*mini_batch
        end_idx=(i+1)*mini_batch
        a[0]=(train_list[:,idx[start_idx:end_idx]])
        y=(train_labels[:,idx[start_idx:end_idx]])
        #784*100
        #forward computation
        for l in range(0,L-1):
            a[l+1], z[l+1] = fc(w[l], a[l])
            #print(len(a_temp))
        #compute delta of last layer
        delta[L-1] = (a[L-1] - y) * (a[L-1] * (1 - a[L-1]))
        #a[L-1]*(1-a[L-1])=f'(z[L-1])=f(z[L-1])*(1-f(z[L-1])
        #backward computation
        for l in range(L - 2, 0, -1):
            # print('aaa')
            delta[l] = bc(w[l], z[l], delta[l + 1])
        # update weight
        for l in range(0, L-1):
            grad_w = np.dot(delta[l + 1], a[l].T)
            w[l] = w[l] - alpha * grad_w
        # training cost on training batch
        J.append(1/mini_batch*cost(a[L-1],y))
        Acc.append(accuracy(a[L-1].T,y.T))
        # print("----小循环完----")
    a = {}
    a[0]=test_list
    for l in range(0, L - 1):
        a[l+1], z[l+1] = fc(w[l], a[l])
    test_acc = accuracy(a[L - 1].T, test_labels.T)
    print("J:%10.7f" % J[-1] + '\n' + "Acc:%10.7f" % Acc[-1])
    print("Accuracy on Test dataset is:")
    print("Test_Acc:%10.7f\n" % test_acc)
print("----大循环完----")
plt.plot(Acc,label='Acc')
plt.plot(J,label='J')
plt.legend()
plt.show()
np.save("w0.npy",w[0])
np.save("w1.npy",w[1])
np.save("w2.npy",w[2])
np.save("w3.npy",w[3])
np.save("J.npy",J)
np.save("Acc.npy",Acc)
# step7 test the network
#  test on testing set

 #
 # test on testing set

