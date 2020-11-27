import cv2 as cv
import numpy as np
import os
import random
import matplotlib.pyplot as plt
def sigmoid(z):
    for i in range(len(z)) :
        for j in range(len(z[i])):
            z[i][j]=for_sigmoid(z[i][j])
    return z
def for_sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))
def d_sigmoid(z):
    return (1-sigmoid(z))*sigmoid(z)

def fc(w,a):
    z_next=np.dot(w,a)
    a_next=sigmoid(z_next)
    return [a_next,z_next]

def bc(w,z,delta_next):
    w_trans=np.transpose(w)
    delta=np.dot(w_trans,delta_next)*d_sigmoid(z)
    return delta
#step4: define cost function
def cost(a,y):
    J=1/2*sum((a-y)*(a-y))
    return J
#step5: define evaluation index
def accuracy(a,y):
    a_index=[]
    y_index=[]
    sum=0
    mini_batch = len(a)
    a=np.array(a)
    y=np.array(y)
    #每列最大axis=0  每行最大axis=1
    a_index.append(a.argmax(axis=1))
    a_index=np.array(a_index)
    y_index.append(y.argmax(axis=1))
    y_index=np.array(y_index)
    for i in range(0,len(a_index)):
        for j in range(0,len(a_index[i])):
            if(a_index[i][j]==y_index[i][j]):
                sum=sum+1
    #print(sum)
    return 1.0*sum/mini_batch
train_size=60000
test_size=10000
train_of_img=[]
test_of_img=[]
train_labels=[]
test_labels=[]
label=[0,0,0,0,0,0,0,0,0,0]
J = []
Acc = []
max_epoch=10
mini_batch = 100
a=[]
y=[]
z=[]
alpha=0.001
delta=[]
z.append([])
a.append([])
L=5
layer_size=np.array([784, 256, 128,64,10])
w=[]
test_list=np.load("test_list.npy")
train_list=np.load("train_list.npy")
test_labels=np.load("test_labels.npy")
train_labels=np.load("train_labels.npy")
w.append(np.load("w0.npy"))
w.append(np.load("w1.npy"))
w.append(np.load("w2.npy"))
w.append(np.load("w3.npy"))
for iter in range(0,max_epoch):
    idx = list(range(train_size))
    random.shuffle(idx)
    for i in range(0,int(train_size/mini_batch)):
        start_idx = i*mini_batch
        end_idx=(i+1)*mini_batch
        a=[[]]
        y=[]
        delta=[]
        a[0].append(train_list[idx[start_idx:end_idx],:])
        y.append(train_labels[idx[start_idx:end_idx],:])
        a[0]=(np.reshape(np.array(a[0]),(-1,layer_size[0]))).T#784*100
        y=(np.reshape(y,(-1,layer_size[L-1]))).T#10*100
        
        #forward computation
        for l in range(0,L-1):
            [a_temp,z_temp]=fc(w[l],a[l])
            a.append(a_temp)
            z.append(z_temp)
            #print(len(a_temp))
        #compute delta of last layer
        delta.append(np.array((a[L-1]-y)*a[L-1]*(1-a[L-1]) ))#10*100
        #a[L-1]*(1-a[L-1])=f'(z[L-1])=f(z[L-1])*(1-f(z[L-1])
        #backward computation
        for l in reversed(range(1,L-1)):#3,2,1
            dtemp=bc(w[l],z[l],delta[L-2-l])
            delta.append(dtemp)
        # update weight
        for l in range(0,L-1):#0,1,2,3
            grad_w=np.dot(delta[L-l-2],a[l].T)#3,2,1,0
            w[l]=w[l]-alpha*grad_w
        # training cost on training batch
        J.append(1/mini_batch*sum(cost(a[L-1],y)))
        Acc.append(accuracy(a[L-1].T,y.T))
        print("J:%10.7f"%J[-1]+'\n'+"Acc:%10.7f"%Acc[-1]+'\n')
        plt.plot(Acc,label='Acc')
        plt.plot(J,label='J')
        plt.legend()
        plt.show()
        # print("----小循环完----")
    np.save("w0.npy",w[0])
    np.save("w1.npy",w[1])
    np.save("w2.npy",w[2])
    np.save("w3.npy",w[3])
    print("----大循环完----")
 # step7 test the network
 # test on training set
# a[0]=train_list
# for l in range(0,L-1):
#     [a_temp,z_temp]=fc(w[l],a[l])
#     a.append(a_temp)
# train_acc=accuracy(a[L-1].T,train_labels.T)
# print("Accuracy on training dataset is:")
# print(train_acc)

 # test on testing set   
            
