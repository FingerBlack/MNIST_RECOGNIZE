import cv2 as cv
import numpy as np
import os
import random

def change_dimention(image):
    input=[]
    for img in image:#遍历所有图片
        input_column=[]
        for im in img:#im是某一张图片的某个向量 28个数字
            input_column.extend(im)
        input.append(input_column)
    return input
def read_directorty(directory_name):
    for filename in os.listdir(r"D:/BaiduNetdiskDownload/MNIST_dataset/MNIST_dataset/"+directory_name):

        img=cv.imread("D:/BaiduNetdiskDownload/MNIST_dataset/MNIST_dataset/"+directory_name+"/"+filename)

        gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        np_img = np.array(gray_img)
        if directory_name[0:5]=="train":
            train_of_img.append(np_img)
            temp=label.copy()
            temp[int(directory_name[6:7])]=1
            train_labels.append(temp)
        else:
            test_of_img.append(np_img)
            temp=label.copy()
            temp[int(directory_name[5:6])]=1
            test_labels.append(temp)
#step1:data preparation
train_size=60000
test_size=10000
train_of_img=[]
test_of_img=[]
train_labels=[]
test_labels=[]
label=[0,0,0,0,0,0,0,0,0,0]
for i in range(0,10):
    s="train/"+str(i)
    read_directorty(s)
    #print("第"+str(i)+"次"+str(len(array_of_img)))
    print("训练数据"+str(i)+"完")
print("-------")
for i in range(0,10):
    s="test/"+str(i)
    read_directorty(s)
    print("测试数据"+str(i)+"完")
test_list=(np.array(change_dimention(test_of_img)))
train_list=(np.array(change_dimention(train_of_img)))
test_labels=(np.array(test_labels))
train_labels=(np.array(train_labels))
np.save('test_list.npy',test_list)
np.save('train_list.npy',train_list)
np.save('test_labels.npy',test_labels)
np.save('train_labels.npy',train_labels)

