import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn import svm

mnist_path="D:\master_courses\pattern recgonition\course_homework\mnist_data/"

def load_mnist(path):
    #path:mnist data path
    mnist=fetch_openml('mnist_784',data_home=path)
    image=mnist['data']#shape:(70000,784)
    image=image/255
    label=mnist['target']#shape:(70000,)
    return image,label


#class SVM:
    #def __init__(self,):

    #def kernel():

if __name__=="__main__":
    svc=svm.SVC()
    image,label=load_mnist(mnist_path)
    train_image=image[:60000,:]
    test_image=image[60000:,:]
    train_label=label[:60000]
    test_label=label[60000:]
    svc.fit(train_image,train_label)