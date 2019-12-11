import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

mnist_path="D:\master_courses\pattern recgonition\course_homework\mnist_data/"

def load_mnist(path):
    #path:mnist data path
    mnist=fetch_openml('mnist_784',data_home=path)
    image=mnist['data']#shape:(70000,784)
    image=image/255
    label=mnist['target']#shape:(70000,)
    return image,label

def knn(train_image,train_label,test_image,test_label,k):
    #input:(60000,784)/(60000,)/(10000,784) type:ndarray
    predict=np.zeros((np.shape(test_image)[0],k))
    for i in range(np.size(test_image,0)):
        knn_list=[]
        for j in range(np.shape(train_image)[0]):
            knn_list.append(np.linalg.norm(train_image[j,:]-test_image[i,:]))
        print('-----------------------------------')
        index_list=sorted(range(len(knn_list)),key=lambda p:knn_list[p])[:k]
        predict[i,:]=np.array([train_label[index_list]])
       
        print("test image id:%d label:%s prediction:"%(i+1,test_label[i]))
        print(predict[i,:])
    #print(predict)
    return predict

if __name__=="__main__":
    image,label=load_mnist(mnist_path)
    train_image=image[:60000,:]
    test_image=image[60000:,:]
    train_label=label[:60000]
    test_label=label[60000:]
    #neigh=KNeighborsClassifier(n_neighbors=10)
    #neigh.fit(train_image,train_label)
    #print(test_image[:10])
    #print(neigh.predict(test_image[:10]))
    #print(test_label[:10])
    predict=knn(train_image,train_label,test_image,test_label,3)