#!E:\machinelearning\KNN\python
# -*- coding:UTF-8 -*-
import kNN 
from numpy import array
import matplotlib
import matplotlib.pyplot as plt

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.xlabel("play games time %")
plt.ylabel("eat icecream L")
plt.title("First Example")
#plt.legend()
plt.show()
#fig.savefig("./test.png")