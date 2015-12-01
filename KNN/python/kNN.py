# -*- coding:UTF-8 -*-
#Filename:kNN.py
from  numpy import *
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1],[5,5],[5.1,4.9]])
	labels = ['A','A','B','B','C','C']
	return group,labels


def file2matrix(filename):
    with open(filename, 'r') as f1:  
        arrayOfLines = f1.readlines()
        numberOfLines = len(arrayOfLines)
        returnMat = zeros((numberOfLines,3))
        classLabelVetor = []
        index =0
    for line in arrayOfLines:
        line = line.split('\t')
        #listFromLine = line.strip()
        returnMat[index,:] = line[0:3]
        if (line[3].strip().decode('utf-8')=='largeDoses'):
            line[3] = 3
        elif (line[3].strip().decode('utf-8')=='smallDoses'):
            line[3]=2
        else:
            line[3]=1
        classLabelVetor.append(line[3])
        index +=1
    return returnMat,classLabelVetor
