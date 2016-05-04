#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: yuqing5
# date: 20151014

import numpy

class Percton(object):
    def  __init__(self,w,b,alpha):
        self.w = w
        self.b = b
        self.alpha = alpha

    def loss(self, x, y):
        return np.sum(y*(np.dot(x,self.w) + self.b)) 

    def sgd(self,x,y):
        self.w += self.alpha*y*x
        self.b += self.alpha*y

    def train(self,X,Y):
        while True:
            M = len(X)  #错误分类数
            for i in range(len(X)):
                if self.loss(X[i],Y[i]) <=0:
                    self.sgd(X[i],Y[i])
                    print self.w,self.b
                else:
                    M -= 1
            if not M:
                print "final para:"
                break


class Perceptron_dual(object):
    def __init__(self, alpha,b, ita):
        self.alpha = alpha
        self.b = b
        self.ita = ita

    def gram(self,X):
        return np.dot(X,X.T)


    def train(self,X,Y):
        g = self.gram(X)
        while True:
            M = len(X)
            for j in range(len(X)):
                if Y[j] * (np.sum(self.alpha*Y*g[j])+self.b) <=0:
                    self.alpha[i] += self.ita
                    self.b += self.ita * Y[i]
                else:
                    M -= 1
            if M == 0:
                print "final optimal"
                break


if __name__ == '__main__':
    X = np.array([[3,3],[4,3],[1,1]])
    Y = np.array([1,1,-1])
    perc_d = Perceptron_dual(np.zeros(Y.shape),0,1)
    perc_d.train(X, Y)