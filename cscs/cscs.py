#### This is the CSCS algorithm by Khare, Oh, Rahman and Rajaratnam. Please look at accompanying file for an example of how to use

from __future__ import division
import math
import numpy
from scipy import random, linalg
import copy

def softthresh(x,lmbda):
    out = numpy.sign(x)*(math.fabs(x)-lmbda)
    return out

def Tj(lmbda,A,x,j,k):
    out = softthresh(-2*sum(A[numpy.arange(0,k)!=(j-1),(j-1)]*x[numpy.arange(0,k)!=(j-1)]),lmbda)/(2*A[j-1,j-1])
    return out

def Tk(A,x,k):
    temp = sum(A[numpy.arange(0,k)!=(k-1),(k-1)]*x[numpy.arange(0,k)!=(k-1)])
    out = (-temp + math.sqrt(math.pow(temp,2)+4*A[k-1,k-1]))/(2*A[k-1,k-1])
    return out

def h(k,A,lmbda,x0,maxitr,eps):
    itr = 1
    diff = 1
    x1 = numpy.zeros(numpy.size(x0))
    while((diff>eps) and (itr<maxitr)):
        for j in range(1,k):
            x1[j-1] = Tj(lmbda,A,x1,j,k)
        x1[k-1] = Tk(A,x1,k)
        diff = max(numpy.absolute(x1-x0))
        itr += 1
        x0 = copy.copy(x1)
        print 'diff = ', diff, '\n'
        print 'itr = ', itr, '\n'
    return x1

def cscs(Y,lmbda,initOm,maxitr,eps):
    n = Y.shape[0]
    p = Y.shape[1]
    S = (1/n)*numpy.dot(Y.transpose(),Y)
    L = numpy.identity(p)
    L[0,0] = 1/math.sqrt(S[0,0])
    for i in range(1,p):
        L[i,numpy.arange(0,i+1)] = h(i+1,S[0:(i+1),0:(i+1)],lmbda,initOm[i,numpy.arange(0,i+1)],maxitr,eps)
    return L



    


    
    
