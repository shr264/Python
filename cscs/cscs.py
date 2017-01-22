#### This is the CSCS algorithm by Khare, Oh, Rahman and Rajaratnam. Please look at accompanying file for an example of how to use

from __future__ import division
import math
import numpy
from scipy import random, linalg

def softthresh(x,lmbda):
    out = numpy.sign(x)*(math.fabs(x)-lmbda)
    return out

def Tj(lmbda,A,x,j,k):
    out = softthresh(sum(A[numpy.arange(0,k+1)!=j,j]*x[numpy.arange(0,k+1)!=j]),lmbda)/(2*A[j,j])
    return out

def Tk(A,x,k):
    temp = sum(A[numpy.arange(0,k+1)!=k,k]*x[numpy.arange(0,k+1)!=k])
    out = (-temp + math.sqrt(math.pow(temp,2)+4*A[k,k]))/(2*A[k,k])
    return out

def h(k,A,lmbda,x0,maxitr,eps):
    itr = 1
    diff = 1
    while((diff>eps) and (itr<maxitr)):
        x1 = x0
        for j in range(0,k):
            x1[j] = Tj(lmbda,A,x1,j,k)
        x1[k] = Tk(A,x1,k)
        diff = max(numpy.absolute(x1-x0))
        itr += 1
    return x1

def cscs(Y,lmbda,initOm,maxitr,eps):
    n = Y.shape[0]
    p = Y.shape[1]
    S = (1/n)*numpy.dot(Y.transpose(),Y)
    L = numpy.identity(p)
    L[0,0] = 1/math.sqrt(S[0,0])
    for i in range(1,p):
        L[i,numpy.arange(0,i+1)] = h(i,S[0:(i+1),0:(i+1)],lmbda,initOm[i,numpy.arange(0,i+1)],maxitr,eps)
    return L



    


    
    
