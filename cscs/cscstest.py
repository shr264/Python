from __future__ import division
import math
import numpy
from scipy import random, linalg
from cscs import cscs


p = 5
n = 7
A = random.rand(p,p)
Sigma = numpy.dot(A,A.transpose())
mu = numpy.zeros(p)
Y = numpy.random.multivariate_normal(mu,Sigma,n)
cscs(Y,2,numpy.identity(p),100,0.0001)
