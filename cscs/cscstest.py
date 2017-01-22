from __future__ import division
import math
import numpy
from scipy import random, linalg
from cscs import cscs

random.seed(5)
p = 5
n = 10
A = random.rand(p,p)
b = .5*A.max()
A[A<b]=0
Sigma = numpy.dot(A.transpose(),A)
mu = numpy.zeros(p)
Y = numpy.random.multivariate_normal(mu,Sigma,n)
cscs(Y,20,numpy.identity(p),100,0.0001)
