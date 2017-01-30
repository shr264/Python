from __future__ import division
import pylab
import networkx as nx
import math
import matplotlib
import numpy as np
from scipy import random, linalg
from cscs import cscs

random.seed(5)
p = 5
n = 10
A = random.rand(p,p)
b = .5*A.max()
A[A<b]=0
Sigma = np.dot(A.transpose(),A)
mu = np.zeros(p)
Y = np.random.multivariate_normal(mu,Sigma,n)
L = cscs(Y,50,np.identity(p),100,0.0001)
L[L<0.001] = 0
adj = abs(L)>0
adj = adj.astype(int)
G = nx.Graph(adj)
nx.draw(G, with_labels=True)
pylab.show()
