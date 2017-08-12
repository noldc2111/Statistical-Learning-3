# Casey Nold; nold@pdx.edu
# Simulated K-means 
#

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnorm
import csv
import numpy as np
from operator import itemgetter

def readCSV(fileName):
    vals = []
    with open(fileName,'r') as d:
        img_data = csv.reader(d,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        for row in img_data:
            vals.append(row)
    return np.array(vals)

fn = "151_kmeans.csv"

n =  11825
p = 25
K = 3

pi = np.array([1]*K)/K
Y = np.array([0,1,2])
y= np.random.choice(Y,n,True,pi)

x = readCSV(fn)
mu = np.zeros((K,p))
for k in range(0,K):
    mu[k] = x[k,]

m = np.array([np.random.normal(.5,.5,(K*p))]).reshape((K,p))


plt.plot(np.mean(mu[1,]),'r',np.mean(mu[2,]),'b',np.mean(mu[0,]),'c')
plt.show()

s_1=s_2=s_3 = np.identity(p)
sigma = [s_1,s_2,s_3]

d = np.zeros((n,p))
r = np.zeros((n,K))

for i in range(0,n):
    d[i,] = np.random.multivariate_normal(mu[y[i]],sigma[y[i]])

r_new = np.array([[1]]*n)
dist = np.zeros((K))

#plt.imshow(d)
#plt.show()

t = 0
quack = True
while(quack):
    
   r = r_new
   for i in range(0,n):
       for k in range(0,K):
           dist[k] = sum((d[i,]- m[k])**2)
       r_new[i] = np.argmin(dist)

   for k in range(0,K):
       h = []
       for i in range(0,n):
           if(r_new[i] == k):
               h.append(d[i])
       if(h):
           m[k,] = np.mean(h,axis=0)

   t +=1
   if(t > 50):
       quack = False

plt.plot(np.mean(m[1,]),'r',np.mean(m[2,]),'b',np.mean(m[0,]),'c')
plt.show()

#plt.plot(d,'x')
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(d,'x')
#ax1.plot(m,'bo')
#ax1.plot(mu,'ro')
#plt.show()


