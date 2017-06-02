import csv
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import operator

 
data = []
means = []
blocks = 'training_ss_151_blocks_python_labeled_coords.csv'
#blocks = 'training_ss_151_blocks_python_labeled.csv'
x = 'training_ss_151_blocks_python.csv'
k_means = 'kmeans_python.csv'
im = mpimg.imread("training_ss_151.png")

def readCSV(fileName):
    vals = []
    with open(fileName,'r') as d:
        img_data = csv.reader(d,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        for row in img_data:
            vals.append(row)
    return np.array(vals)

data = readCSV(blocks)
d = readCSV(x)
means = readCSV(k_means)

n = len(d)
p = len(d[1])
K = 3
mu_init = np.zeros((K,p))
mu_init[0,] = means[0][1:]
mu_init[1,] = means[1][1:]
mu_init[2,] = means[2][1:]

pi = np.array([1,1,1])/K

sigma_1 = sigma_2 = sigma_3 = np.identity(p)
sigma = [sigma_1,sigma_2,sigma_3]

r = np.zeros((n,K))
R = np.sum(r,axis=0)

# initialize sigma
for k in range(0,K):
        h = 0
        g = 0
        for i in range(0,n):
            h+= r[i,k]*(d[i,]*d[i,].transpose())
        sigma[k] = np.diag(np.abs(g - (mu_init[k]*mu_init[k].transpose())))

# initialize responsibility matrix
b = 0
for i in range(0,n):
    b= (pi[0]*mvnorm.pdf(d[i],mu_init[0],sigma[0]) +
        pi[1]*mvnorm.pdf(d[i],mu_init[1],sigma[1]) +
        pi[2]*mvnorm.pdf(d[i],mu_init[2],sigma[2]))
    for k in range(0,K):
        r[i,k] = (pi[k]*mvnorm.pdf(d[i],mu_init[k],sigma[k]))/b

sentry = 0
ll = []

while(sentry < 25):
    
    # E
    for i in range(0,n):
        b= (pi[0]*mvnorm.pdf(d[i],mu_init[0],sigma[0]) +
            pi[1]*mvnorm.pdf(d[i],mu_init[1],sigma[1]) +
            pi[2]*mvnorm.pdf(d[i],mu_init[2],sigma[2]))
        for k in range(0,K):
            r[i,k] = (pi[k]*mvnorm.pdf(d[i],mu_init[k],sigma[k]))/b


    R = np.sum(r,axis=0)
    pi = R/n

    # mu update
    for k in range(0,K):
        lh = 0
        if(R[k] != 0):
            for i in range(0,n):
                lh+=(r[i,k]*d[i,])
            mu_init[k,] = (lh/R[k])


    # sigma update. 
    for k in range(0,K):
        h = 0
        g = 0
        for i in range(0,n):
            h+= r[i,k]*(d[i,]*d[i,].transpose())
        g = h/R[k]
        sigma[k] = np.diag(np.abs(g - (mu_init[k]*mu_init[k].transpose())))
   
    ll_prior = 0
    for i in range(0,n):
        for k in range(0,K):
            ll_prior += r[i,k]*np.log(pi[k])

    ll_param = 0
    for k in range(0,K):
        for i in range(0,n):
            ll_param += r[i,k]*mvnorm.logpdf(d[i,],mu_init[k],sigma[k]) #

    print(ll_param + ll_prior)
    ll.append((ll_param+ll_prior))

    print(sentry)
    sentry += 1
plt.plot(ll)
plt.show()


## visualize...

im2 = im

c = np.zeros((n,p+1))
d = np.zeros((n,p))
for i in range(0,n):
    index, value = max(enumerate(r[i,]), key=operator.itemgetter(1))
    c[i,] = np.append(d[i,],(index+1))

for i in range(0,n):
    if(c[i,25] == 1):
        im2[data[i,26],data[i,27],] = np.mean( mu_init[0])
    elif c[i,25] == 2:
        im2[data[i,26],data[i,27],] = np.mean( mu_init[1])
    else:
        im2[data[i,26],data[i,27],] = np.mean( mu_init[2])

plt.imshow(im2)
plt.show()
 



