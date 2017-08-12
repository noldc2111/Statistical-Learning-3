# Casey Nold; nold@pdx.edu
# EM with Potts Adaptation
#

from PIL import Image
from scipy import misc
from numpy import unique
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal as mvnorm
import csv,copy,operator
from random import randint

def readCSV(fileName):
    vals = []
    with open(fileName,'r') as d:
        img_data = csv.reader(d,delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        for row in img_data:
            vals.append(row)
    return np.array(vals)

data = []
means = []
blocks = 'training_ss_149_blocks_python_labeled_coords.csv'
#x = 'training_ss_151_blocks_python.csv'
k_means = '149_kmeans.csv'

data = readCSV(blocks)
#d = readCSV(x)

means = readCSV(k_means)
# open the segmented image, this acts as the prior
im_seg = Image.open("../bruno_data/training_seg_149.png",'r')
im = mpimg.imread("../bruno_data/training_ss_149.png")
wid,hei = im_seg.size
# then extract the pixel values
pixvals = np.array(list(im_seg.getdata())).reshape(wid,hei,3)

J = .35
K = 3
n = len(data)      
p = len(data[1,:25])

d = np.zeros((n,p))
for i in range(0,n):
    d[i,] = data[i,:25]

mu_init = np.zeros((K,p))
mu_init[0,] = means[0]#[1:]
mu_init[1,] = means[1]#[1:]
mu_init[2,] = means[2]#[1:]

sigma_1 = sigma_2 = sigma_3 = np.identity(p)
sigma = [sigma_1,sigma_2,sigma_3]

v,c = unique(pixvals,return_counts=True)
v = v[1:]

labels = []
neb = []
for i in range(0,n):
    x,y = data[i,26:28]
    x = int(x)
    y = int(y)
    neb.append(pixvals[(x-1):(x+2),(y-1):(y+2),1])
    labels.append(pixvals[x,y,1])

one = 0
two = 0
three = 0

for i in range(0,n):
    if(labels[i] == v[0]):
        one += np.exp(J)
    elif(labels[i] == v[1]):
        two += np.exp(J)
    else:
        three += np.exp(J)

Z = one+two+three

## refactoring
potentials = np.zeros((n))
d_one = []
d_two = []
d_three = []

for i in range(0,n):
    nn = list(neb[i].flatten())
    y_i = nn.pop(4)
    potential = 0
    for each in nn:
        if(each == y_i):
            potential += np.exp(J)
    potentials[i] = potential/Z
    if(y_i == v[0]):
        d_one.append(d[i])
    elif(y_i == v[1]):
        d_two.append(d[i])
    elif(y_i == v[2]):
        d_three.append(d[i])

## end refactoring
#pi = [0,0,0]
r = np.zeros((n,K))
R = np.sum(r,axis=0)

# initialize sigma
for k in range(0,K):
        h = 0
        g = 0
        for i in range(0,n):
            h+= r[i,k]*(d[i,]*d[i,].transpose())
        sigma[k] = np.diag(np.abs(g - (mu_init[k]*mu_init[k].transpose())))

sentry = 0
ll = []

while(sentry < 4):
    
    # E
    for i in range(0,n):
        b= (potentials[i]*mvnorm.pdf(d[i],mu_init[0],sigma[0]) +
            potentials[i]*mvnorm.pdf(d[i],mu_init[1],sigma[1]) +
            potentials[i]*mvnorm.pdf(d[i],mu_init[2],sigma[2]))
        if(b == 0):
            b= (potentials[i]*mvnorm.pdf(d[i],mu_init[0],sigma[0]) +
                potentials[i]*mvnorm.pdf(d[i],mu_init[1],sigma[1]) +
                potentials[i]**mvnorm.pdf(d[i],mu_init[2],sigma[2]))
        for k in range(0,K):
            r[i,k] = (potentials[i]*mvnorm.pdf(d[i],mu_init[k],sigma[k]))

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
   
    #ll_prior = 0
    #for i in range(0,n):
    #    for k in range(0,K):
            #ll_prior += r[i,k]*np.log(pi[k])

    ll_param = 0
    for k in range(0,K):
        for i in range(0,n):
            ll_param += r[i,k]*mvnorm.logpdf(d[i,],mu_init[k],sigma[k]) #

    #print(ll_param + ll_prior)
    print(ll_param)
    #ll.append((ll_param+ll_prior))
    ll.append((ll_param))

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
 



