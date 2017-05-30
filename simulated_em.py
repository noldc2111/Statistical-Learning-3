from scipy.stats import multivariate_normal as mvnorm
import numpy as np
import matplotlib.pyplot as plt


K = 3
n = 20
p = 2

# mu for our simulation
mu = np.zeros((K,p))
mu[0,] = [1,5] 
mu[1,] = [5,5]
mu[2,] = [5,1]

# mu as initial params
mu_init = np.zeros((K,p))
mu_init[0,] = [1,1]
mu_init[1,] = [3,3]
mu_init[2,] = [5,5]

pi = np.array([1,1,1])/K

Y = np.array([0,1,2])
y = np.random.choice(Y,n,True,pi)

sig_1 = sig_2 = sig_3 = np.identity(p)
sigma = [sig_1,sig_2,sig_3]
d = np.zeros((n,p))
r = np.zeros((n,K))

## create the data
for i in range(0,n):
    d[i,] = np.random.multivariate_normal(mu[y[i]],sigma[y[i]])

#plt.plot(d,'x')
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.plot(d,'x')
#ax1.plot(mu,'bo')
#plt.show()

## initalize responsibility matrix
b = 0
for i in range(0,n):
    b = (pi[0]*mvnorm.pdf(d[i,],mu_init[0,],sigma[0]) +
          pi[1]*mvnorm.pdf(d[i,],mu_init[1,],sigma[1])+
          pi[2]*mvnorm.pdf(d[i,],mu_init[2,],sigma[2]))
    for k in range(0,K):
        r[i,k] = (pi[k]*mvnorm.pdf(d[i,],mu_init[k,],sigma[k]))/b


R = np.sum(r,axis=0)

## Initialize sigma
for k in range(0,K):
    h = 0
    g = 0
    for i in range(0,n):
        h+= r[i,k]*(d[i,]*d[i,].transpose())
        g = h/R[k]
    sigma[k] = np.diag(np.abs(g - (mu_init[k]*mu_init[k].transpose())))
    #sigma[k] = np.diag(np.abs(h - (mu_init[k]*mu_init[k].transpose())))
 

sentry = 0
ll = []
while(sentry < 29):

    ## calculate responsibilties
    b = 0
    for i in range(0,n):
        b = (pi[0]*mvnorm.pdf(d[i,],mu_init[0,],sigma[0]) +
             pi[1]*mvnorm.pdf(d[i,],mu_init[1,],sigma[1]) +
             pi[2]*mvnorm.pdf(d[i,],mu_init[2,],sigma[2]))
        for k in range(0,K):
            r[i,k] = (pi[k]*mvnorm.pdf(d[i,],mu_init[k,],sigma[k]))/b
    
    R = np.sum(r,axis=0)

    #update pi
    pi = R/n

    #update mu
    for k in range(0,K):
        lh = 0
        for i in range(0,n):
            lh += (r[i,k]*d[i,])
        mu_init[k,] = (lh/R[k])

    #update sigma
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

    ll.append((ll_param+ll_prior))

    sentry += 1
    
    #print(mu_init,'\n')
    print(ll_param+ll_prior)
    print(sentry,'\n')


plt.plot(ll)
plt.show()
