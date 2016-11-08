import os, sys, pickle
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
from PIL import Image
import spams
import MyTools
import gc
import matplotlib.pyplot as plt
# import pdb
import warnings
warnings.filterwarnings("ignore")

# K-SVD off-the-shelf initialization
elad = sio.loadmat('./globalTrainedDictionary.mat')
D0 = elad['currDictionary']

# # DCT initialization
# D0 = MyTools.dct2mtx(8)

# # random Gaussian initialization
# D0 = np.random.randn(64,256)
# D0 = D0 / np.sqrt((D0**2).sum(axis=0))

n, p = D0.shape
b = int(np.sqrt(n))
N = 4096

m = int(n*.5)/2*2
lam = .01
k = .95

T = 100
alpha = .1

eps = np.zeros(N)
g_t = np.zeros(T)

Psi = []
gc.collect()
for j in range(N):
    phiQ, phiR = np.linalg.qr(np.random.randn(n,n))
    Psi.append(phiQ)

img_test = 'images/barbara.png'

imx = Image.open(img_test).convert('L')
x = np.asarray(imx).astype(float)
x /= 255.0
X = MyTools.im2col(x,b)
X_clean = np.array(X)
X[X>1] = 1.
X[X<0] = 0.
Xmean = np.mean(X,axis=0)
X -= Xmean
X_clean -= Xmean

PSNR = np.zeros(T)
Verr = np.zeros(T)
PSNR0 = 0
Verr0 = 0

D = D0.copy()
for j in range(N):
    Phi = Psi[j][range(m)]
    Vhi = np.mean(Psi[j],axis=0)
    yj = np.asfortranarray(Phi.dot(X[:,j]).reshape(m,1))
    aj = spams.lasso(yj,D=np.asfortranarray(Phi.dot(D)),lambda1=lam)
    xj = D.dot(aj.toarray()).squeeze()
    eps[j] = np.sum(np.power(yj.squeeze() - Phi.dot(xj),2))
    PSNR0 += np.power(X_clean[:,j]-xj,2).mean()
    Verr0 += np.power(Vhi.dot(X[:,j]-xj),2).mean()
PSNR0 = -10 * np.log10(PSNR0 /N)
print 'image={0}, m={1}, lambda={2}, PSNR={3}, V-error={4}'.format(img_test, m, lam, PSNR0, Verr0)

for i in range(T):
    Grad = 0
    A = []
    for j in range(N):
        reord = np.random.permutation(range(m))
        Phi1 = Psi[j][reord[:m/2]]
        Phi2 = Psi[j][reord[m/2:]]
        yj1 = np.asfortranarray(Phi1.dot(X[:,j]).reshape(Phi1.shape[0],1))
        yj2 = np.asfortranarray(Phi2.dot(X[:,j]).reshape(Phi2.shape[0],1))
        aj1 = spams.lasso(yj1,D=np.asfortranarray(Phi1.dot(D)),mode=1,lambda1=eps[j])
        A.append(aj1)
        xj1 = D.dot(aj1.toarray())
        Grad += (Phi2.T).dot(yj2 - Phi2.dot(xj1)).dot(aj1.toarray().T)

    # Q = 0
    # for j in range(N):
    #     Phi = Psi[j][range(m)]
    #     aj = A[j].toarray()
    #     Q += (Phi.T).dot(Phi).dot(Grad).dot(aj).dot(aj.T)
    # alpha = np.trace((Grad.T).dot(Grad)) / np.trace((Grad.T).dot(Q))

    D += Grad * alpha
    D /= np.sqrt(np.sum(np.power(D,2))) / np.sqrt(p)

    for j in range(N):
        Phi = Psi[j][range(m)]
        Vhi = np.mean(Psi[j],axis=0)
        yj = np.asfortranarray(Phi.dot(X[:,j]).reshape(m,1))
        aj = spams.lasso(yj,D=np.asfortranarray(Phi.dot(D)),mode=1,lambda1=eps[j])
        xj = D.dot(aj.toarray()).squeeze()
        eps[j] = np.sum(np.power(Phi.dot(X[:,j]-xj),2))
        PSNR[i] += np.power(X_clean[:,j]-xj,2).mean()
        Verr[i] += np.power(Vhi.dot(X[:,j]-xj),2).mean()
    g_t[i] = eps.sum() * n / m / N
    eps *= k
    PSNR[i] = -10 * np.log10(PSNR[i] /N)
    print 'iter={0}, g_t={1}, PSNR={2}, V-error={3}'.format(i, g_t[i], PSNR[i], Verr[i])

with open('curve','w') as fw:
    pickle.dump([PSNR,Verr,PSNR0,Verr0,g_t,D],fw)










