import os, sys, pickle
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
from PIL import Image
import spams
import MyTools
import gc

import warnings
warnings.filterwarnings("ignore")

elad = sio.loadmat('./globalTrainedDictionary.mat')
D0 = elad['currDictionary']

n, p = D0.shape
b = int(np.sqrt(n))
N = 4096
ms = [int(n*r)/2*2 for r in [.2, .35, .5]]


lams = [.001, .005, .01, .05]
T = 40
k = .95
alpha = .1

eps = np.zeros(N)
I = np.eye(n)

for trial in range(5):

    Psi = []
    gc.collect()
    for j in range(N):
        Psi.append(I[np.random.permutation(n)])

    img_list = [it for it in os.listdir('./images/') if it.endswith('.png')]
    for img_test in img_list:
        imx = Image.open('./images/' + img_test).convert('L')
        x = np.asarray(imx).astype(float)
        x /= 255.0
        X = MyTools.im2col(x,b)
        Xmean = np.mean(X,axis=0)
        X -= Xmean

        for m in ms:
            pth = os.path.join('.','out_inpainting',img_test[:-4],'m%d'%m)
            if not os.path.exists(pth):
                os.makedirs(pth)
            elif os.path.exists(os.path.join(pth,'t{0}.pickle'.format(trial))):
                continue
            print 'trial={0}, m={1}, image={2}'.format(trial,m,img_test)

            PSNR = np.zeros((T,len(lams)))
            Verr = np.zeros((T,len(lams)))
            PSNR0 = np.zeros(len(lams))
            Verr0 = np.zeros(len(lams))

            for l in range(len(lams)):
                D = D0.copy()
                for j in range(N):
                    Phi = Psi[j][range(m)]
                    Vhi = np.mean(Psi[j][range(m,n)],axis=0)
                    yj = np.asfortranarray(Phi.dot(X[:,j]).reshape(m,1))
                    aj = spams.lasso(yj,D=np.asfortranarray(Phi.dot(D)),lambda1=lams[l])
                    xj = D.dot(aj.toarray()).squeeze()
                    eps[j] = np.sum(np.power(yj.squeeze() - Phi.dot(xj),2))
                    PSNR0[l] += np.power(X[:,j]-xj,2).mean()
                    Verr0[l] += np.power(Vhi.dot(X[:,j]-xj),2).mean()
                PSNR0[l] = -10 * np.log10(PSNR0[l] /N)
                print img_test, m, lams[l], PSNR0[l], Verr0[l]

                if lams[l] < .01:
                    Verr[:,l] = np.inf
                    continue

                for i in range(T):

                    Grad = 0
                    A = []
                    for j in range(N):
                        reord = np.random.permutation(range(m))
                        Phi1 = Psi[j][reord[:m/2]]
                        Phi2 = Psi[j][reord[m/2:]]
                        yj1 = np.asfortranarray(Phi1.dot(X[:,j]).reshape(m/2,1))
                        yj2 = np.asfortranarray(Phi2.dot(X[:,j]).reshape(m/2,1))
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
                        Vhi = np.mean(Psi[j][range(m,n)],axis=0)
                        yj = np.asfortranarray(Phi.dot(X[:,j]).reshape(m,1))
                        aj = spams.lasso(yj,D=np.asfortranarray(Phi.dot(D)),mode=1,lambda1=eps[j])
                        xj = D.dot(aj.toarray()).squeeze()
                        eps[j] = np.sum(np.power(Phi.dot(X[:,j]-xj),2))
                        PSNR[i,l] += np.power(X[:,j]-xj,2).mean()
                        Verr[i,l] += np.power(Vhi.dot(X[:,j]-xj),2).mean()
                    eps *= k
                    PSNR[i,l] = -10 * np.log10(PSNR[i,l] /N)
                    print i, PSNR[i,l], Verr[i,l]

            with open(os.path.join(pth,'t{0}.pickle'.format(trial)), 'w') as f:
                pickle.dump([PSNR, Verr, PSNR0, Verr0], f)














